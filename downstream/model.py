import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, to_dense_batch, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from subgraph_model import GNNSubgraph
import math

# from torch.nn.parameter import Parameter

import numpy as np
import random

from utils import drop_nodes


num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, aggr = "add"):
        super(GINConv, self).__init__()
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        self.atom_num = x.shape[0]
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr, edge_index):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):

    def __init__(self, emb_dim, aggr = "add"):
        super(GCNConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_index = edge_index[0]
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr = "add"):
        super(GATConv, self).__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, heads * emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_index = edge_index[0]
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):

        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        norm = self.norm(edge_index, x.size(0), x.dtype)

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr = "mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_index = edge_index[0]
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p = 2, dim = -1)



class GNN(torch.nn.Module):
    """
    

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin", subgraph_branch=False):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        # self.subgraph_branch = subgraph_branch

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # if subgraph_branch:
            # self.x_embedding1 = torch.nn.Embedding(num_atom_type * 2, emb_dim)
        # else:
        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)
        # if subgraph_branch:
        #     self.x_embedding3 = torch.nn.Embedding(2, emb_dim)
        #     torch.nn.init.xavier_uniform_(self.x_embedding3.weight.data)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr = "add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        # if self.subgraph_branch:
        #     x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1]) + self.x_embedding3(x[:,2])
        # else:
        x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1]) 

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation
    

class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer, share, emb_dim, num_tasks, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.share = share
        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)
        self.sub_gnn = GNN(self.share, emb_dim, JK, drop_ratio, gnn_type = gnn_type, subgraph_branch=False)
        self.re_gnn = GNNSubgraph(num_layer - self.share, emb_dim, JK, drop_ratio, gnn_type = gnn_type)
        # self.re_gnn = GNN(2, emb_dim, JK, drop_ratio, gnn_type = gnn_type)

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1
        
        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim//2, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * emb_dim, self.num_tasks)
            self.graph_pred_linear2 = torch.nn.Linear(self.mult * emb_dim, self.num_tasks)
            self.graph_pred_linear3 = torch.nn.Linear(self.mult * emb_dim, self.num_tasks)
            # self.graph_pred_linear3 = torch.nn.Parameter(torch.Tensor(self.num_tasks, emb_dim))
            # torch.nn.init.kaiming_uniform_(self.graph_pred_linear3, a=math.sqrt(5))

        self.w_l = torch.nn.Linear(emb_dim , 1, bias=False)
        self.w_v = torch.nn.Linear(emb_dim , 1, bias=False)

        # self.scale = torch.nn.Parameter(torch.Tensor([1.]))
        # self.w_a = torch.nn.Linear(2 * emb_dim, emb_dim, bias=False)
        # self.head1 = torch.nn.Linear(emb_dim, emb_dim)
        # self.head2 = torch.nn.Linear(emb_dim, emb_dim)
        # self.head3 = torch.nn.Linear(emb_dim, emb_dim)
        # self.head4 = torch.nn.Linear(emb_dim, emb_dim)
        # self.head5 = torch.nn.Linear(emb_dim, emb_dim)
        # self.head6 = torch.nn.Linear(emb_dim, emb_dim)

    def from_pretrained(self, model_file):
        state_dict = torch.load(model_file, map_location='cpu')
        # ce_d = torch.load('/data00/jiahao/mole_supervise/Mole-BERT.pth', map_location='cpu')
        miss1, miss2 = self.gnn.load_state_dict(state_dict['gnn'], strict=False)
        # miss1, miss2 = self.gnn.load_state_dict(ce_d, strict=False)
        print(f'gnn load {miss1}, {miss2}')
        miss1, miss2 = self.sub_gnn.load_state_dict(state_dict['sub_gnn'], strict=False)
        print(f'sub gnn load {miss1}, {miss2}')
        miss1, miss2 = self.re_gnn.load_state_dict(state_dict['re_gnn'], strict=False)
        print(f're gnn load {miss1}, {miss2}')

        # miss1, miss2 = self.w_l.load_state_dict(state_dict['w_l'])
        # miss1, miss2 = self.w_v.load_state_dict(state_dict['w_v'])
        # miss1, miss2 = self.head3.load_state_dict(state_dict['head3'])
        # miss1, miss2 = self.head4.load_state_dict(state_dict['head4'])
        # miss1, miss2 = self.head5.load_state_dict(state_dict['head5'])
        # miss1, miss2 = self.head6.load_state_dict(state_dict['head6'])

    def gnn_forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)
        return node_representation

    def sub_gnn_forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr= data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")
        
        node_representation = self.sub_gnn(x, edge_index, edge_attr)
        # node_representation = self.gnn.sub_forward(x, edge_index, edge_attr, self.share)
        return node_representation

    def forward(self, argv1, argv2, re_graph):
        '''
        node_rep1: (N1, C), N1 is the number of molecules
        node_rep2: (N2, C), N2 is the number of the first principle graph
        '''
        batch_index = argv1[3].cpu()
        # batch_num = int(max(batch_index) + 1)
        node_rep1 = self.gnn_forward(argv1[0], argv1[1], argv1[2])
        node_rep2 = self.sub_gnn_forward(argv2[0], argv2[1], argv2[2])

        sub_batch_idx = torch.tensor([-1 for _ in range(node_rep2.shape[0])], dtype=torch.long)
        shift = 0
        max_piece = 0
        groups, weights = argv2[3], argv2[4]
        for b in range(len(groups)):
            g = groups[b]
            sub_batch_idx[shift:shift+len(g)] = torch.tensor(g) + max_piece
            shift = shift + len(g)
            max_piece = max_piece + max(g) + 1

        group_idx = self.pool(batch_index.view(-1, 1), sub_batch_idx)
        group_idx = group_idx.long().flatten().to(node_rep2.device)
        sub_batch_idx = sub_batch_idx.to(node_rep2.device)

        node_rep2 = self.pool(node_rep2, sub_batch_idx)

        shift = 0
        mask = torch.ones(re_graph.edge_index.shape[-1] + node_rep2.shape[0]).to(node_rep2.device)
        for b in range(len(weights)):
            w = weights[b]
            mask[shift:shift+len(w)] = torch.tensor(w)
            shift = shift + len(w)
        mask = mask.view(-1, 1)
        node_rep2 = self.re_gnn(x=node_rep2, edge_index=re_graph.edge_index, edge_attr=re_graph.edge_attr, mask=mask)
         
        output1 = self.pool(node_rep1, batch_index.to(node_rep1.device)) #+ self.pool(cross_att, group_idx)
        output2 = self.pool(node_rep2, group_idx)

        score1 = self.w_l(output1).sigmoid()
        score2 = self.w_v(output2).sigmoid()
        # score = ((score1 + score2)).sigmoid()

        # # # score1, score2 = score[:, :, 0], score[:, :, 1]
        # # # output3 = score1[:, :, None] * output1[:, None, :] + score2[:, :, None] * output2[:, None, :]
        # # output3_gate = score[:, :, None] * output1[:, None, :]+ (1 - score[:, :, None]) * output2[:, None, :]
        
        output3 = (output1 + output2) / 2 + score1 * output1 + score2 * output2   #* self.scale
        # output3 = torch.cat([(output1 + output2) / 2, (score * output1 + (1 score) * output2)], dim=1)
        # output3 = F.dropout(output3, p=0.005, training=self.training)
        # output3_gate = output1.detach() * score + output2.detach() * (1 - score)
        # output3 = (output1 + output2) / 2
        # output3 = (output1 * score + output2 * (1 - score))
        res1 = self.graph_pred_linear(output1)
        res2 = self.graph_pred_linear2(output2)
        res3 = self.graph_pred_linear3(output3)
        # res_gate = torch.sum(self.graph_pred_linear3.weight * output3_gate, dim=-1) + self.graph_pred_linear3.bias
        # res_gate = self.graph_pred_linear3(output3_gate)
        # if self.training:
        #     output3 = (output1 + output2) / 2
        #     res3 = self.graph_pred_linear3(output3)
        #     return res1 , res2, res3, res_gate
        # res3 = self.graph_pred_linear(output1) + self.graph_pred_linear2(output2)
        # res3 = torch.sum(self.graph_pred_linear3.weight * output3, dim=-1) #+ self.graph_pred_linear3.bias
        return res1, res2, res3
        # return score1, score2
    

if __name__ == "__main__":
    m = GNN_graphpred(5, 2, 3000, 1)
    print(m)




        # output_batch = self.group_node_rep(output, batch_data.batch.cpu().numpy(), len(node_rep_groups))
        # final_features = []
        # for b in range(len(node_rep_groups)):
        #     final_rep = output_batch[b]
        #     weights = F.linear(F.normalize(final_rep), F.normalize(self.adaptative_weight))
        #     weights = torch.softmax(weights, dim=0)
        #     final_rep = final_rep * weights #* self.scale
        #     final_features.append(torch.mean(final_rep, dim=0))
        # output = torch.stack(final_features, dim=0)