import torch
from loader import mol_to_graph_data_obj_simple

from rdkit.Chem import AllChem
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, to_dense_batch, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from subgraph_model import GNNSubgraph
from torch_geometric.data import Batch
from torch_geometric.data import Data

from torch.nn.parameter import Parameter

import numpy as np
import random

from utils import drop_nodes

torch.set_printoptions(precision=4, sci_mode=False)
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
        # mask = torch.zeros((self.atom_num, self.atom_num)).cuda()
        # mask[edge_index[0], edge_index[1]] = 1
        # mask = mask.nonzero().t()
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


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
            # self.graph_pred_linear = torch.nn.Linear(self.mult * emb_dim, self.num_tasks)
            self.graph_pred_linear = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), 
                                                         torch.nn.ReLU(),
                                                         torch.nn.Linear(emb_dim, self.num_tasks))
    def from_pretrained(self, model_file):
        state_dict = torch.load(model_file, map_location='cpu')
        miss1, miss2 = self.gnn.load_state_dict(state_dict['gnn'], strict=False)
        # p = '/data00/jiahao/mole_supervise/pretraining_graphmvpc.pth'
        # miss1, miss2 = self.gnn.load_state_dict(torch.load(p, map_location='cpu'), strict=False)
        # print(f'gnn load {miss1}, {miss2}')
        # miss1, miss2 = self.sub_gnn.load_state_dict(state_dict['sub_gnn'], strict=False)
        # print(f'sub gnn load {miss1}, {miss2}')
        # miss1, miss2 = self.re_gnn.load_state_dict(state_dict['re_gnn'], strict=False)
        # print(f're gnn load {miss1}, {miss2}')

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

    def forward(self, argv1):
        '''
        node_rep1: (N1, C), N1 is the number of molecules
        node_rep2: (N2, C), N2 is the number of the first principle graph
        '''
        batch_index = argv1[3].cpu()
        # batch_num = int(max(batch_index) + 1)
        node_rep1 = self.gnn_forward(argv1[0], argv1[1], argv1[2])

        output1 = self.pool(node_rep1, batch_index.to(node_rep1.device))
    

        return self.graph_pred_linear(output1)


mol_list = ['CC(C)(C)C1=CC=C(C=C1)O.CC(C)(C)C1=CC=C(C=C1)O.[OH-]',
'[H+].C1=CC=C(C=C1)C(=O)N[C@@H](CC2=CC=CC=C2)C(=O)O',
'CC=CC(CC1SCCCC1(CCl)C(C)Cl)=[SH]OCS', 'CC(=CCC=CCCC(C)S)CCBr', 'OCI(Cl)I(OI)[IH]O', 'FC1=CC=CC=I1', 'CC1(C=C[IH]OO)C=IC1', 'FC=CC1CSCSC1CCC=CCl', 'CC(O)O[SH]=NN=NN=NN', 'CC(C)C12C3(C)C45C6CCC7CC8C64C31C(C)(C7)C852', 'C=CCC(C(C)CC=CCCl)C(F)OC(O)O', 'CC=[IH]1C2C=CCC21', 'CC1CP(C)CPCC(P)COCC(C)C(P)CP1', 'CC(Br)C[SH](S)C(OCC1CC1)C(=[IH])CSCCBr', 'C=CC=CC(=CCCS)CI=C', 'C[SH]1CC(=S)CCC(C2CC2(C)CSF)O1', 'CCCCC(C)CS', 'CCSC12C=C1CC(SC)SCC(C)C2Cl', 'C=C(CC)C(I)C[SH](CCC)C(Cl)CBr', 'C[IH]C(COO)CONN=N', 'ClNO[IH]NI', 'NN(Cl)O[IH]N(Cl)N=NNCl', 'NOCO[IH]I(O)O', 'CC(OO)[IH]1=CC=CC=[IH]1CO', 'CC=C(CCCCC)C(Cl)CCCC(C)Br', 'C=C(CC)CC1PC(F)C(=C)CCC=C1CC', 'FC(Cl)CCSCCCl', 'NN(Cl)N=NCl', 'ClN[IH][IH]ON(Cl)N[IH]Cl', 'CCCC1=CCP(S)CPCC(CF)NC(C)OCC1P', 'ON(Cl)[IH]N(O)[IH]NCl', 'OI(O)[IH]Cl', 'OCC1(CP)C(CCl)CCCCCP1Br', 'CCC(=[IH])C=C1CC(=CCl)C(CCS(=O)COS)C1', 'N[IH][IH][IH][IH](Cl)=II(O)OI', 'CC1=CC=C(c2n[nH][nH]on2O)C=I1', 'ON[IH][IH]I=CCOC=NOO', 'ON[IH][IH]I(O)O[IH]O', 'CCCCC(Cl)CC(S)Cl', 'ON(Cl)O[IH]NCl', 'FCSC(C=CC=CNBr)CCBr', 'ClC[IH]OC(=[IH])COC(I)=IC=COI(NO[IH]I)[IH]OOI', 'FOCCOC[IH]1=CC=CI=[IH]1CO[IH]O[IH]Cl', 'CCC=PCC(N)OC1CNCC1CC(S)CC', 'N[IH]N(N)O', 'C=CCC1(CPC)C=ICC=S1O', 'C=CC(=S)SC(Br)O[SH]1CCC(C=CBr)C1Br', 'OI(O)C(I)C=COI', 'N[IH]O[IH]I(O)I', 'CC(=CO)NNC(C)(O)[IH](O)(I)I([IH]O[IH]Cl)[IH]O[IH]Cl', 'C=C(CCC=CC1C2C=CC1CC2)CCCC=CC=S(C)S', 'C=CC(CCSC)=I(=C)N', 'ON[IH]O[IH]NCl', 'OOCC=CC(O)C(F)C(I)CC=C(I)COI', 'S[SH]1CCC2=[IH](I)C1C=C2', 'O=CCOCCl', 'CC(F)(CS)C(F)Cl', 'O=C1CCCC(=[IH])C(C(=CBr)CCCBr)SCCCCS1', 'C[IH]C(CCCC=CCCCCBr)C(O)OO', 'NON(N)NNN(O)N(O)ONNCl', 'N=NNCl', 'ON([IH]NCl)[IH]NI']

# mol_list = ['C(C[C@@H](CC(=O)O)[C@H]1[C@H]([C@H]([C@@H]([C@H](O1)O[C@@H]2[C@H]([C@H]([C@H](O[C@H]2O)O[C@@H]3[C@H](O[C@@H]([C@@H]([C@H](O3)C)O)O)O)O)N(C)C)O)O)O']
mol_object_list = []

for s in mol_list:
    # print(s)
    m = mol_to_graph_data_obj_simple(AllChem.MolFromSmiles(s))
    mol_object_list.append(m)

batch_mol = Batch.from_data_list(mol_object_list)
model = GNN_graphpred(5, 0, 300, 1)
model.eval()

key, miss = model.load_state_dict(torch.load('/data00/jiahao/mole_supervise/downstream/save_models/esol.pth', map_location='cpu'), strict=False)
print(key, miss)
with torch.no_grad():
    r = model((batch_mol.x, batch_mol.edge_index, batch_mol.edge_attr, batch_mol.batch))
    # r = r.sigmoid()
    r = torch.max(r, dim=1)[0]
    print(r)
# print(batch_mol)