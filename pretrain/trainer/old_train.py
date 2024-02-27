import torch
from torch_geometric.data import Batch
# from torch_geometric.data import Data
import os
from torch_geometric.nn import global_mean_pool
# from utils.datautils import moltree_to_graph_data, drop_nodes, mol_to_graph_data_obj_simple, subgraph_connect
# import numpy as np
import  torch.nn.functional as F
from utils.nn_utils import MaskAtom, BatchMasking, MaskSubgraph, JointMask
import torch.optim as optim
import torch.nn as nn
# import copy
# from utils.datautils import drop_nodes


def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)


class Trainer:
    def __init__(self, dataloader, model,device, folder, interval, rate, emb_dim, cl_loss='exe') -> None:
        self.dataloader = dataloader
        self.model = model
        self.device = device

        self.folder = folder
        self.interval = interval
        self.rate = rate

        self.pool = global_mean_pool

        self.proj_head1 = nn.Sequential(nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300,300)).to(device)
        # self.proj_head2 = nn.Sequential(nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300,300)).to(device)

        self.mask_atom = MaskAtom(num_atom_type = 119, num_edge_type = 5, 
                                  mask_rate = rate, mask_edge=False)
        self.mask_atom2 = MaskAtom(num_atom_type = 119, num_edge_type = 5, 
                                  mask_rate = rate * 2, mask_edge=False)
        self.mask_subgraph = MaskSubgraph(mask_rate=rate, mask_embedding=model.re_gnn.mask_embedding)
        
        # self.joit_mask = JointMask(mask_rate=rate)

        self.linear_pred_atoms = torch.nn.Linear(emb_dim, 119).to(device)
        # self.linear_pred_bonds = torch.nn.Linear(emb_dim, 4).to(device)
        self.linear_pred_subgraphs = torch.nn.Linear(emb_dim, 100).to(device)

        self.opt_model = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
        self.opt_linear_pred_atoms = optim.Adam(self.linear_pred_atoms.parameters(), lr=0.001, weight_decay=0)
        # self.opt_linear_pred_bonds = optim.Adam(self.linear_pred_bonds.parameters(), lr=0.001, weight_decay=0)
        self.opt_linear_pred_subgraphs = optim.Adam(self.linear_pred_subgraphs.parameters(), lr=0.001, weight_decay=0)

        self.opt_proj_head1 = optim.Adam(self.proj_head1.parameters(), lr=0.001, weight_decay=0)
        # self.opt_proj_head2 = optim.Adam(self.proj_head2.parameters(), lr=0.001, weight_decay=0)

        self.mask_loss = True
        self.cl_loss = False
        if cl_loss == 'exe':
            self.cl_loss = True
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)

    def train_one_step(self, si_data_batch, s_data_batch, group_data_batch, piece_data_batch, 
                       o_data_batch, weights, mask_subgraph_indices=None):

        node_rep = self.model.gnn_forward(o_data_batch)
        pred_node = self.linear_pred_atoms(node_rep[o_data_batch.masked_atom_indices])
        loss_atom = F.cross_entropy(pred_node.double(), o_data_batch.mask_node_label[:,0])
        
        #  subgraph masking
        node_rep_sub = self.model.sub_gnn_forward(si_data_batch)
        node_rep_sub, mask_indices, labels, group_indices, sub_batch_idx = self.mask_subgraph(node_rep_sub, group_data_batch,
                                                                piece_data_batch, node_rep_sub.shape[0],
                                                                mask_subgraph_indices)
        node_rep_sub = self.model.re_gnn_forward(node_rep_sub, s_data_batch.edge_index, s_data_batch.edge_attr, weights)
        pred_subgraph = self.linear_pred_subgraphs(node_rep_sub[mask_indices])
        loss_kl = F.cross_entropy(pred_subgraph, labels[mask_indices], ignore_index=-100)

        loss = loss_atom + loss_kl

        acc_node = compute_accuracy(pred_node, o_data_batch.mask_node_label[:,0]) 
        acc_subgraph =  compute_accuracy(pred_subgraph, labels[mask_indices])

        return loss, acc_node, acc_subgraph, node_rep, node_rep_sub, group_indices

    def train_one_epoch(self):
        cnt_loss = 0.
        cnt_loss_cl = 0.
        cnt_acc_node, cnt_acc_subgraph = 0., 0.
        for idx, batch in enumerate(self.dataloader):
            self.opt_model.zero_grad()

            group_data_batch = []
            piece_data_batch = []
            si_data_batch = []
            s_data_batch = []
            o_data_batch = []
            ocon_data_batch = []
            anchor_data_batch = []

            weights = []
            # for mol in batch:
            #     group_data_batch.append(mol['groups'])
            #     weights.append(mol['subgraph_weights'])

            # mask_subgraph_indices, mask_atom_indices = self.joit_mask(group_data_batch)
            # mask_subgraph_indices2, mask_atom_indices2 = self.joit_mask(group_data_batch)

            for ind, mol in enumerate(batch):
                group_data_batch.append(mol['groups'])
                weights.append(mol['subgraph_weights'])
                si_data_batch.append(mol['si_data'])
                s_data_batch.append(mol['s_data'])
                piece_data_batch.append(mol['pieces'])
                # o_data_batch.append(self.mask_atom(mol['o_data'], masked_atom_indices=mask_atom_indices[ind]))
                o_data_batch.append(self.mask_atom(mol['o_data']))
                if self.cl_loss:
                    # ocon_data_batch.append(self.mask_atom(mol['o_data2'], masked_atom_indices=mask_atom_indices2[ind]))
                    ocon_data_batch.append(self.mask_atom2(mol['o_data2']))
                    # anchor_data_batch.append(mol['anchor_data'])

            si_data_batch = Batch().from_data_list(si_data_batch).to(self.device)
            s_data_batch = Batch().from_data_list(s_data_batch).to(self.device)
            o_data_batch = BatchMasking().from_data_list(o_data_batch).to(self.device)

            if self.cl_loss:
                ocon_data_batch = BatchMasking().from_data_list(ocon_data_batch).to(self.device)

            loss = 0
            loss_m = 0
            loss_cl_item = 0
            if self.mask_loss:
                self.opt_linear_pred_atoms.zero_grad()
                self.opt_linear_pred_subgraphs.zero_grad()
                loss_m, acc_node, acc_subgraph, node_rep, node_rep_sub, group_indices = self.train_one_step(si_data_batch, s_data_batch, group_data_batch, piece_data_batch, 
                                                                    o_data_batch, weights)
                loss_m2, acc_node2, acc_subgraph2, node_rep2, node_rep_sub2, _ = self.train_one_step(si_data_batch, s_data_batch, group_data_batch, piece_data_batch, 
                                                                    ocon_data_batch, weights)
                loss = loss + (loss_m + loss_m2) * 0.5
            if self.cl_loss:
                self.opt_proj_head1.zero_grad()
                # self.opt_proj_head2.zero_grad()
                
                loss_cl = self.contrastive_learning(node_rep, node_rep_sub, node_rep2, node_rep_sub2, ocon_data_batch.batch, group_indices)
                loss_cl_item = loss_cl.item()
                loss = loss + loss_cl
            
            loss.backward()
            self.opt_model.step()

            if self.mask_loss:
                self.opt_linear_pred_atoms.step()
                self.opt_linear_pred_subgraphs.step()
                # self.opt_linear_pred_bonds.step()
            if self.cl_loss:
                self.opt_proj_head1.step()
                # self.opt_proj_head2.step()

            cnt_loss, cnt_acc_node, cnt_acc_subgraph, cnt_loss_cl =\
                  cnt_loss + loss.item(), cnt_acc_node + acc_node, cnt_acc_subgraph + acc_subgraph, cnt_loss_cl + loss_cl_item

            if (idx + 1) % 200 == 0:
                print(f'Loss: {cnt_loss / 200:.4f}, Acc node: {cnt_acc_node / 200:.4f}, '\
                      f'Acc subgraph: {cnt_acc_subgraph / 200:.4f}, |cnt_loss {cnt_loss_cl / 200:.4f}')
                cnt_loss = 0.
                cnt_acc_node, cnt_loss_cl, cnt_acc_subgraph = 0., 0., 0.

    def train(self, num_epoch):
        self.model.train()
        for epoch in range(1, num_epoch + 1):
            print("====epoch " + str(epoch))

            self.train_one_epoch()
            if epoch % self.interval == 0:
                ckpt = {'gnn':self.model.gnn.state_dict(),
                        'sub_gnn':self.model.sub_gnn.state_dict(),
                        're_gnn':self.model.re_gnn.state_dict(),
                        'head1':self.model.head1.state_dict(),
                        'head2':self.model.head2.state_dict(),
                        'head3':self.model.head3.state_dict(),
                        'head4':self.model.head4.state_dict(),
                        'head5':self.model.head5.state_dict(),
                        'head6':self.model.head6.state_dict()}

                torch.save(ckpt, os.path.join(self.folder, 'epoch_{}.pth'.format(epoch)))


    def contrastive_learning(self, node_rep, node_rep_sub, node_rep2, node_rep_sub2, batch_idx, group_idx):

        # node_rep = self.pool(node_rep, batch_idx.to(self.device))
        # node_rep_ori = self.pool(node_rep2, batch_idx.to(self.device))

        # node_rep_sub = self.pool(node_rep_sub, group_idx.to(self.device))
        # node_rep_sub_ori = self.pool(node_rep_sub2, group_idx.to(self.device))
        batch_idx, group_idx =  batch_idx.to(self.device), group_idx.to(self.device)
        cross_att1, cross_att3 = self.model.cross_attention(node_rep, node_rep_sub, batch_idx, group_idx)
        cross_att2, cross_att4 = self.model.cross_attention(node_rep2, node_rep_sub2, batch_idx, group_idx)
        node_merge = (self.pool(cross_att1, batch_idx) + self.pool(cross_att3, group_idx)) / 2
        node_merge_ori = (self.pool(cross_att2, batch_idx) + self.pool(cross_att4, group_idx)) / 2

        # node_merge = self.model.gate_fusion(node_rep, node_rep_sub)
        # node_merge_ori = self.model.gate_fusion(node_rep_ori, node_rep_sub_ori)

        labels = torch.arange(0, node_merge.shape[0]).to(self.device)

        node_merge = self.proj_head1(node_merge)
        node_merge_ori = self.proj_head1(node_merge_ori)

        node_merge = F.normalize(node_merge)
        node_merge_ori = F.normalize(node_merge_ori)

        matrix2 = node_merge @ node_merge_ori.t()
        loss2 = F.cross_entropy(matrix2 / 0.1, labels)
        loss = loss2
        
        return loss.mean() * 0.1
    
    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss
