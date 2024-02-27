#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
sys.path.append('/home/u2022000162/code/mole_supervise/pretrain')
import torch
from torch.utils.data import Dataset
import numpy as np
# from rdkit import Chem
from torch_geometric.data import Data
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.datautils import subgraph_connect, subgraph_inner_connect, mol_to_graph_data_obj_simple,\
                            get_prior, drop_nodes
from utils.chem_utils import smi2mol
import random


class SuperDataset(Dataset):
    def __init__(self, fname, tokenizer, pre_load=''):
        super(SuperDataset, self).__init__()
        self.root_path, self.file_path = os.path.split(fname)
        self.save_path = os.path.join(self.root_path, pre_load)
        self.tokenizer = tokenizer

        self.pre_load = pre_load
        # if 'all' in pre_load:
        #     self.data = self.process_smiles()
        # else:
        try:
            if 'all' in self.save_path:
                self.data = torch.load(self.save_path)
            else:
                self.data = torch.load(self.save_path)
        except FileNotFoundError:
            self.data = self.process()
        # self.data =  self.process_smiles()

    # @staticmethod
    def process_step1(self, s):
        pieces, groups = self.tokenizer(s)
        mol = smi2mol(s)
        mol_data = mol_to_graph_data_obj_simple(mol)
        s_edges, s_attr = subgraph_connect(groups, mol_data.edge_index, mol_data.edge_attr)
        si_edges, si_attr = subgraph_inner_connect(groups, mol_data.edge_index, mol_data.edge_attr)

        assert len(s_edges) + len(si_edges) == mol_data.edge_index.shape[1]

        subgraph_weights = []
        mask = [1 for _ in range(len(s_edges))]
        his = {}
        for i in range(len(s_edges)):
            b, e = int(s_edges[i][0]), int(s_edges[i][1])
            if (b, e) in his:
                his[(b, e)] += 1.
            else:
                his[(b, e)] = 1.

        for i in range(len(s_edges)):
            b, e = int(s_edges[i][0]), int(s_edges[i][1])
            mask[i] = mask[i] / his[(b, e)]
        
        subgraph_weights.append(mask)

        return {
            'smile':s,
            'pieces': pieces,
            'groups': groups,
            'sub_group': [s_edges, s_attr],
            'subi_group': [si_edges, si_attr],
            'subgraph_weights': subgraph_weights
        }
    
    def process(self):
        # load smiles
        file_path = os.path.join(self.root_path, self.file_path)
        with open(file_path, 'r') as fin:
            lines = fin.readlines()
        smiles = [s.strip('\n') for s in lines]
        # turn smiles into data type of PyG
        data_list = []
        for s in tqdm(smiles):
            data_list.append(self.process_step1(s))
        # prior = get_prior(data_list)
        torch.save(data_list, self.save_path)
        # torch.save((data_list, prior), self.save_path)
        return data_list#, prior

    def process_smiles(self):
        # load smiles
        file_path = os.path.join(self.root_path, self.file_path)
        with open(file_path, 'r') as fin:
            lines = fin.readlines()
        smiles = [s.strip('\n') for s in lines]
        # turn smiles into data type of PyG
        data_list = []
        for s in tqdm(smiles):
            data_list.append(s)
        # prior = get_prior(data_list)
        return data_list
    
    def __getitem__(self, idx):
        s = self.data[idx]

        s_edges, s_attr = s['sub_group'][0], s['sub_group'][1]
        si_edges, si_attr = s['subi_group'][0], s['subi_group'][1]
        
        mol = smi2mol(s['smile'])
        ori_data, ori_data2, anchor_data = mol_to_graph_data_obj_simple(mol, return_num= 3)
        
        ori_data = drop_nodes(ori_data, 0.2)
        anchor_data = drop_nodes(anchor_data, 0.2)

        s_edges = torch.tensor(np.array(s_edges).T, dtype=torch.long)
        s_attr = torch.tensor(np.array(s_attr), dtype=torch.long)   
        s_data = Data(x=torch.zeros(len(s['groups']), 1), edge_index=s_edges, edge_attr=s_attr)
        si_edges = torch.tensor(np.array(si_edges).T, dtype=torch.long)
        si_attr = torch.tensor(np.array(si_attr), dtype=torch.long)
        si_data = Data(x=ori_data.x, edge_index=si_edges, edge_attr=si_attr)

        if len(s['groups']) != 0:
            random_drop_subgraph1 = random.sample(range(0, len(s['groups'])), 1)[0]
            random_drop_subgraph2 = random.sample(range(0, len(s['groups'])), 1)[0]

        return {'pieces':s['pieces'], 
                'groups':s['groups'], 
                'o_data':ori_data, 
                's_data':s_data, 
                'si_data':si_data, 
                # 'o_data2': ocon_data_batch,
                'o_data2':drop_nodes(ori_data2, 0.2),
                'anchor_data':drop_nodes(anchor_data, 0.2),
                'mask_indices': [random_drop_subgraph1, random_drop_subgraph2],
                # 'o_data2':None,
                # 'anchor_data':None,
                'subgraph_weights': None}
    
    def __len__(self):
        return len(self.data)

def get_dataloader(fname, tokenizer, batch_size, shuffle=True, num_workers=4, pre_load=''):
    dataset = SuperDataset(fname, tokenizer, pre_load=pre_load)
    # return dataset
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=lambda x: x, num_workers=num_workers)
    

if __name__ == '__main__':
    # pass
    tokenizer = Tokenizer('../downstream/table300.txt')
    dataloader = get_dataloader('./data/zinc250.txt', tokenizer, 2, shuffle=True)

    for item in dataloader:
        print(item)

    # m = dataloader[10]
    # print(dataloader[10])
    # data = mol_to_graph_data_obj_simple(smiles2molecule(m['smile']))

    # edge, attr = subgraph_inner_connect(m['groups'], data.edge_index, data.edge_attr)

    # new_data, pie = drop_nodes(data, 1, m['groups'])
    # # print(edge)

    # # graph_showing(edge, 'motif.png')
    # graph_showing(new_data.edge_index, 'drop.png')
    # graph_showing(data.edge_index, 'disconnect.png')

    # m_data1 = dataloader[10]
    # print(m_data1['groups'])
    # print(m_data1['pieces'])
    # print(a['pieces'])
    # print(dataloader[10])
    # print(Chem.MolToSmiles(m_data1['mol']))

    # data1 = mol_to_graph_data_obj_simple(m_data1['smile'])

    # edge = subgraph_connect(m_data1['groups'], m_data1['pieces'], data1.edge_index, mole_num=data1.x.shape[0])

    # drop, sub_drop = drop_nodes(data1, rate=0.1, sub_graph=m_data1['groups'], sub_pieces=m_data1['pieces'])
    

    # print(Chem.MolToSmiles(m_data1['mol']))


