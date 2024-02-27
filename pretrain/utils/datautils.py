import torch
import numpy as np
from rdkit import Chem
from torch_geometric.data import Batch
from torch_geometric.data import Data
import networkx as nx
import random


allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}


def mol_to_graph_data_obj_simple(mol, return_num=1):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    num_atom_features = 2  # atom type,  chirality tag
    atom_features_list = []
    # mol = smiles2molecule(mol)
    # mol = Chem.MolFromSmiles(mol, sanitize=True)

    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
                                         'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        if return_num > 1:
            edge_index2 = torch.tensor(np.array(edges_list).T, dtype=torch.long)
            edge_attr2 = torch.tensor(np.array(edge_features_list), dtype=torch.long)
            edge_index3 = torch.tensor(np.array(edges_list).T, dtype=torch.long)
            edge_attr3 = torch.tensor(np.array(edge_features_list), dtype=torch.long)
        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
        if return_num > 1:
            edge_index2 = torch.empty((2, 0), dtype=torch.long)
            edge_attr2 = torch.empty((0, num_bond_features), dtype=torch.long)
            edge_index3 = torch.empty((2, 0), dtype=torch.long)
            edge_attr3 = torch.empty((0, num_bond_features), dtype=torch.long)
    if return_num == 1:
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data
    else:
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data2 = Data(x=x, edge_index=edge_index2, edge_attr=edge_attr2)
        data3 = Data(x=x, edge_index=edge_index3, edge_attr=edge_attr3)
        return data, data2, data3


def moltree_to_graph_data(graph_data_batch):
    # graph_data_batch = []
    new_batch = Batch().from_data_list(graph_data_batch)
    return new_batch


def connect(group1, group2, dic, edge_list, edge_feature_list, i, j):
    for n1 in group1:
        for n2 in group2:
            if (n1, n2) in dic:
                edge_list.append((i, j))
                edge_feature_list.append(dic[(n1,n2)])

                edge_list.append((j, i))
                edge_feature_list.append(dic[(n2,n1)])

                assert dic[(n1,n2)] == dic[(n2,n1)]


def subgraph_connect(groups, mole_edge, attr):
    # connect_matrix = torch.zeros((mole_num, mole_num))
    mole_edge = mole_edge.numpy()

    dic = {}
    for i in range(mole_edge.shape[-1]):
        x, y = mole_edge[0, i], mole_edge[1, i]
        if (x, y) in dic:
            assert 0
        else:
            dic[(x,y)] = attr[i, :].tolist()

    edge_list = []
    edge_feature_list = []

    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            connect(groups[i], groups[j], dic, edge_list, edge_feature_list, i, j)

    return edge_list, edge_feature_list

def subgraph_inner_connect(groups, mole_edge, attr):
    mole_edge = mole_edge.numpy()

    dic = {}
    for i in range(mole_edge.shape[-1]):
        x, y = mole_edge[0, i], mole_edge[1, i]
        if (x, y) in dic:
            assert 0
        else:
            dic[(x,y)] = attr[i, :].tolist()

    edge_list = []
    edge_feature_list = []

    for g in groups:
        for i in range(len(g)):
            for j in range(i+1, len(g)):
                start, end = g[i], g[j]
                if (start, end) in dic:
                    edge_list.append((start, end))
                    edge_feature_list.append(dic[(start, end)])

                    edge_list.append((end, start))
                    edge_feature_list.append(dic[(end, start)])

    return edge_list, edge_feature_list


def get_connect(cur_ps, adjacent, map_loc):
    connects = []
    for i in range(len(adjacent)):
        if adjacent[i][cur_ps] == 1:
            connects.append(map_loc[i])
    connects.sort()
    return connects


def get_prior(data_list):
    m_pieces = []
    m_connect = []
    for idx in range(len(data_list)):
        pieces = data_list[idx]['pieces']
        connect = torch.tensor(data_list[idx]['sub_group'][0]).t()
        if len(pieces) == 1:
            adj = None
        else:
            adj = torch.zeros((len(pieces), len(pieces)))
            adj[connect[0], connect[1]] = 1

        m_pieces.append(pieces)
        m_connect.append(adj)
    
    prior = {}
    for i in range(len(m_pieces)):
        ps = m_pieces[i]
        if len(ps) == 1:
            continue
        adjacent = m_connect[i]
        for j in range(len(ps)):
            cur_ps = ps[j]
            neighbors = str(get_connect(j, adjacent, ps))
            
            if neighbors not in prior:
                prior[neighbors] = [0 for _ in range(100)]

            # for k in cur_ps:
            prior[neighbors][cur_ps] += 1

    res = []
    for i in range(len(m_pieces)):
        ps = m_pieces[i]
        if len(ps) == 1:
            res.append([-100])
            continue
        adjacent = m_connect[i]
        res_map = []
        for j in range(len(ps)):
            cur_ps = ps[j]
            neighbors = str(get_connect(j, adjacent, ps))
            
            piece_prior = torch.tensor(prior[neighbors])
            piece_prior = piece_prior / piece_prior.sum()
            res_map.append(piece_prior.tolist())
        res.append(res_map)
    return res

def drop_nodes(data, aug_ratio=0.2, idx=None):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num  * aug_ratio)

    if idx is None:
        idx_perm = np.random.permutation(node_num)
        idx_drop = idx_perm[:drop_num]
        idx_nondrop = idx_perm[drop_num:]
    else:
        idx_perm = [i for i in range(node_num)]
        idx_drop = np.array(idx)
        idx_nondrop = np.array(list(set(idx_perm) - set(idx_drop)))

    idx_nondrop.sort()
    idx_dict = {idx_nondrop[n]:n for n in list(range(idx_nondrop.shape[0]))}

    edge_index = data.edge_index.numpy()
    edge_mask = np.array([n for n in range(edge_num) if not (edge_index[0, n] in idx_drop or edge_index[1, n] in idx_drop)])

    edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    try:
        data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
        data.x = data.x[idx_nondrop]
        data.edge_attr = data.edge_attr[edge_mask]
    except:
        data = data

    return data


# def drop_subgraph(s_edges, s_attr, si_edges, si_attr, groups, x, aug_ratio=0.15):
#     drop_num = int(len(groups)  * aug_ratio) + 1
#     if drop_num == 0:
#         s_edges = torch.tensor(np.array(s_edges).T, dtype=torch.long)
#         s_attr = torch.tensor(np.array(s_attr), dtype=torch.long)   
#         s_data = Data(x=torch.zeros(len(groups), 1), edge_index=s_edges, edge_attr=s_attr)

#         si_edges = torch.tensor(np.array(si_edges).T, dtype=torch.long)
#         si_attr = torch.tensor(np.array(si_attr), dtype=torch.long)
#         si_data = Data(x=x, edge_index=si_edges, edge_attr=si_attr)
#         return s_data, si_data
    
#     masked_subgraph_indices = random.sample(range(len(groups)), drop_num)

    
    
