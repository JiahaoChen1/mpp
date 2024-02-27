import torch
from collections import defaultdict
from queue import Queue
import heapq
from copy import copy, deepcopy

import numpy as np

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdchem import BondType
import torch
from torch_geometric.data import Data

import os, sys
import networkx as nx
# import copy


MAX_VALENCE = {'B': 3, 'Br':1, 'C':4, 'Cl':1, 'F':1, 'I':1, 'N':5, 'O':2, 'P':5, 'S':6} #, 'Se':4, 'Si':4}


def smi2mol(smiles: str, kekulize=False, sanitize=True):
    '''turn smiles to molecule'''
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if kekulize:
        Chem.Kekulize(mol, True)
    return mol


def mol2smi(mol):
    return Chem.MolToSmiles(mol)


def get_submol(mol, atom_indices, kekulize=False):
    if len(atom_indices) == 1:
        atom_symbol = mol.GetAtomWithIdx(atom_indices[0]).GetSymbol()
        if atom_symbol == 'Si':
            atom_symbol = '[Si]'
        return smi2mol(atom_symbol, kekulize)
    aid_dict = { i: True for i in atom_indices }
    edge_indices = []
    for i in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(i)
        begin_aid = bond.GetBeginAtomIdx()
        end_aid = bond.GetEndAtomIdx()
        if begin_aid in aid_dict and end_aid in aid_dict:
            edge_indices.append(i)
    mol = Chem.PathToSubmol(mol, edge_indices)
    return mol


def get_submol_atom_map(mol, submol, group, kekulize=False):
    if len(group) == 1:
        return { group[0]: 0 }
    # turn to smiles order
    smi = mol2smi(submol)
    submol = smi2mol(smi, kekulize, sanitize=False)
    # # special with N+ and N-
    # for atom in submol.GetAtoms():
    #     if atom.GetSymbol() != 'N':
    #         continue
    #     if (atom.GetExplicitValence() == 3 and atom.GetFormalCharge() == 1) or atom.GetExplicitValence() < 3:
    #         atom.SetNumRadicalElectrons(0)
    #         atom.SetNumExplicitHs(2)
    
    matches = mol.GetSubstructMatches(submol)
    old2new = { i: 0 for i in group }  # old atom idx to new atom idx
    found = False
    for m in matches:
        hit = True
        for i, atom_idx in enumerate(m):
            if atom_idx not in old2new:
                hit = False
                break
            old2new[atom_idx] = i
        if hit:
            found = True
            break
    assert found
    return old2new


def cnt_atom(smi, return_dict=False):
    atom_dict = { atom: 0 for atom in MAX_VALENCE }
    for i in range(len(smi)):
        symbol = smi[i].upper()
        next_char = smi[i+1] if i+1 < len(smi) else None
        if symbol == 'B' and next_char == 'r':
            symbol += next_char
        elif symbol == 'C' and next_char == 'l':
            symbol += next_char
        if symbol in atom_dict:
            atom_dict[symbol] += 1
    if return_dict:
        return atom_dict
    else:
        return sum(atom_dict.values())
    

class SubgraphNode:
    '''
    The node representing a subgraph
    '''
    def __init__(self, smiles: str, pos: int, atom_mapping: dict, kekulize: bool):
        self.smiles = smiles
        self.pos = pos
        self.mol = smi2mol(smiles, kekulize, sanitize=False)
        # map atom idx in the molecule to atom idx in the subgraph (submol)
        self.atom_mapping = copy(atom_mapping)
    
    def get_mol(self):
        '''return molecule in rdkit form'''
        return self.mol

    def get_atom_mapping(self):
        return copy(self.atom_mapping)

    def __str__(self):
        return f'''
                    smiles: {self.smiles},
                    position: {self.pos},
                    atom map: {self.atom_mapping}
                '''


class SubgraphEdge:
    '''
    Edges between two subgraphs
    '''
    def __init__(self, src: int, dst: int, edges: list):
        self.edges = copy(edges)  # list of tuple (a, b, type) where the canonical order is used
        self.src = src
        self.dst = dst
        self.dummy = False
        if len(self.edges) == 0:
            self.dummy = True
    
    def get_edges(self):
        return copy(self.edges)
    
    def get_num_edges(self):
        return len(self.edges)

    def __str__(self):
        return f'''
                    src subgraph: {self.src}, dst subgraph: {self.dst},
                    atom bonds: {self.edges}
                '''
    

class Molecule(nx.Graph):
    '''molecule represented in subgraph-level'''

    def __init__(self, smiles: str=None, groups: list=None, kekulize: bool=False):
        super().__init__()
        if smiles is None:
            return
        self.graph['smiles'] = smiles
        rdkit_mol = smi2mol(smiles, kekulize)
        # processing atoms
        aid2pos = {}
        for pos, group in enumerate(groups):
            for aid in group:
                aid2pos[aid] = pos
            subgraph_mol = get_submol(rdkit_mol, group, kekulize)
            subgraph_smi = mol2smi(subgraph_mol)
            atom_mapping = get_submol_atom_map(rdkit_mol, subgraph_mol, group, kekulize)
            node = SubgraphNode(subgraph_smi, pos, atom_mapping, kekulize)
            self.add_node(node)
        # process edges
        edges_arr = [[[] for _ in groups] for _ in groups]  # adjacent
        for edge_idx in range(rdkit_mol.GetNumBonds()):
            bond = rdkit_mol.GetBondWithIdx(edge_idx)
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()

            begin_subgraph_pos = aid2pos[begin]
            end_subgraph_pos = aid2pos[end]
            begin_mapped = self.nodes[begin_subgraph_pos]['subgraph'].atom_mapping[begin]
            end_mapped = self.nodes[end_subgraph_pos]['subgraph'].atom_mapping[end]

            bond_type = bond.GetBondType()
            edges_arr[begin_subgraph_pos][end_subgraph_pos].append((begin_mapped, end_mapped, bond_type))
            edges_arr[end_subgraph_pos][begin_subgraph_pos].append((end_mapped, begin_mapped, bond_type))

        # add egdes into the graph
        for i in range(len(groups)):
            for j in range(len(groups)):
                if not i < j or len(edges_arr[i][j]) == 0:
                    continue
                edge = SubgraphEdge(i, j, edges_arr[i][j])
                self.add_edge(edge)
    
    @classmethod
    def from_nx_graph(cls, graph: nx.Graph, deepcopy=True):
        if deepcopy:
            graph = deepcopy(graph)
        graph.__class__ = Molecule
        return graph

    @classmethod
    def merge(cls, mol0, mol1, edge=None):
        # reorder
        node_mappings = [{}, {}]
        mols = [mol0, mol1]
        mol = Molecule.from_nx_graph(nx.Graph())
        for i in range(2):
            for n in mols[i].nodes:
                node_mappings[i][n] = len(node_mappings[i])
                node = deepcopy(mols[i].get_node(n))
                node.pos = node_mappings[i][n]
                mol.add_node(node)
            for src, dst in mols[i].edges:
                edge = deepcopy(mols[i].get_edge(src, dst))
                edge.src = node_mappings[i][src]
                edge.dst = node_mappings[i][dst]
                mol.add_edge(src, dst, connects=edge)
        # add new edge
        edge = deepcopy(edge)
        edge.src = node_mappings[0][edge.src]
        edge.dst = node_mappings[1][edge.dst]
        mol.add_edge(edge)
        return mol

    def get_edge(self, i, j) -> SubgraphEdge:
        return self[i][j]['connects']
    
    def get_node(self, i) -> SubgraphNode:
        return self.nodes[i]['subgraph']

    def add_edge(self, edge: SubgraphEdge) -> None:
        src, dst = edge.src, edge.dst
        super().add_edge(src, dst, connects=edge)
    
    def add_node(self, node: SubgraphNode) -> None:
        n = node.pos
        super().add_node(n, subgraph=node)

    def subgraph(self, nodes: list):
        graph = super().subgraph(nodes)
        assert isinstance(graph, Molecule)
        return graph

    def to_rdkit_mol(self):
        mol = Chem.RWMol()
        aid_mapping, order = {}, []
        # add all the subgraphs to rwmol
        for n in self.nodes:
            subgraph = self.get_node(n)
            submol = subgraph.get_mol()
            local2global = {}
            for global_aid in subgraph.atom_mapping:
                local_aid = subgraph.atom_mapping[global_aid]
                local2global[local_aid] = global_aid
            for atom in submol.GetAtoms():
                new_atom = Chem.Atom(atom.GetSymbol())
                new_atom.SetFormalCharge(atom.GetFormalCharge())
                mol.AddAtom(atom)
                aid_mapping[(n, atom.GetIdx())] = len(aid_mapping)
                order.append(local2global[atom.GetIdx()])
            for bond in submol.GetBonds():
                begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                begin, end = aid_mapping[(n, begin)], aid_mapping[(n, end)]
                mol.AddBond(begin, end, bond.GetBondType())
        for src, dst in self.edges:
            subgraph_edge = self.get_edge(src, dst)
            pid_src, pid_dst = subgraph_edge.src, subgraph_edge.dst
            for begin, end, bond_type in subgraph_edge.edges:
                begin, end = aid_mapping[(pid_src, begin)], aid_mapping[(pid_dst, end)]
                mol.AddBond(begin, end, bond_type)
        mol = mol.GetMol()
        new_order = [-1 for _ in order]
        for cur_i, ordered_i in enumerate(order):
            new_order[ordered_i] = cur_i
        mol = Chem.RenumberAtoms(mol, new_order)
        # sanitize, we need to handle mal-formed N+
        mol.UpdatePropertyCache(strict=False)
        ps = Chem.DetectChemistryProblems(mol)
        if not ps:  # no problem
            Chem.SanitizeMol(mol)
            return mol
        for p in ps:
            if p.GetType()=='AtomValenceException':  # for N+, we need to set its formal charge
                at = mol.GetAtomWithIdx(p.GetAtomIdx())
                if at.GetAtomicNum()==7 and at.GetFormalCharge()==0 and at.GetExplicitValence()==4:
                    at.SetFormalCharge(1)
        Chem.SanitizeMol(mol)
        return mol

    def to_SVG(self, path: str, size: tuple=(200, 200), add_idx=False) -> str:
        # save the subgraph-level molecule to an SVG image
        # return the content of svg in string format
        mol = self.to_rdkit_mol()
        if add_idx:  # this will produce an ugly figure
            for i in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(i)
                atom.SetAtomMapNum(i)
        tm = rdMolDraw2D.PrepareMolForDrawing(mol)
        view = rdMolDraw2D.MolDraw2DSVG(*size)
        option = view.drawOptions()
        option.legendFontSize = 18
        option.bondLineWidth = 1
        option.highlightBondWidthMultiplier = 20
        sg_atoms, sg_bonds = [], []
        atom2subgraph, atom_color, bond_color = {}, {}, {}
        # atoms in each subgraph
        for i in self.nodes:
            node = self.get_node(i)
            # random color in rgb. mix with white to obtain soft colors
            color = tuple(((np.random.rand(3) + 1)/ 2).tolist())
            for atom_id in node.atom_mapping:
                sg_atoms.append(atom_id)
                atom2subgraph[atom_id] = i
                atom_color[atom_id] = color
        # bonds in each subgraph
        for bond_id in range(mol.GetNumBonds()):
            bond = mol.GetBondWithIdx(bond_id)
            begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if atom2subgraph[begin] == atom2subgraph[end]:
                sg_bonds.append(bond_id)
                bond_color[bond_id] = atom_color[begin]
        view.DrawMolecules([tm], highlightAtoms=[sg_atoms], \
                           highlightBonds=[sg_bonds], highlightAtomColors=[atom_color], \
                           highlightBondColors=[bond_color])
        view.FinishDrawing()
        svg = view.GetDrawingText()
        with open(path, 'w') as fout:
            fout.write(svg)
        return svg

    def to_smiles(self):
        rdkit_mol = self.to_rdkit_mol()
        return mol2smi(rdkit_mol)

    def __str__(self):
        desc = 'nodes: \n'
        for ni, node in enumerate(self.nodes):
            desc += f'{ni}:{self.get_node(node)}\n'
        desc += 'edges: \n'
        for src, dst in self.edges:
            desc += f'{src}-{dst}:{self.get_edge(src, dst)}\n'
        return desc
    

def is_connect(group1, group2, connect_matrix):
    for n1 in group1:
        for n2 in group2:
            if connect_matrix[n1][n2] == 1:
                assert connect_matrix[n2][n1] == 1
                return True
    return False


def subgraph_connect_single(groups, mole_edge, mole_num):
    connect = torch.zeros((mole_num, mole_num))
    mole_edge = mole_edge.numpy()
    connect[mole_edge[0], mole_edge[1]] = 1

    node_num = len(groups)
    adj = torch.zeros((node_num, node_num), dtype=torch.long) 

    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            if is_connect(groups[i], groups[j], connect):
                adj[i][j] = 1
                adj[j][i] = 1

    edge_index = adj.nonzero().tolist()
    edge_attr = [[0 for _ in range(2)] for _ in range(len(edge_index))]

    return edge_index, edge_attr


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

    # edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long)

    # edge_attr = torch.tensor(np.array(edge_feature_list), dtype=torch.long)

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


    # edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long)

    # edge_attr = torch.tensor(np.array(edge_feature_list), dtype=torch.long)

    return edge_list, edge_feature_list



def drop_nodes(data, rate=1):
    """
    Randomly dropping certain ratio of nodes.
    For those nodes to be dropped, remove all their edges by the following statements:
    adj[drop_node_idx, :] = 0, adj[:, drop_node_idx] = 0.
    :param data: input (class: torch_geometric.data.Data)
    :param rate: drop node rate
    :return: output (class: torch_geometric.data.Data)
    """
    data = deepcopy(data)
    node_num, _ = data.x.size()
    # _, edge_num = data.edge_index.size()
    # drop_num = int(node_num * rate)
    if node_num == 1:
        return data
    idx_drop = np.random.choice(node_num, rate, replace=False)
#     print(idx_drop)
    edge_index = data.edge_index.numpy()
    ori_edge_index = edge_index.T.tolist()

    # adj = torch.zeros((node_num, node_num))
    # adj[edge_index[0], edge_index[1]] = 1
    # adj[idx_drop, :] = 0
    # adj[:, idx_drop] = 0
    # edge_index = adj.nonzero().t()

    # data.edge_index = edge_index
    # aft_edge_index = edge_index.numpy().T.tolist()

    keep_idx_atom = []
    keep_idx = []
    for idx, each in enumerate(ori_edge_index):
        if idx_drop not in each:
            keep_idx.append(idx)

    # for idx in  range(node_num):
    #     if idx != idx_drop:
    #         keep_idx_atom.append(idx)

    # data.x = data.x[keep_idx_atom, :]
    data.edge_attr = data.edge_attr[keep_idx, :]
    data.edge_index = data.edge_index[:, keep_idx]

    return data