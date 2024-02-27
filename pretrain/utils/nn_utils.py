import torch
import random
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
import numpy as np


class MaskAtom:
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge=True):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge

    def __call__(self, data, masked_atom_indices=None):
        """
        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """

        if masked_atom_indices is None:
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
            num_atoms = data.x.size()[0]
            sample_size = int(num_atoms * self.mask_rate + 1)
            masked_atom_indices = random.sample(range(num_atoms), sample_size)

        # create mask node label by copying atom feature of mask atom
        mask_node_labels_list = []
        for atom_idx in masked_atom_indices:
            mask_node_labels_list.append(data.x[atom_idx].view(1, -1))
        data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        data.masked_atom_indices = torch.tensor(masked_atom_indices)

        # modify the original node feature of the masked node
        for atom_idx in masked_atom_indices:
            data.x[atom_idx] = torch.tensor([self.num_atom_type, 0])

        if self.mask_edge:
            # create mask edge labels by copying edge features of edges that are bonded to
            # mask atoms
            connected_edge_indices = []
            for bond_idx, (u, v) in enumerate(data.edge_index.cpu().numpy().T):
                for atom_idx in masked_atom_indices:
                    if atom_idx in set((u, v)) and \
                        bond_idx not in connected_edge_indices:
                        connected_edge_indices.append(bond_idx)

            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to
                # the mask atoms
                mask_edge_labels_list = []
                for bond_idx in connected_edge_indices[::2]: # because the
                    # edge ordering is such that two directions of a single
                    # edge occur in pairs, so to get the unique undirected
                    # edge indices, we take every 2nd edge index from list
                    mask_edge_labels_list.append(
                        data.edge_attr[bond_idx].view(1, -1))

                data.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)
                # modify the original bond features of the bonds connected to the mask atoms
                for bond_idx in connected_edge_indices:
                    data.edge_attr[bond_idx] = torch.tensor(
                        [self.num_edge_type, 0])

                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices[::2])
            else:
                data.mask_edge_label = torch.empty((0, 2)).to(torch.int64)
                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices).to(torch.int64)

        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)


class BatchMasking(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchMasking, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchMasking()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0
        cumsum_edge = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index', 'masked_atom_indices']:
                    item = item + cumsum_node
                elif key  == 'connected_edge_indices':
                    item = item + cumsum_edge
                batch[key].append(item)

            cumsum_node += num_nodes
            cumsum_edge += data.edge_index.shape[1]

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
        batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ['edge_index', 'face', 'masked_atom_indices', 'connected_edge_indices']

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class MaskSubgraph:
    def __init__(self, mask_rate, mask_embedding, mask_edge=False) -> None:
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge

        self.pool = global_mean_pool
        self.embedding = mask_embedding

    def __call__(self, node_rep2, group_data_batch, piece_data_batch, num_atoms, mask_subgraph_indices=None):
        cnt = 0
        shift = 0
        group_idx = []
        sub_batch_idx = torch.tensor([0 for _ in range(num_atoms)], dtype=torch.long)
        for b in range(len(group_data_batch)):
            g = group_data_batch[b]
            num_atom = 0
            for i in range(len(g)):
                subgraph_ = torch.tensor(g[i])
                sub_batch_idx[subgraph_ + shift] = cnt
                group_idx.append(b)
                cnt += 1
                num_atom = max(num_atom, int(torch.max(subgraph_)))
            shift = shift + num_atom + 1

        labels = torch.tensor([item for sublist in piece_data_batch for item in sublist], dtype=torch.long).flatten()
        group_idx = torch.tensor(group_idx, dtype=torch.long)
        node_rep2 = self.pool(node_rep2, sub_batch_idx.to(node_rep2.device))
        mask_indices = []
        left = 0

        if mask_subgraph_indices is None:
            for b in range(len(group_data_batch)):
                # num_subgraph = torch.sum(group_idx == b) 
                num_subgraph = len(group_data_batch[b])
                drop_size = int(num_subgraph * self.mask_rate + 1)
                if num_subgraph == 1:   
                    drop_size = 0
                mask_subgraph = random.sample(range(left, left + num_subgraph), drop_size)
                node_rep2[mask_subgraph] = self.embedding
                mask_indices.extend(list(mask_subgraph))
                left += num_subgraph
            return node_rep2, mask_indices, labels.to(node_rep2.device), group_idx, sub_batch_idx
        else:
            for b in range(len(group_data_batch)):
                # num_subgraph = torch.sum(group_idx == b) 
                num_subgraph = len(group_data_batch[b])
                mask_subgraph = mask_subgraph_indices[b] + int(left)
                node_rep2[mask_subgraph] = self.embedding
                mask_indices.extend(list(mask_subgraph))
                left += num_subgraph
            return node_rep2, mask_indices, labels.to(node_rep2.device), group_idx, sub_batch_idx


class JointMask:
    def __init__(self, mask_rate) -> None:
        self.mask_rate = mask_rate

    def __call__(self, group_data_batch):

        mask_subgraphs = []
        mask_atoms = []

        for b in range(len(group_data_batch)):
            num_subgraph = len(group_data_batch[b])
            drop_size = int(num_subgraph * self.mask_rate + 1)
            if num_subgraph == 1:   
                drop_size = 0
            sample_range = [i for i in range(0, num_subgraph)]
            mask_subgraph_indices =  np.random.choice(sample_range, drop_size, replace=False)
            mask_subgraphs.append(mask_subgraph_indices)

            mask_atom_indices = []
            for ind in mask_subgraph_indices:
                mask_atom_indices.extend(group_data_batch[b][ind])
            mask_atoms.append(mask_atom_indices)
        return mask_subgraphs, mask_atoms
            