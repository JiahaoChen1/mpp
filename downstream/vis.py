import argparse

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim

# from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd
import matplotlib.pyplot as plt

import os
# import shutil
import sys


criterion = nn.BCEWithLogitsLoss(reduction = "none")


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []
    y_scores_add = []
    gates = []


    for step, batch in enumerate(loader):
        graph = batch[0].to(device)
        sub_graph = batch[1].to(device)
        re_graph = batch[2].to(device)
        with torch.no_grad():

            pred, pred_add,_ = model((graph.x, graph.edge_index, graph.edge_attr, graph.batch), 
                        (sub_graph.x, sub_graph.edge_index, sub_graph.edge_attr, sub_graph.groups, sub_graph.subgraph_weights),
                        re_graph)
            
        y_true.append(graph.y.view(pred.shape))
        y_scores.append(pred)
        y_scores_add.append(pred_add)
        # gates.append(gate)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()
    y_scores_add = torch.cat(y_scores_add, dim=0).cpu().numpy()

    # gate = torch.cat(gates, dim=0).cpu().numpy()

    roc_list = []
    roc_list_add = []
    roc_list_final = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0

            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))
            roc_list_add.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores_add[is_valid,i]))
            roc_list_final.append(roc_auc_score((y_true[is_valid,i] + 1)/2, (y_scores_add[is_valid,i] + y_scores[is_valid,i] )) )


    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return y_true, y_scores, y_scores_add



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'sider', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    # parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 1, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')


    parser.add_argument("--vocab_path", type=str, default='../ps/zinc250_table_size100.txt', help='the path of the fist subgraph')
    parser.add_argument('--share', type=int, default=2, help='number of workers for dataset loading')

    parser.add_argument('--clean', action='store_true', default=False, help='clean current folder')

    args = parser.parse_args()


    if args.clean:
        try:
            p = "/data00/jiahao/mole_supervise/dataset/" + args.dataset
            os.remove(os.path.join(p, 'processed/geometric_data_processed.pt'))
            os.remove(os.path.join(p, 'processed/pre_filter.pt'))
            os.remove(os.path.join(p, 'processed/pre_transform.pt'))
            os.remove(os.path.join(p, 'processed/process.pkl'))
            os.remove(os.path.join(p, 'processed/smiles.csv'))
        except:
            pass

    print(args)
    #Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")

    # runseed = args.seed
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    
    print('run seed is {}'.format(args.runseed))
    
    dataset = MoleculeDataset("/data00/jiahao/mole_supervise/dataset/" + args.dataset, vocab_path=args.vocab_path, dataset=args.dataset)

    if args.split == "scaffold":
        smiles_list = pd.read_csv('/data00/jiahao/mole_supervise/dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset, sm = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, return_smiles=True)
        # with open('/data00/jiahao/mole_supervise/torch-geo/{}.csv'.format(args.dataset), 'w') as f:
        #     f.writelines(sm)
        # assert 0
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv('/data00/jiahao/mole_supervise/dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.runseed)
        # print("random scaffold, seed is {}".format(seeds[i]))
    else:
        raise ValueError("Invalid split option.")


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)


    #set up model
    model = GNN_graphpred(args.num_layer, args.share, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    if not args.input_model_file == "":
        model.load_state_dict(torch.load(args.input_model_file, map_location='cpu'), strict=False)
    
    model.to(device)

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    train_acc_list2 = []
    val_acc_list2 = []
    test_acc_list2 = []

    train_acc_list3 = []
    val_acc_list3 = []
    test_acc_list3 = []

    # y_true, y_scores, y_scores_add = eval(args, model, device, test_loader)

    model.eval()
    y_scores = []
    y_scores_add = []

    for step, batch in enumerate(val_loader):
        graph = batch[0].to(device)
        sub_graph = batch[1].to(device)
        re_graph = batch[2].to(device)
        with torch.no_grad():

            s1, s2 = model((graph.x, graph.edge_index, graph.edge_attr, graph.batch), 
                        (sub_graph.x, sub_graph.edge_index, sub_graph.edge_attr, sub_graph.groups, sub_graph.subgraph_weights),
                        re_graph)
            
        y_scores.append(s1)
        y_scores_add.append(s2)

    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()
    y_scores_add = torch.cat(y_scores_add, dim=0).cpu().numpy()

    valid_smiles = sm[1]
    # print(valid_smiles[56])
    print(valid_smiles[131])
    # print(y_scores.shape)
    #122, 0.88, 0.34
    #105 0.32, 0.89
    # for i in range(y_scores.shape[0]):
    #     print(y_scores[i], y_scores_add[i], i)

    # plt.hist(y_scores, range=(0,1), bins=20, density=True, alpha=0.5, label='Atom-wise')
    # plt.hist(y_scores_add, range=(0,1), bins=20, density=True, alpha=0.5, label='Subgraph-wise')
    # plt.legend()
    # plt.xlabel('Score')
    # plt.ylabel('Denisty')
    # plt.ylim(0, 6)
    # plt.savefig(f'{args.dataset}.png')

    
    # y_true = y_true[:,5]
    # y_scores = y_scores[:, 5]
    # y_scores_add = y_scores_add[:, 5]
    
    # select = y_true == -1
    # y_indices = np.array([i for i in range(y_scores.shape[0])])
    # plt.scatter(y_scores[select], y_indices[select], c=y_true[select])
    # plt.savefig('test.png')

    # y_true = y_true[:,0]
    # y_scores = y_scores[:, 0]
    # y_scores_add = y_scores_add[:, 0]
    # select = (y_true == 1)
    # print(sm[112])
    # print(select[112])
    # print(y_scores[select])
    # print(y_scores_add[select])

if __name__ == "__main__":
    main()
