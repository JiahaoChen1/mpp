import argparse

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

import os
import sys
import math

criterion = nn.BCEWithLogitsLoss(reduction = "none")


def label_smooth(pred, label):
    label[label == 1] = 0.9
    label[label == -1] = 0.1
    pred = pred.sigmoid()
    loss = label * torch.log(pred + 1e-7) + (1 - label) * torch.log(1 - pred + 1e-7)
    return -loss

def label_smooth2(pred, label):
    label[label == 1] = 0.95
    label[label == -1] = 0.05
    pred = pred.sigmoid()
    loss = label * torch.log(pred + 1e-7) + (1 - label) * torch.log(1 - pred + 1e-7)
    return -loss

def train(args, model, device, loader, optimizer, e):
    model.train()

    for step, batch in enumerate(loader):
        graph = batch[0].to(device)
        sub_graph = batch[1].to(device)
        re_graph = batch[2].to(device)
        pred, pred_add, pred_final = model((graph.x, graph.edge_index, graph.edge_attr, graph.batch), 
                     (sub_graph.x, sub_graph.edge_index, sub_graph.edge_attr, sub_graph.groups, sub_graph.subgraph_weights),
                     re_graph)

        y = graph.y.view(pred.shape).to(torch.float64)
        is_valid = y**2 > 0

        loss_mat = criterion(pred.double(), (y+1)/2)
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
        
        loss_mat_add = label_smooth(pred_add.double(), (y+1)/2)
        loss_mat_add = torch.where(is_valid, loss_mat_add, torch.zeros(loss_mat_add.shape).to(loss_mat_add.device).to(loss_mat_add.dtype))
        
        loss_mat_final = criterion(pred_final.double(), (y+1)/2)
        loss_mat_final = torch.where(is_valid, loss_mat_final, torch.zeros(loss_mat_final.shape).to(loss_mat_final.device).to(loss_mat_final.dtype))

        # loss_mat_gate = criterion(pred_gate.double(), (y+1)/2)
        # loss_mat_gate = torch.where(is_valid, loss_mat_gate, torch.zeros(loss_mat_gate.shape).to(loss_mat_gate.device).to(loss_mat_gate.dtype))

        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss_add = torch.sum(loss_mat_add) / torch.sum(is_valid)
        loss_final = torch.sum(loss_mat_final) / torch.sum(is_valid)
        # loss_gate = torch.sum(loss_mat_gate) / torch.sum(is_valid)

        loss = loss * 1. + loss_add * 1. + loss_final * 1. #+ loss_gate 
        loss.backward()
        optimizer.step()


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

            pred, pred_add, pred_final = model((graph.x, graph.edge_index, graph.edge_attr, graph.batch), 
                        (sub_graph.x, sub_graph.edge_index, sub_graph.edge_attr, sub_graph.groups, sub_graph.subgraph_weights),
                        re_graph)
            
        y_true.append(graph.y.view(pred.shape))
        y_scores.append(pred)
        y_scores_add.append(pred_add)
        gates.append(pred_final)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()
    y_scores_add = torch.cat(y_scores_add, dim=0).cpu().numpy()

    gates = torch.cat(gates, dim=0).cpu().numpy()

    roc_list = []
    roc_list_add = []
    roc_list_final = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0

            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))
            roc_list_add.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores_add[is_valid,i]))
            roc_list_final.append(roc_auc_score((y_true[is_valid,i] + 1)/2, gates[is_valid,i]) )

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list), sum(roc_list_add)/len(roc_list), sum(roc_list_final)/len(roc_list)



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


    # parser.add_argument("--vocab_path", type=str, default='../ps/zinc250_table_size300.txt', help='the path of the fist subgraph')
    parser.add_argument("--vocab_path", type=str, default='brics', help='the path of the fist subgraph')
    parser.add_argument('--share', type=int, default=2, help='number of workers for dataset loading')

    parser.add_argument('--clean', action='store_true', default=False, help='clean current folder')

    args = parser.parse_args()

    save_name = args.vocab_path.split('/')[-1][:-4] if 'brics' not in args.vocab_path else 'brics'
    if args.input_model_file:
        save_name_model = args.input_model_file.split('/')[-2]
    else:
        save_name_model = ''
    father_path = f'{args.dataset}_{args.share}_{save_name}_{save_name_model}'
    exp_path = f'../results/{father_path}/'

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    
    sys.stdout = open(os.path.join(exp_path, f'{args.runseed}.txt'), 'w')

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

    # test_val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    #set up model
    model = GNN_graphpred(args.num_layer, args.share, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)
    
    model.to(device)

    # model_param_group = []
    # model_param_group.append({"params": model.gnn.parameters()})
    # model_param_group.append({"params": model.re_gnn.parameters()})
    # model_param_group.append({"params": model.sub_gnn.parameters()})
    # model_param_group.append({"params": model.graph_pred_linear.parameters(),})
    # model_param_group.append({"params": model.graph_pred_linear2.parameters(),})
    # model_param_group.append({"params": model.graph_pred_linear3.parameters(),})
    # model_param_group.extend([{"params":model.w_l.parameters(), "lr":args.lr, "weight_decay":1e-4}, 
    #                           {"params":model.w_v.parameters(), "lr":args.lr, },])
    # optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    # print(optimizer)
    # assert 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    output_val, output_test = 0., 0.
    output_val2, output_test2 = 0., 0.
    output_val3, output_test3 = 0., 0.

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    train_acc_list2 = []
    val_acc_list2 = []
    test_acc_list2 = []

    train_acc_list3 = []
    val_acc_list3 = []
    test_acc_list3 = []

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        train(args, model, device, train_loader, optimizer, e=epoch)
        # train_gate(args, model, device, val_loader, optimizer_gate)
        
        print("====Evaluation")
        if args.eval_train:
            train_acc, train_acc2, train_acc3 = eval(args, model, device, train_loader)
        else:
            print("omit the training accuracy computation")
            train_acc, train_acc2, train_acc3 = 0, 0, 0
        val_acc, val_acc2, val_acc3 = eval(args, model, device, val_loader)
        test_acc, test_acc2, test_acc3 = eval(args, model, device, test_loader)
        
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)

        val_acc_list2.append(val_acc2)
        test_acc_list2.append(test_acc2)
        train_acc_list2.append(train_acc2)

        val_acc_list3.append(val_acc3)
        test_acc_list3.append(test_acc3)
        train_acc_list3.append(train_acc3)

        if val_acc > output_val:
            output_val = val_acc
            output_test = test_acc
        
        if val_acc2 > output_val2:
            output_val2 = val_acc2
            output_test2 = test_acc2
        
        if val_acc3 > output_val3:
            output_val3 = val_acc3
            output_test3 = test_acc3

        print("train: %f val: %f test: %f best test %f" %(train_acc, val_acc, test_acc, output_test))
        print("train: %f val: %f test: %f best test %f" %(train_acc2, val_acc2, test_acc2, output_test2))
        print("train: %f val: %f test: %f best test %f" %(train_acc3, val_acc3, test_acc3, output_test3))


    df = pd.DataFrame({'train_atom':train_acc_list,'valid_atom':val_acc_list,'test_atom':test_acc_list, 
                       'train_subgraph':train_acc_list2,'valid_subgraph':val_acc_list2,'test_subgraph':test_acc_list2,
                       'train_merge':train_acc_list3,'valid_merge':val_acc_list3,'test_merge':test_acc_list3})
    
    df.to_csv(exp_path + 'seed{}.csv'.format(args.runseed))

    logs = 'Dataset:{}, Best Acc atom:{:.5f}, Best Acc subgraph:{:.5f}, Best Acc merge:{:.5f},'\
                        .format(args.dataset, test_acc_list[val_acc_list.index(max(val_acc_list))], test_acc_list2[val_acc_list2.index(max(val_acc_list2))], test_acc_list3[val_acc_list3.index(max(val_acc_list3))])
    
    with open(exp_path + '{}_log.csv'.format(args.dataset),'a+') as f:
        f.write('\n')
        f.write(logs)


if __name__ == "__main__":
    main()
