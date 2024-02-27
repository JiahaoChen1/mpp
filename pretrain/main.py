import torch
import torch.nn as nn
import numpy as np
import random
import argparse

from trainer.trainer import Trainer
from model.gnn_model import GNN_graphpred
# from model.gnn_subgraph import GNNSubgraph
from dataset.sup_dataset import get_dataloader
from dataset.tokenizer import Tokenizer


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True




if __name__ == '__main__':
     setup_seed(2022)
     parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
     parser.add_argument('--device', type=int, default=0,
                         help='which gpu to use if any (default: 0)')
     parser.add_argument('--batch_size', type=int, default=32,
                         help='input batch size for training (default: 32)')
     parser.add_argument('--epochs', type=int, default=100,
                         help='number of epochs to train (default: 100)')
     parser.add_argument('--lr', type=float, default=0.001,
                         help='learning rate (default: 0.001)')
     parser.add_argument('--decay', type=float, default=0,
                         help='weight decay (default: 0)')
     parser.add_argument('--num_layer', type=int, default=5,
                         help='number of GNN message passing layers (default: 5).')
     parser.add_argument('--emb_dim', type=int, default=300,
                         help='embedding dimensions (default: 300)')
     parser.add_argument('--dropout_ratio', type=float, default=0.5,
                         help='dropout ratio (default: 0.2)')
     parser.add_argument('--graph_pooling', type=str, default="mean",
                         help='graph level pooling (sum, mean, max, set2set, attention)')
     parser.add_argument('--JK', type=str, default="last",
                         help='how the node features across layers are combined. last, sum, max or concat')
     parser.add_argument('--dataset', type=str, default='./data/zinc250.txt',
                         help='root directory of dataset. For now, only classification.')
     parser.add_argument('--gnn_type', type=str, default="gin")
     parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataset loading')

     parser.add_argument("--subgraph_table", type=str, default='./zinc250_table_size100.txt', help='the path of the fist subgraph')
     parser.add_argument("--interval", type=int, default=20, help='latent size')
     parser.add_argument('--folder', type=str, default='./pretrain_100_ce_smi_no_edge')
     parser.add_argument('--rate', type=float, default=0.15, help='drop part of subgraph')
     parser.add_argument('--pre_load', type=str, default='processed_data_table100_kl_all.pkl', help='drop part of subgraph')

     parser.add_argument('--num_layer_subgraph1', type=int, default=2,
                         help='number of GNN message passing layers (default: 2).')
     parser.add_argument('--num_layer_subgraph2', type=int, default=3,
                         help='number of GNN message passing layers (default: 3).')
     
     parser.add_argument('--cl_loss',  type=str, default='exe', help='clean current folder')
     args = parser.parse_args()
     print(args)

     tokenizer = Tokenizer(args.subgraph_table)
     dataloader = get_dataloader(fname=args.dataset, tokenizer=tokenizer, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                 pre_load=args.pre_load)
     # assert 0
     device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
     model = GNN_graphpred(args.num_layer, args.num_layer_subgraph1, args.num_layer_subgraph2, 
                           args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, 
                           gnn_type=args.gnn_type).to(device)
     print(len(tokenizer.subgraph2idx))

     # assert 0
     trainer = Trainer(dataloader=dataloader, model=model, device=device, 
                       folder=args.folder, interval=args.interval, rate=args.rate, 
                       emb_dim=args.emb_dim, cl_loss=args.cl_loss)
     trainer.train(num_epoch=args.epochs)


    