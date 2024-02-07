import argparse

from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix
import torch
import numpy as np
import sys
from sklearn import svm
from augment import gen_counter_data
from utils import load_dataset, pos_graphs_pool, print_dataset_stat,prepare_dataset_x
from GNN import GmapAD_GCN, GmapAD_GAT, train_gnn,GmapAD_SAGE,GADAll_GCN,GmapAD_GIN
from evolution import evolution_svm
import os
import random
import logging
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=5, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='IMDB-BINARY', help="['IMDB-BINARY', 'REDDIT-BINARY','REDDIT-MULTI-5K','IMDB-MULTI']")
    parser.add_argument('--ds_rate', type=float, default=0.1, help='Dataset downsampling rate for Graph classification datasets.')
    parser.add_argument('--ds_cl', type=int, default=0, help='The default downsampled class.')

    # GNN related parameters
    parser.add_argument('--gnn_layer', type=str, default='GCN', help="['GCN','GAT','GIN']")
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--learning_rate', default=0.005, help='Learning rate of the optimiser.')
    parser.add_argument('--weight_decay', default=5e-4, help='Weight decay of the optimiser.')
    parser.add_argument('--train_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=0.6)
    parser.add_argument('--tol', type=int, default=300)
    parser.add_argument('--early_stop', type=bool, default=True, help="Early stop when training GNN")

    # Node pooling related parameters
    parser.add_argument('--n_p_g', type=str, default='positive', help="['positive', 'negative']")
    parser.add_argument('--n_p_stg', type=str, default='mean', help="['mean','max', 'min']")
    
    # Evolving related parameters
    parser.add_argument('--w_stg', type=str, default='one-hot', help="['one-hot']")
    parser.add_argument('--clf', type=str, default='svm', help="['svm', 'others']")
    parser.add_argument('--mut_rate', type=float, default=0.5, help="['svm','nb', 'others']")
    parser.add_argument('--cros_rate', type=float, default=0.9, help="['svm','nb', 'others']")
    parser.add_argument('--evo_gen', type=int, default=2000, help="number of evolution generations")
    parser.add_argument('--cand_size', type=int, default=64, help="candidates in each generation")

    # Model hyperparameters
    parser.add_argument('--gnn_dim', type=int, default=128)
    parser.add_argument('--fcn_dim', type=int, default=32)
    parser.add_argument('--gce_q', type=int, default=0.7, help='gce q')
    parser.add_argument('--alpha', type=float, default=1.5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--topk', type=int, default=64, help="number of the most informative nodes, this parameter also decides the finally graph embedding dimension.")

    # For GAT only, num of attention heads
    parser.add_argument('--gat_heads', default=8, help='GAT heads')

    # Test round
    parser.add_argument('--round', type=int, default=1, help='test round')

    args = parser.parse_args()
    return args

def downsample(ds_rate, ds_cl, graphs):
    if args.dataset not in ['KKI', 'OHSU']:
        ds_rate = args.ds_rate
        ds_cl = args.ds_cl
        ds_graphs = []
        all_graphs = []
        num_nodes = 0
        for graph in graphs:
            num_nodes += graph.num_nodes
            if graph.y == ds_cl:
                ds_graphs.append(graph)
            all_graphs.append(graph)
        ds_graphs = ds_graphs[int(len(ds_graphs)*ds_rate):]
        [all_graphs.remove(graph) for graph in ds_graphs]
        return all_graphs
    else:
        return graphs

if __name__ == "__main__":

    args = arg_parser()
    logging.basicConfig(filename=f"./train_log/{args.dataset}+GmapAD-{args.gnn_layer}.log",filemode="a",format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",level=logging.INFO)
    logger=logging.getLogger('GmapAD')    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    print(f"Training device: {device}")
    print(f"loading dataset {args.dataset}")
    print(f"Testing Round: {args.round}")

    graph_path = f"data/{args.dataset}/{args.gnn_layer}/graph{args.round}.pt"
    train_path = f"data/{args.dataset}/{args.gnn_layer}/train_graph{args.round}.pt"
    val_path = f"data/{args.dataset}/{args.gnn_layer}/val_graph{args.round}.pt"
    test_path = f"data/{args.dataset}/{args.gnn_layer}/test_graph{args.round}.pt"
    
    if not os.path.exists(graph_path) or not os.path.exists(train_path) or not os.path.exists(val_path) or not os.path.exists(test_path):
        graphs = load_dataset(args.dataset, args)
        if args.dataset in ['KKI',"REDDIT-MULTI-5K",'OHSU','REDDIT-BINARY','IMDB-BINARY','IMDB-MULTI']:
            random.shuffle(graphs)
        else:
            graphs = graphs.shuffle()
        if not  os.path.exists(f"data/{args.dataset}/{args.gnn_layer}"):
            os.makedirs(f"data/{args.dataset}/{args.gnn_layer}")
        torch.save(graphs, f"data/{args.dataset}/{args.gnn_layer}/graph{args.round}.pt")
        train_ratio = args.train_ratio
        val_ratio = args.test_ratio
        train_graphs = graphs[:int(len(graphs)*train_ratio)]
        val_graphs = graphs[int(len(graphs)*train_ratio): int(len(graphs)*(train_ratio+val_ratio))]
        test_graphs = graphs[int(len(graphs)*(train_ratio+val_ratio)):]
        # Downsampling
        if args.dataset not in ['REDDIT-MULTI-5K']:
            train_graphs = downsample(args.ds_rate, args.ds_cl, train_graphs)
            val_graphs = downsample(args.ds_rate, args.ds_cl, val_graphs)
            test_graphs = downsample(args.ds_rate, args.ds_cl, test_graphs)        
        # Save downsampled datasets
        torch.save(train_graphs, f"data/{args.dataset}/{args.gnn_layer}/train_graph{args.round}.pt")
        torch.save(val_graphs, f"data/{args.dataset}/{args.gnn_layer}/val_graph{args.round}.pt")
        torch.save(test_graphs, f"data/{args.dataset}/{args.gnn_layer}/test_graph{args.round}.pt")

    else:
        print("load from pre-splitted data.")
        graphs = torch.load(f"data/{args.dataset}/{args.gnn_layer}/graph{args.round}.pt")
        train_graphs = torch.load(f"data/{args.dataset}/{args.gnn_layer}/train_graph{args.round}.pt")
        val_graphs = torch.load(f"data/{args.dataset}/{args.gnn_layer}/val_graph{args.round}.pt")
        test_graphs = torch.load(f"data/{args.dataset}/{args.gnn_layer}/test_graph{args.round}.pt")
    gen_counter_data(f"data/{args.dataset}/{args.gnn_layer}",args.round)
    # aug_data = 
    aug_graphs = torch.load(f"data/{args.dataset}/{args.gnn_layer}/aug_graph{args.round}.pt")
    # aug_graphs = random.sample(aug_graphs, 40)

    # train_graphs=train_graphs + aug_graphs[-30:]   
    # train_graphs=train_graphs + aug_graphs[3:]   
    print_dataset_stat(args, graphs)
    
    train_graphs = prepare_dataset_x(train_graphs,3062,args)
    # t = []
    # for graph in train_graphs:
    #     if graph.y == 1:
    #         t.append(graph)
    # train_graphs = t
    # for data in aug_graphs:
    #     train_graphs.append(data.cpu())
    val_graphs = prepare_dataset_x(val_graphs,3062,args)
    test_graphs = prepare_dataset_x(test_graphs,3062,args)   
    if args.gnn_layer == "GCN":
        model = GmapAD_GCN(num_nodes=train_graphs[0].x.shape[0], input_dim=train_graphs[0].x.shape[1], hidden_channels=args.gnn_dim, num_classes=2)
    elif args.gnn_layer == "GAT":
        model = GmapAD_GAT(num_nodes=train_graphs[0].x.shape[0], input_dim=train_graphs[0].x.shape[1], hidden_channels=args.gnn_dim, num_classes=2, num_heads=args.gat_heads)
    elif args.gnn_layer == "GAD":
        model = GmapAD_GCN(num_nodes=train_graphs[0].x.shape[0], input_dim=train_graphs[0].x.shape[1], hidden_channels=args.gnn_dim, num_classes=2)
        model = GmapAD_GCN(num_nodes=train_graphs[0].x.shape[0], input_dim=train_graphs[0].x.shape[1], hidden_channels=args.gnn_dim, num_classes=2)
    elif args.gnn_layer == "GIN":
        model = GmapAD_GIN(num_nodes=train_graphs[0].x.shape[0], input_dim=train_graphs[0].x.shape[1], hidden_channels=args.gnn_dim, num_classes=2)

    else:
        model = GmapAD_SAGE(num_nodes=train_graphs[0].x.shape[0], input_dim=train_graphs[0].x.shape[1], hidden_channels=args.gnn_dim, num_classes=2)
    
    model = model.to(device)
    print(f"Start training model {args.gnn_layer}")
    train_gnn(model, train_graphs, val_graphs, test_graphs, args)

    # Get the candidate pool, grpah reprsentations
    pos_graphs = []
    neg_graphs = []

    for graph in train_graphs:
        if graph.y == 1:
            pos_graphs.append(graph)
       
        else:
            neg_graphs.append(graph)
    pos_graphs = prepare_dataset_x(pos_graphs,3062,args)
    neg_graphs = prepare_dataset_x(neg_graphs,3062,args)
    node_pool = pos_graphs_pool(pos_graphs, model, args)#从正常数据中获取
    node_pool = node_pool.cpu()
    print(f"Generating Node pool size: {node_pool.size()}")

    if args.clf == "svm":
        clf = svm.SVC(kernel='linear', C=1.0, cache_size=1000)
        print(f"Test on {args.dataset}, using SVM, graph pool is {args.n_p_g}, node pool stg is {args.n_p_stg}")
        clf, x_train_pred, Y_train, x_val_pred, Y_val, x_test_pred, Y_test = evolution_svm(clf, model, node_pool, args, train_graphs, val_graphs, test_graphs,logger)
