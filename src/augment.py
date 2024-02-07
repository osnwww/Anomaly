import copy
import random
import scipy.sparse as sp
import numpy as np
import torch
import math
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_dense_adj
from typing import List, Tuple
from augment_utils import split_class_graphs,align_graphs,stat_graph,universal_svd,graphon_to_graph,add_self_loops,prepare_dataset_x
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
def gen_counter_data(path,round):
    graphs =[]
    data = torch.load(path+"/train_graph"+str(round)+".pt")
    avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density = stat_graph(data)
    resolution = int(median_num_nodes)
    class_graphs = split_class_graphs(data)
    pyg_graph = []
    align_graph = []
    graphons = []  # on label mixup
    for label, all_graphs in class_graphs:
            # logger.info(f"label: {label}, num_graphs:{len(graphs)}" )
        align_graphs_list, normalized_node_degrees, max_num, min_num = align_graphs(
            all_graphs, padding=True)
        align_graph.append((label,align_graphs_list))
        align_graphs_list1 = []
        for graphs in align_graphs_list:
            align_graphs_list1.append(graphs[:resolution,:resolution])
        graphon = universal_svd(align_graphs_list1, threshold=0.2)

        graphons.append((label, graphon))
        # logger.info( f"create new graph" )
        # for graphon in graphons:
        #     pyg_graph.append(graphon_to_graph(graphon))
    aug_graphs = []
    for label, all_graphs in align_graph:
        for label1,graphon in graphons:
            if label1 != label[0]:
                for graph in all_graphs:
                    if graph.shape[0]> median_num_nodes:
                        graph[:int(median_num_nodes),:int(median_num_nodes)] = graphon
                        new_graph = (label1,graph)
                        aug_graphs.append(graphon_to_graph(new_graph))
    torch.save(aug_graphs,path+"/aug_graph"+str(round)+".pt")
                    
            