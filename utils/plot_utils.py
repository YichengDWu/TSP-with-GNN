# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 19:34:23 2019

@author: lenovo
"""
import torch
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

import dgl 

def plot_tsp(g, tour, num_neighbors = -1, title='Default'):
    """
    g: a complete graph
    tour: 1*n
    num_neighbors: int
    """
    n_nodes = g.number_of_nodes()
    coord = g.ndata['coord']
    g_copy = dgl.DGLGraph()
    g_copy.add_nodes(n_nodes)
    g_copy.add_edges(list(tour[:-1]), list(tour[1:]))
    pos = dict(zip(range(n_nodes), coord.numpy()))
    if num_neighbors > 0:
        sub_g = dgl.transform.knn_graph(coord, num_neighbors)  
    else:
        sub_g = g
    nx.draw(sub_g.to_networkx(), with_labels=True, pos = pos, alpha=0.3, width=0.5)
    nx.draw(g_copy.to_networkx(), with_labels=True, pos = pos, alpha=1, width=1, edge_color='r')
    plt.title(title)

