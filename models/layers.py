# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 16:33:52 2019

@author: lenovo
"""
import torch as th
from torch import nn
import torch.nn.functional as F

import dgl.function as fn
from dgl.nn.pytorch.softmax import edge_softmax

class TSPConv(nn.Module):
    def __init__(self,
                 in_feats):
        super(TSPConv, self).__init__()
        self._in_feats = in_feats
        
        self.W_list = nn.ModuleList([nn.Linear(in_feats, in_feats, bias=False) for i in range(5)])
        self.relu = nn.ReLU()
        self.edge_batch_norm = nn.BatchNorm1d(in_feats, track_running_stats=False)
        self.node_batch_norm = nn.BatchNorm1d(in_feats, track_running_stats=False)

    def forward(self, graph, n_feat, e_feat):
        graph = graph.local_var()
        for i in range(4):
            graph.ndata[f'W_{i}h'] = self.W_list[i](n_feat)  
       
        # update e_feat
        new_e = self.edge_update(graph, e_feat)
        
        # update n_feat
        new_h = self.node_update(graph, e_feat, n_feat)        
        return new_h, new_e
    
    def edge_update(self, graph, e_feat):
        W_4e =  self.W_list[4](e_feat)  
        graph.apply_edges(fn.u_add_v("W_0h", "W_1h", "e_tmp"))
        e_ = self.relu(self.edge_batch_norm(graph.edata['e_tmp'] + W_4e))
        return e_ + e_feat
    
    def node_update(self, graph, e_feat, n_feat): 
        graph.edata['a'] = edge_softmax(graph, e_feat)
        graph.update_all(fn.u_mul_e('W_3h', 'a', 'm'),
                         fn.sum('m', 'n_tmp'))
        n_ = self.relu(self.node_batch_norm(graph.ndata['n_tmp'] + graph.ndata['W_2h']))
        return n_ + n_feat

 
class MLP(nn.Module):
    """Multi-layer Perceptron for output prediction.
    """

    def __init__(self, hidden_dim, output_dim, L=2):
        super(MLP, self).__init__()
        self.L = L
        U = []
        for layer in range(self.L - 1):
            U.append(nn.Linear(hidden_dim, hidden_dim, True))
        self.U = nn.ModuleList(U)
        self.V = nn.Linear(hidden_dim, output_dim, True)

    def forward(self, x):
        """
        Args:
            x: Input features (batch_size, hidden_dim)
        Returns:
            y: Output predictions (batch_size, output_dim)
        """
        Ux = x
        for U_i in self.U:
            Ux = U_i(Ux)  # B x H
            Ux = F.relu(Ux)  # B x H
        y = self.V(Ux)  # B x O
        return y