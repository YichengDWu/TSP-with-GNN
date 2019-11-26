
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from models.layers import MLP, TSPConv

class TSPModel(nn.Module):
    def __init__(self, config):
        super(TSPModel, self).__init__()
        # Define net parameters
        self.num_nodes = config['num_nodes']
        self.node_dim = config['node_dim']
        self.voc_nodes_in = config['voc_nodes_in']
        self.voc_nodes_out = config['num_nodes']  # config['voc_nodes_out']
        self.voc_edges_in = config['voc_edges_in']
        self.voc_edges_out = config['voc_edges_out']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.mlp_layers = config['mlp_layers']
        # Node and edge embedding layers/lookups
        self.nodes_embedding = nn.Linear(self.node_dim, self.hidden_dim, bias=False)
        self.edges_embedding = nn.Linear(1+self.voc_edges_in, self.hidden_dim, bias=False)

        gcn_layers = []
        for layer in range(self.num_layers):
            gcn_layers.append(TSPConv(self.hidden_dim))
        self.gcn_layers = nn.ModuleList(gcn_layers)
        # Define MLP classifiers
        self.mlp_edges = MLP(self.hidden_dim, self.voc_edges_out, self.mlp_layers)

    def forward(self, graph, n_feat, e_feat):
        """
        """
        x = self.nodes_embedding(n_feat)
        e = self.edges_embedding(e_feat)
        for layer in range(self.num_layers):
            x, e = self.gcn_layers[layer](graph, x, e)  # (B x V) x H, (B x V x V) x H
        # MLP classifier
        y_pred_edges = self.mlp_edges(e)  # (B x V x V) x 2
        bs = graph.batch_size
        y_pred_edges = y_pred_edges.view(bs, self.num_nodes,self.num_nodes, 2)
        # B x V x V x H
        return y_pred_edges

def regress(model, bg):
    x = bg.ndata['coord']
    e = bg.edata['e']
    #x, e = x.cuda(), e.cuda()
    return model(bg, x, e)