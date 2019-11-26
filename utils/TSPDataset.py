import torch as th
import torch.nn.functional as F
import torch.nn as nn

import time
import numpy as np
import pandas as pd

import dgl
import dgl.function as fn
from dgl.data.utils import load_graphs, save_graphs

from concorde.tsp import TSPSolver

class TSPDataset(object):
    def __init__(
        self,
        num_samples = 10000,
        num_nodes = 10,
        node_dim = 2,
        num_neighbors = -1,
        file_name = None,
        load_mode = 'read',
        set_type = 'train',
        seed = 0
        ):
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.num_neighbors = num_neighbors
        if file_name == None:
            self.file_name = f"./data/tsp{num_nodes}_concorde_{num_samples}_{set_type}.bin"
        else:
            self.file_name = file_name
        self.load_mode = load_mode
        self.seed = seed
        self._load()
        
    def _load(self):
        if self.load_mode == 'read':
            self.graphs, tour_dict = load_graphs(self.file_name)
            self.tours = tour_dict['tsp_tours']
        else:
            print('Start generating dataset...')
            self.graphs, tour_dict = self.generate_data(
                            self.num_samples,
                            self.num_nodes,
                            self.node_dim,
                            self.num_neighbors,
                            self.file_name,
                            self.seed)
            self.tours = tour_dict['tsp_tours']
            
    def generate_data(
                self,
                num_samples,
                num_nodes,
                node_dim = 2,
                num_neighbors = -1,
                file_name = None,
                seed = 0
        ):
        """
        return graph dataset
        """
        np.random.seed(seed)
        set_nodes_coord = np.random.random([num_samples, num_nodes, node_dim])
        graphs, labels = [],[]
        start_time = time.time()
        for nodes_coord in set_nodes_coord:
            #compute complete graph
            g = dgl.transform.knn_graph(th.tensor(nodes_coord), num_nodes)        
            solver = TSPSolver.from_data(nodes_coord[:,0], nodes_coord[:,1], norm="GEO")  
            solution = solver.solve()
            nodes_coord = th.tensor(nodes_coord).float()
            g.ndata['coord'] = nodes_coord
            g.apply_edges(fn.u_sub_v('coord', 'coord', 'e'))
            #distance
            g.edata['e'] = th.norm(g.edata['e'], dim = 1)
            #neighbors embedding
            e_tags = th.zeros(g.number_of_edges(), 3).float()
            e_tags[:,0] = 1.
            if num_neighbors != -1:
                knn = dgl.transform.knn_graph(nodes_coord, num_neighbors)  
                #remove self loop
                knn = dgl.transform.remove_self_loop(knn) 
                src, dst = knn.edges()
                edge_nb = g.edge_ids(src, dst)
                e_tags[edge_nb, :] = th.tensor([0.,1.,0.])
                #for self loop
                self_loop_id = g.edge_ids(list(range(num_nodes)),
                                          list(range(num_nodes)))
        
                e_tags[self_loop_id, :] = th.tensor([0.,0.,1.])
            g.edata['e'] = th.cat((g.edata['e'].unsqueeze(1), e_tags), dim = 1)
            graphs.append(g)
            labels.append(solution.tour)
        graph_labels = {'tsp_tours': th.tensor(labels).long()}
        save_graphs(file_name, graphs, graph_labels)
        end_time = time.time() - start_time
        print(f"Completed generation of {num_samples} samples of TSP{num_nodes}.")
        print(f"Total time: {end_time/60:.1f}min")
        print(f"Average time: {(end_time/60)/num_samples:.2f}min")
        print(f"Saved at {file_name}")
        return graphs, graph_labels

    def __getitem__(self, item):
        g,t = self.graphs[item],self.tours[item]
        w = self.tour2w(t)
        return g, w, t
    
    def __len__(self):
        """Length of the dataset
        Returns
        -------
        int
            Length of Dataset
        """
        return len(self.graphs)

    def tour2w(self, tour):
        """
        input: num_nodes
        return: B*(V*V)
        """
        num_nodes = tour.size()[0]
        w = th.zeros(num_nodes , num_nodes)
        src, dst = tour[:-1].long(), tour[1:].long()
        w[src, dst] = 1
        w[tour[-1], tour[0]] = 1
        return w.long()
    def collate_tspgraphs(self,data):
        """
        data: list of 2-tuples, w is the flat adj matrix
        """
        graphs, w, t = map(list, zip(*data))
        bg = dgl.batch(graphs)
        ws, tours = th.stack(w, dim = 0),th.stack(t, dim = 0)
        return bg, ws, tours