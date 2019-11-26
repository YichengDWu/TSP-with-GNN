import torch as th
import torch.nn as nn
import torch.nn.functional as F

import time
import numpy as np

import dgl
import dgl.function as fn
from dgl.data.utils import load_graphs, save_graphs
from utils.TSPDataset import TSPDataset
from concorde.tsp import TSPSolver

def load_dataset_for_regression(args):
    """Load dataset for regression tasks.
    Parameters
    ----------
    args : dict
        Configurations.
    Returns
    -------
    train_set
        Subset for training.
    val_set
        Subset for validation.
    test_set
        Subset for test.
    """

    if args['train_filepath'] == 'None':
        train_set = TSPDataset(
            num_samples = args['train_num_samples'],
            num_nodes = args['num_nodes'],
            node_dim = args['node_dim'],
            num_neighbors = args['num_neighbors'],
            file_name = None,
            load_mode = 'generate',
            set_type = 'train',
            seed = 0)
        
        val_set = TSPDataset(
            num_samples = args['val_num_samples'],
            num_nodes = args['num_nodes'],
            node_dim = args['node_dim'],
            num_neighbors = args['num_neighbors'],
            file_name = None,
            load_mode = 'generate',
            set_type = 'val',
            seed = 1)
        
        test_set = TSPDataset(
            num_samples = args['test_num_samples'],
            num_nodes = args['num_nodes'],
            node_dim = args['node_dim'],
            num_neighbors = args['num_neighbors'],
            file_name = None,
            load_mode = 'generate',
            set_type = 'test',
            seed = 2)
    else:
        train_set = TSPDataset(
            file_name = args['train_filepath'],
            load_mode = 'read')
        
        val_set = TSPDataset(
            file_name = args['val_filepath'],
            load_mode = 'read')
        
        test_set = TSPDataset(
            file_name = args['test_filepath'],
            load_mode = 'read')

    return train_set, val_set, test_set


def loss_edges(y_pred_edges, y_edges, edge_cw):
    """
    Loss function for edge predictions.
    Args:
        y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes)
        y_edges: Targets for edges (batch_size, num_nodes, num_nodes)
        edge_cw: Class weights for edges loss
    Returns:
        loss_edges: Value of loss function
    
    """
    # Edge loss
    y = F.log_softmax(y_pred_edges, dim=3)  # B x V x V x voc_edges
    y = y.permute(0, 3, 1, 2)  # B x voc_edges x V x V
    loss_edges = nn.NLLLoss(edge_cw)(y, y_edges)
    return loss_edges

class Meter(object):
    """Track and summarize model performance on a dataset for
    (multi-label) binary classification."""
    def __init__(self):
        self.mask = []
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred, y_true, mask):
        """Update for the result of an iteration
        Parameters
        ----------
        y_pred : float32 tensor
            Predicted molecule labels with shape (B, T),
            B for batch size and T for the number of tasks
        y_true : float32 tensor
            Ground truth molecule labels with shape (B, T)
        mask : float32 tensor
            Mask for indicating the existence of ground
            truth labels with shape (B, T)
        """
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        self.mask.append(mask.detach().cpu())

    def length(self):
        """Compute length score for each task.
        Returns
        -------
        list of float
            length for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(roc_auc_score(task_y_true, task_y_pred))
        return scores

    def gap(self):
        """Compute gap loss for each task.
        Returns
        -------
        list of float
            gap loss for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            scores.append(F.l1_loss(task_y_true, task_y_pred, reduction=reduction).item())
        return scores

    def rmse(self):
        """Compute RMSE for each task.
        Returns
        -------
        list of float
            rmse for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            scores.append(np.sqrt(F.mse_loss(task_y_pred, task_y_true).cpu().item()))
        return scores

    def compute_metric(self, metric_name, reduction='mean'):
        """Compute metric for each task.
        Parameters
        ----------
        metric_name : str
            Name for the metric to compute.
        reduction : str
            Only comes into effect when the metric_name is l1_loss.
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task
        Returns
        -------
        list of float
            Metric value for each task
        """
        assert metric_name in ['roc_auc', 'l1', 'rmse'], \
            'Expect metric name to be "roc_auc", "l1" or "rmse", got {}'.format(metric_name)
        assert reduction in ['mean', 'sum']
        if metric_name == 'roc_auc':
            return self.roc_auc_score()
        if metric_name == 'l1':
            return self.l1_loss(reduction)
        if metric_name == 'rmse':
            return self.rmse()
