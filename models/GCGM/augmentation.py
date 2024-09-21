import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import networkx as nx
import math
import scipy.sparse as sp
import re

# * record the k latest weights with a deque to limit memory usage
from collections import Counter, defaultdict, deque
# * create augmentation pairs
from itertools import combinations_with_replacement
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.utils import dropout_adj, subgraph
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, degree, from_scipy_sparse_matrix
from torch_geometric.loader import DataLoader
from torch.distributions import Uniform, Normal
from torch.multiprocessing import Manager, Process
from statistics import mean

class Augmentation():
    def __init__(self, cfg, num_views=2):
        super(Augmentation, self).__init__()
        self.cfg = cfg
        self.num_views = num_views
        self.feature_augmentations = [Augmentation.node_feature_masking, # requires hyperparameter p_nm 
                                      Augmentation.random_amplitude_scaling_single,
                                      Augmentation.random_amplitude_scaling_multivariate,
                                      Augmentation.gaussian_noise,
                                      Augmentation.attribute_shuffling,
                                      Augmentation.mixup
                                      ]

        self.structure_augmentations = [Augmentation.edges_insertion, # requires hyperparameters p_ei 
                                        Augmentation.edge_removing, # requires hyperparameters p_er 
                                        # Augmentation.edges_deletion, # requires hyperparameters p_ed
                                        Augmentation.nodes_insertion, # requires hyperparameters p_ni 
                                        Augmentation.nodes_deletion, # requires hyperparameters p_nd 
                                        Augmentation.sub_sampling, # requires hyperparameters p_sub
                                        Augmentation.nodes_modification,
                                        Augmentation.knn,
                                        Augmentation.knn_thold,
                                        Augmentation.graph_diffusion,
                                        Augmentation.sparse_graph_diffusion,
                                        # Augmentation.graph_coarsen
                                        ]
        
        # * record the augmentations and their parameters
        self.augs = []
        self.params = []
        
        for aug, params in cfg.AUGMENTATION.items():
            # * para is a list of parameters
            for param in params:
                para = Augmentation.parse_para(param)
                if None in para:
                    continue
                # * if the augmentation is not initialized, initialize it randomly
                elif 'random' in para:
                    # * remove the duplicated augmentations
                    n, random_params = Augmentation.random_aug(para)
                    self.params.extend(random_params)
                else:
                    n = 1 
                    self.params.append(para)
                self.augs.extend([x for x in self.feature_augmentations + self.structure_augmentations if x.__name__.upper() == aug.upper()] * n)
        
        # * sample the augmentations
        if self.cfg.BiAS.get('SAMPLE', None) is not None:
            # self.uniform_sample(self.cfg.BiAS.SAMPLE)
            self.stratified_sample(self.cfg.BiAS.SAMPLE)
        
        # * append NA to the augmentations
        self.augs.append(Augmentation.noop)
        self.params.append([])
        self.categories.append('NOOP')
    
        self.indices = list(range(len(self.augs)))
        self.aug_names = ['{}_{}'.format(x.__name__.upper(), i) for i, x in enumerate(self.augs)]
        print('Number of augmentations: {}'.format(len(self.augs)))
        print('Augmentations: {}'.format(', '.join([x.__name__.upper() for x in set(self.augs)])))
        print(dict(zip(self.aug_names, self.params)))
        
        # * create pairs of augmentations and parameters
        # self.aug_pairs = self.create_pairs()
        self.aug_pairs = self.stratified_pairs(self.cfg.BiAS.PAIRS)
        self.aug_pair_names = ['({}_{}, {}_{})'.format(x[0][0].__name__.upper(), x[0][1], x[1][0].__name__.upper(), x[1][1]) for x in self.aug_pairs]
        self.aug_indices = list(range(len(self.aug_pairs)))
        print('Number of augmentation pairs: {}'.format(len(self.aug_pairs)))
        
        # * initialize the weights of the augmentation pairs with the maximum value of the BiAS function
        self.weights = np.full(len(self.aug_pairs), np.exp(self.cfg.BiAS.ALPHA))
        self.avg_f1s = np.zeros(len(self.aug_pairs))
        # * record the number of times each augmentation pair is applied
        self.prev_counts = np.zeros(len(self.aug_pairs))
        # * record a list of augmentation pairs applied for each batch
        self.cur_pairs = []
        self.cur_indices = []
    
    def apply(self, data_dict):
        """ augment the views based on the augmentation pairs
        """
        
        def _apply_aug(data, aug, param):
            aug_name = aug.__name__.upper()
            if aug_name == 'MIXUP':
                data = aug(data, *param)
            elif any(s in aug_name for s in ['NODES_INSERTION', 'NODES_DELETION', 'NODES_MODIFICATION', 'EDGE_REMOVING', 'NODE_FEATURE_MASKING', 'RANDOM_AMPLITUDE_SCALING']):
                data = aug(data, *param)
            elif aug_name == 'EDGES_INSERTION':
                data, inserted = aug(data, max(self.cfg.PROBLEM.RESCALE), *param)
            elif any(s in aug_name for s in ['KNN', 'KNN_THOLD', 'GRAPH_DIFFUSION', 'SPARSE_GRAPH_DIFFUSION', 'GRAPH_COARSEN']):
                data = aug(data, max(self.cfg.PROBLEM.RESCALE), *param)
            else:
                data = aug(data)
            return data
        
        batch_size = data_dict['batch_size'] // len(self.cfg.GPUS)
        
        if self.cfg.BiAS.REWEIGHT:
            # * adjust the weights to be less skewed and preserve the dominant augmentation pairs
            weights = self.weights.copy()
            adjusted_weights = Augmentation.reweight(weights)
            if len(self.aug_pairs) >= 1:
                indices = np.random.choice(self.aug_indices, batch_size, p=adjusted_weights / adjusted_weights.sum())
        else:
            if len(self.aug_pairs) >= 1:
                indices = np.random.choice(self.aug_indices, batch_size, p=self.weights / self.weights.sum())
        
        # * update the current augmentations with len of the batch size
        self.cur_pairs = [self.aug_pairs[i] for i in indices]
        self.cur_indices = indices
        
        views_1, views_2 = [], []
        for i in range(batch_size):
            aug_1, _, param_1 = self.cur_pairs[i][0]
            aug_2, _, param_2 = self.cur_pairs[i][1]
            # * apply the augmentation
            src_graph = _apply_aug(data_dict['src_graphs'][i], aug_1, param_1)
            tgt_graph = _apply_aug(data_dict['tgt_graphs'][i], aug_2, param_2)
            views_1.append(src_graph)
            views_2.append(tgt_graph)
        
        # * adjusted batch_size when using multiple gpus
        loader_1 = DataLoader(views_1, batch_size=batch_size)
        views_1 = next(iter(loader_1))
        loader_2 = DataLoader(views_2, batch_size=batch_size)
        views_2 = next(iter(loader_2))
        
        data_dict.update({'views_1': views_1, 
                          'views_2': views_2})
        
        device = data_dict['src_graphs'][0].x.device
        data_dict.update({
                        'cur_indices': torch.from_numpy(self.cur_indices).to(device),
                        'prev_counts': torch.from_numpy(self.prev_counts).to(device),
                        'avg_f1s': torch.from_numpy(self.avg_f1s).to(device),
                        'weights': torch.from_numpy(self.weights).to(device),
                          })
        return data_dict

    @staticmethod
    def update(prev_counts, cur_counts, cur_indices, avg_f1s, weights, cur_f1s, bias_lambda, bias_alpha, update_weights=True):
        """ update the weight of the current augmentation based on the average f1 scores of the current batch
        better performance -> lower probability
        
        w_batch *= np.exp(alpha * (1 - batch_f1_score)) / scaling_factor
        or we can use a smoothing factor (momentum update)
        w_batch = lambda * w_batch + (1 - lambda) * np.exp(alpha * (1 - batch_f1_score)) / scaling_factor
        """
        for i in set(cur_indices):
            avg_f1s[i] = Augmentation.update_moving_avg(avg_f1s[i], prev_counts[i], torch.sum(cur_f1s[np.where(cur_indices==i)]), cur_counts[i])
            if update_weights:
                weights[i] = bias_lambda * weights[i] + (1 - bias_lambda) * np.exp(bias_alpha * (1 - avg_f1s[i]))
        return weights.copy(), avg_f1s.copy(), cur_counts.copy()
            
    @staticmethod
    def noop(data):
        """ 'no operation' which is an empty augmentation
        """
        data.dummy_label = torch.cat((data.dummy_label, data.dummy_label), dim=1)
        return data
    
    @staticmethod
    def graph_diffusion(data, rescale, alpha, epsilon=1e-5):
        """
        Reference: [Multi-Scale Contrastive Siamese Networks for Self-Supervised Graph Representation Learning](https://arxiv.org/pdf/2105.05682.pdf)
        conduct graph diffusion over the original graph to obtain a diffused graph and S is diffused adjency matrix
        """
        
        def _normalize_adj_tensor(S):
            """normalizes S by the inverse square root of the sum of its adjacent nodes. 
            Specifically, for each node, it calculates the sum of the weights of the edges connected to the node, takes the square root, and then inverts the result. 
            This produces a diagonal matrix where the main diagonal contains the inverse square root degree of each node. 
            Then it multiplies this diagonal matrix with the adjacency matrix and its transpose to normalize the adjacency matrix.
            """
            # convert S to a sparse tensor
            S = S.to_sparse()
            # compute the degree of each node
            deg = torch.sparse.sum(S, dim=1).to_dense()
            # compute D^-1/2
            d_inv_sqrt = deg.pow(-0.5)
            d_inv_sqrt[d_inv_sqrt == float('inf')] = 0
            # compute D^-1/2 * S * D^-1/2
            S_norm = d_inv_sqrt.view(-1, 1) * S.to_dense() * d_inv_sqrt.view(1, -1)
            return S_norm
        
        edge_index = data.edge_index
        num_nodes = data.num_nodes

        # compute the degree matrix and the adjacency matrix
        deg = degree(edge_index[0], num_nodes, dtype=torch.float)
        D = torch.diag(deg)
        # A = to_dense_adj(edge_index).squeeze()
        # ensures that even if a node does not have any edges, it still gets a row and a column in the adjacency matrix
        A = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
        A[edge_index[0], edge_index[1]] = 1
        A = A.squeeze()
        # compute D^-1/2
        D_inv_sqrt = D.pow(-0.5)
        D_inv_sqrt[D_inv_sqrt == float('inf')] = 0
        # calculate the matrix inside the bracket
        eye_tensor = torch.eye(num_nodes, device=edge_index.device)
        inner_matrix = eye_tensor - (1 - alpha) * torch.mm(torch.mm(D_inv_sqrt, A), D_inv_sqrt)
        # invert the matrix and handle singular matrix by adding epsilon
        # epislon = 1e-5
        inner_matrix += eye_tensor * epsilon
        inner_matrix_inv = torch.inverse(inner_matrix)
        # calculate S, which is the diffused adjacency matrix
        S = alpha * inner_matrix_inv
        # normalize S
        S_norm = _normalize_adj_tensor(S).cpu().numpy()
        
        # convert normalized adjacency matrix into edge_index format
        # self-loops in the graph will be excluded in the resulting edge index tensor
        # edge_index, _ = from_dense_adj(S_norm, remove_diag=True)
        np.fill_diagonal(S_norm, 0)
        edge_index = np.nonzero(S_norm)

        P = data.pos.detach().cpu().numpy()
        edge_feat = 0.5 * (np.expand_dims(P, axis=1) - np.expand_dims(P, axis=0)) / rescale + 0.5
        edge_attr = edge_feat[edge_index]
        edge_attr = np.clip(edge_attr, 0, 1)
        
        edge_attr = torch.tensor(edge_attr, device=data.edge_attr.device).to(torch.float32)
        edge_index = torch.tensor(np.array(edge_index), dtype=torch.long, device=data.edge_index.device)
        hyperedge_index = Augmentation.get_hyperedge_index(data.num_nodes, edge_index)
        
        return Data(x=data.x, edge_index=edge_index, edge_attr=edge_attr, 
                    hyperedge_index=hyperedge_index, dummy_label=torch.cat((data.dummy_label, data.dummy_label), dim=1), pos=data.pos)
        
    @staticmethod
    def sparse_graph_diffusion(data, rescale, alpha, beta):
        """ Reference: [Diffusion Improves Graph Learning](https://proceedings.neurips.cc/paper/2019/file/23c894276a2c5a16470e6a31f4618d73-Paper.pdf)
        Sparsify the diffused adjacency matrix S by top-k or thresholding method
        """
        def _normalize_adj_tensor(S):
            """normalizes S by the inverse square root of the sum of its adjacent nodes. 
            Specifically, for each node, it calculates the sum of the weights of the edges connected to the node, takes the square root, and then inverts the result. 
            This produces a diagonal matrix where the main diagonal contains the inverse square root degree of each node. 
            Then it multiplies this diagonal matrix with the adjacency matrix and its transpose to normalize the adjacency matrix.
            """
            # convert S to a sparse tensor
            S = S.to_sparse()
            # compute the degree of each node
            deg = torch.sparse.sum(S, dim=1).to_dense()
            # compute D^-1/2
            d_inv_sqrt = deg.pow(-0.5)
            d_inv_sqrt[d_inv_sqrt == float('inf')] = 0
            # compute D^-1/2 * S * D^-1/2
            S_norm = d_inv_sqrt.view(-1, 1) * S.to_dense() * d_inv_sqrt.view(1, -1)
            return S_norm
        
        def _top_k_sparsify(S, k):
            # for each node, find the top k connections
            topk, idx = torch.topk(S, k, dim=1)
            # create a new sparse matrix
            S_sparse = torch.zeros_like(S)
            S_sparse.scatter_(1, idx, topk)
            return S_sparse

        def _threshold_sparsify(S, epsilon):
            # Create a mask for values greater than the threshold
            mask = (S > epsilon)
            # Apply the mask to the matrix
            S_sparse = S * mask.float()
            return S_sparse
        
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        
        # compute the degree matrix and the adjacency matrix
        deg = degree(edge_index[0], num_nodes, dtype=torch.float)
        D = torch.diag(deg)
        # A = to_dense_adj(edge_index).squeeze()
        # ensures that even if a node does not have any edges, it still gets a row and a column in the adjacency matrix
        A = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
        A[edge_index[0], edge_index[1]] = 1
        A = A.squeeze()
        # compute D^-1/2
        D_inv_sqrt = D.pow(-0.5)
        D_inv_sqrt[D_inv_sqrt == float('inf')] = 0
        # calculate the matrix inside the bracket
        eye_tensor = torch.eye(num_nodes, device=edge_index.device)
        inner_matrix = eye_tensor - (1 - alpha) * torch.mm(torch.mm(D_inv_sqrt, A), D_inv_sqrt)
        # invert the matrix and handle singular matrix by adding epsilon
        inner_matrix += eye_tensor * epsilon
        inner_matrix_inv = torch.inverse(inner_matrix)
        # calculate S, which is the diffused adjacency matrix
        S = alpha * inner_matrix_inv
        
        # sparsify S
        if beta >= 1:
            S_sparse = _top_k_sparsify(S, beta)
        else:
            S_sparse = _threshold_sparsify(S, beta)
        
        # normalize S
        S_sparse_norm = _normalize_adj_tensor(S_sparse).cpu().numpy()
        np.fill_diagonal(S_sparse_norm, 0)
        edge_index = np.nonzero(S_sparse_norm)

        # calculate the edge features and hyperedge index
        P = data.pos.detach().cpu().numpy()
        edge_feat = 0.5 * (np.expand_dims(P, axis=1) - np.expand_dims(P, axis=0)) / rescale + 0.5
        edge_attr = edge_feat[edge_index]
        edge_attr = np.clip(edge_attr, 0, 1)
        
        edge_attr = torch.tensor(edge_attr, device=data.edge_attr.device).to(torch.float32)
        edge_index = torch.tensor(np.array(edge_index), dtype=torch.long, device=data.edge_index.device)
        hyperedge_index = Augmentation.get_hyperedge_index(data.num_nodes, edge_index)
        
        return Data(x=data.x, edge_index=edge_index, edge_attr=edge_attr, 
                    hyperedge_index=hyperedge_index, dummy_label=torch.cat((data.dummy_label, data.dummy_label), dim=1), pos=data.pos)

    @staticmethod
    def graph_coarsen(data, rescale, p_alpha):
        """ create a coarsened graph via graph diffusion
        """
        def _graph_diffusion(data, alpha):
            # data is your torch_geometric.data.Data instance
            edge_index = data.edge_index
            num_nodes = data.num_nodes
            # create the degree matrix
            deg = degree(edge_index[0], num_nodes, dtype=torch.float)
            D = torch.diag(deg)
            # create the adjacency matrix
            # A = to_dense_adj(edge_index).squeeze()
            # ensures that even if a node does not have any edges, it still gets a row and a column in the adjacency matrix
            A = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
            A[edge_index[0], edge_index[1]] = 1
            A = A.squeeze()
            # compute D^-1/2
            D_inv_sqrt = D.pow(-0.5)
            D_inv_sqrt[D_inv_sqrt == float('inf')] = 0
            # calculate the matrix inside the bracket
            inner_matrix = torch.eye(num_nodes) - (1 - alpha) * torch.mm(torch.mm(D_inv_sqrt, A), D_inv_sqrt)
            # invert the matrix
            inner_matrix_inv = torch.inverse(inner_matrix)
            # calculate S
            S = alpha * inner_matrix_inv
            return S

        # compute the diffusion matrix
        S = _graph_diffusion(data, p_alpha)
        # threshold S to get the adjacency matrix of the coarsened graph
        threshold = S.median()
        adj_coarse = (S > threshold).float()
        # create edge_index for coarsened graph
        edge_index_coarse = adj_coarse.nonzero(as_tuple=False).t().numpy()
        # edge features are the normalized coordinates of the nodes
        P = data.pos.detach().cpu().numpy()
        edge_feat = 0.5 * (np.expand_dims(P, axis=1) - np.expand_dims(P, axis=0)) / rescale + 0.5
        edge_attr_coarse = np.clip(edge_feat[edge_index_coarse], 0, 1)
        edge_attr_coarse = torch.tensor(edge_attr_coarse, device=data.edge_attr.device).to(torch.float32)
        # create hyperedge_index for coarsened graph
        edge_index_coarse = torch.tensor(np.array(edge_index_coarse), dtype=torch.long, device=data.edge_index.device)
        hyperedge_index_corse = Augmentation.get_hyperedge_index(data.num_nodes, edge_index_coarse)
        # create a new torch_geometric.data.Data instance for the coarsened graph
        # we assume that the node features and other attributes remain the same
        data_coarse = Data(x=data.x, edge_index=edge_index_coarse, edge_attr=edge_attr, 
                    hyperedge_index=hyperedge_index, dummy_label=torch.cat((data.dummy_label, data.dummy_label), dim=1), pos=data.pos)
        return data_coarse
    
    @staticmethod
    def mixup(data, rate):
        index = torch.randperm(data.num_nodes, device=data.dummy_label.device)
        assert 0 <= rate <= 1
        x_ = data.x * (1 - rate) + data.x[index] * rate
        index = torch.unsqueeze(index, 1)
        return Data(x=x_, edge_index=data.edge_index, edge_attr=data.edge_attr, hyperedge_index=data.hyperedge_index,
                    dummy_label=torch.cat((data.dummy_label, index), dim=1),
                    pos=data.pos,
                    )
    
    @staticmethod
    def knn(data, rescale, k):
        """ each node has a fixed number of neighbors controlled by the hyperparameter k
        """
        def _dot_product(z1: torch.Tensor, z2: torch.Tensor):
            # l2 normalization
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)
            return torch.mm(z1, z2.t())
        
        x = data.x
        sim = _dot_product(x, x)
        A = torch.zeros_like(sim, device=sim.device)
        for i in range(sim.shape[0]):
            sorted_tensor_indices = torch.argsort(sim[i], descending=True)
            ranked_tensor_indices = torch.zeros_like(sorted_tensor_indices)
            ranked_tensor_indices[sorted_tensor_indices] = torch.arange(len(sim[i]), device=ranked_tensor_indices.device)
            A[i] = ranked_tensor_indices
        A = torch.where(A <= k, 1, 0)
        A = torch.where(A + A.T >= 1, 1, 0).detach().cpu().numpy()
        np.fill_diagonal(A, 0)
        edge_index = np.nonzero(A)
        
        P = data.pos.detach().cpu().numpy()
        edge_feat = 0.5 * (np.expand_dims(P, axis=1) - np.expand_dims(P, axis=0)) / rescale + 0.5
        edge_attr = edge_feat[edge_index]
        edge_attr = np.clip(edge_attr, 0, 1)
        
        edge_attr = torch.tensor(edge_attr, device=data.edge_attr.device).to(torch.float32)
        edge_index = torch.tensor(np.array(edge_index), dtype=torch.long, device=data.edge_index.device)
        hyperedge_index = Augmentation.get_hyperedge_index(data.num_nodes, edge_index)
        
        return Data(x=data.x, edge_index=edge_index, edge_attr=edge_attr, 
                    hyperedge_index=hyperedge_index, dummy_label=torch.cat((data.dummy_label, data.dummy_label), dim=1), pos=data.pos)
        
    @staticmethod
    def knn_thold(data, rescale, k):
        """the number of the neighbors of each node is controlled by the predefined similarity threshold
        """
        def _dot_product(z1: torch.Tensor, z2: torch.Tensor):
            # l2 normalization
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)
            return torch.mm(z1, z2.t())
        
        x = data.x
        sim = _dot_product(x, x)
        A = torch.where(sim >= k, 1, 0)
        A = torch.where(A + A.T >= 1, 1, 0).detach().cpu().numpy()
        np.fill_diagonal(A, 0)
        edge_index = np.nonzero(A)
        
        P = data.pos.detach().cpu().numpy()
        edge_feat = 0.5 * (np.expand_dims(P, axis=1) - np.expand_dims(P, axis=0)) / rescale + 0.5
        edge_attr = edge_feat[edge_index]
        edge_attr = np.clip(edge_attr, 0, 1)
        
        edge_attr=torch.tensor(edge_attr, device=data.edge_attr.device).to(torch.float32)
        edge_index = torch.tensor(np.array(edge_index), dtype=torch.long, device=data.edge_index.device)
        hyperedge_index = Augmentation.get_hyperedge_index(data.num_nodes, edge_index)
        
        return Data(x=data.x, edge_index=edge_index, edge_attr=edge_attr, 
                    hyperedge_index=hyperedge_index, dummy_label=torch.cat((data.dummy_label, data.dummy_label), dim=1), pos=data.pos)
        
    @staticmethod
    def attribute_shuffling(data):
        """ shuffle the rows of the nodes features (and dummy labels)
        """
        if data.num_edges == 0:
            data.dummy_label = torch.cat((data.dummy_label, data.dummy_label), dim=1)
            return data
        
        index = list(range(data.num_nodes))
        random.shuffle(index)
        return Data(x=data.x[index,:], edge_index=data.edge_index, edge_attr=data.edge_attr, 
                    hyperedge_index=data.hyperedge_index, dummy_label=torch.cat((data.dummy_label, data.dummy_label), dim=1)[index, :], 
                    pos=data.pos[index, :])
    
    @staticmethod   
    def nodes_modification(data, p_nr, k, aggr, p):
        """ delete n nodes then insert same amount of dummy nodes into the graph
        insipired by 'Edge modification' from @Multi-Scale Contrastive Siamese Networks for Self-Supervised Graph Representation Learning
        """
        
        if data.num_nodes <= 4:
            data.dummy_label = torch.cat((data.dummy_label, data.dummy_label), dim=1)
            return data
        if data.num_edges <= 4:
            data.dummy_label = torch.cat((data.dummy_label, data.dummy_label), dim=1)
            return data
        
        # * random number of nodes to be replaced
        num_replace = min(np.random.randint(1, math.ceil(data.num_nodes * p_nr) + 1), data.num_nodes - 3)
        # * nodes deletion
        for _ in range(num_replace):
            data = Augmentation.node_deletion(data)
            # * ensure the number of nodes left is at least 3
            if data.num_nodes <= 3:
                break
            if data.num_edges <= 3:
                break
        
        # * nodes insertion
        for _ in range(num_replace):
            data = Augmentation.node_insertion(data, k, aggr, p)
        data.dummy_label = torch.cat((data.dummy_label, data.dummy_label), dim=1)
        return data

    @staticmethod
    def get_adjacency_matrix(num_nodes, edge_index):
        '''
        :param num_nodes: number of nodes
        :param edge_index: [2, num_edges]
        :return adjacency matrix: [num_nodes, num_nodes]
        '''
        adjacency_matrix = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
        adjacency_matrix[edge_index[0], edge_index[1]] = 1
        return adjacency_matrix
    
    @staticmethod
    def get_hyperedge_index(num_nodes, edge_index):
        '''get the hyperedge index of the graph
        :param num_nodes: number of nodes
        :param edge_index: [2, num_edges]
        :return hyperedge_index: [3, ]
        '''
        A = Augmentation.get_adjacency_matrix(num_nodes, edge_index).detach().cpu().numpy()
        o3_A = np.expand_dims(A, axis=0) * np.expand_dims(A, axis=1) * np.expand_dims(A, axis=2)
        hyperedge_index = np.nonzero(o3_A)
        return torch.tensor(np.array(hyperedge_index), dtype=torch.long, device=edge_index.device)
    
    # delete only one node
    @staticmethod
    def node_deletion(data):
        '''
        input a graph, randomly  select  an  node  and  remove  it from the graph; remove all edges connecting to this node, return the processed graph
        torch tensor:
        :param data.x: [num_nodes, num_node_features]
        :param data.edge_index: [2, num_edges]
        :param data.edge_attr: [num_edges, num_edge_features] for now we do not consider edge features
        :param data.y: [graph_label_dimension]
        :return: torch_geometric.data.Data
        '''
        if data.num_nodes <= 2:
            return data
        if data.num_edges <= 2:
            return data
        
        # index of the deleted node
        node_index = random.randint(0, data.num_nodes - 1)

        # tensor slicing
        # https://stackoverflow.com/questions/43989310/can-i-slice-tensors-with-logical-indexing-or-lists-of-indices
        node_indices = list(filter(lambda x: x!= node_index, list(range(data.x.shape[0]))))
        node_indices = torch.LongTensor(node_indices)
        node_features = data.x[node_indices,:]
        dummy_label = data.dummy_label[node_indices, :]
        pos = data.pos[node_indices, :]

        # find the loc of deleted node in edge_index
        loc = (data.edge_index==node_index).nonzero()
        col = loc[:,1]
        edge_indices = list(filter(lambda x: x not in col, list(range(data.edge_index.shape[1]))))
        edge_indices = torch.LongTensor(edge_indices)
        edge_index = data.edge_index.clone()[:, edge_indices]
        # an intermediate node is deleted, the index of nodes after it should -1
        edge_index = torch.where(edge_index > node_index, edge_index - 1, edge_index)
        edge_attr = data.edge_attr.clone()[edge_indices, :]
        
        # hyperedge_index
        hyperedge_index = Augmentation.get_hyperedge_index(data.num_nodes - 1, edge_index)
        
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, 
                    hyperedge_index=hyperedge_index, dummy_label=dummy_label,
                    pos=pos)

    # https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
    # controlled by hyperparameters p_nd
    @staticmethod
    def nodes_deletion(data, p_nd):
        # assert 0 <= p_nd <= 1
        
        if data.num_nodes <= 2:
            data.dummy_label = torch.cat((data.dummy_label, data.dummy_label), dim=1)
            return data
        if data.num_edges <= 2:
            data.dummy_label = torch.cat((data.dummy_label, data.dummy_label), dim=1)
            return data
        
        # * delete at most data.num_nodes - 2 nodes
        num_deletion = min(np.random.randint(1, math.ceil(data.num_nodes * p_nd) + 1), data.num_nodes - 2)
        # assert data.num_nodes - num_deletion >= 2
        for _ in range(num_deletion):
            data = Augmentation.node_deletion(data)
            # ensure the number of nodes left is at least 2
            if data.num_nodes <= 2:
                data.dummy_label = torch.cat((data.dummy_label, data.dummy_label), dim=1)
                return data
            if data.num_edges <= 2:
                data.dummy_label = torch.cat((data.dummy_label, data.dummy_label), dim=1)
                return data
        data.dummy_label = torch.cat((data.dummy_label, data.dummy_label), dim=1)
        return data

    # for dummy node insertion
    # check if x \in R^n \times 1 is one-hot encoding or not
    @staticmethod
    def check_one_hot(x):
        '''check whether x is one hot
        '''
        if torch.sum(x) != 1:
            return False
        # if torch.sum(torch.where(x==0, x, 1)) > 1:
        # RuntimeError: expected scalar type float but found long int
        if torch.sum(torch.where(x==0, x, torch.tensor(1, device=x.device, dtype=x.dtype))) > 1:
            return False
        return True

    # check where X is one-hot for all nodes' features
    # X \in R^n1 \times n2 are one-hot encodings or not
    @staticmethod
    def check_one_hot_encodings(X):
        # if torch.sum(torch.where(X!=1, X, 0)).item() != 0:
        if torch.sum(torch.where(X!=1, X, torch.tensor(0, device=x.device, dtype=X.dtype))) != 0:
            return False
        n1, n2 = X.shape
        if n1 > n2:
            n1, n2 = n2, n1
            X = X.T
        one_1 = torch.ones((n1, 1))
        one_2 = torch.ones((n2, 1))
        return torch.equal(torch.matmul(X, one_2), one_1) and torch.le(torch.matmul(X.T, one_1), one_2).all().item()

    # select the mode of the node features i.e., one-hot
    @staticmethod
    def get_one_hot(component_features):
        '''
        find mode and transform to one-hot
        :param component_features : [num_node_in_component, num_features] onehot
        :return: [1, num_features] onehot
        '''
        def _mode(data):
            counter = Counter(data)
            _, max_count = counter.most_common(1)[0]
            return [key for key, count in counter.items() if count == max_count]
        
        # index of the one in the one-hot encodings
        indices = torch.argmax(component_features, dim=1).detach().cpu().numpy().tolist()
        # randomly select the mode from the indices
        index = random.choice(_mode(indices))
        return torch.zeros(1, component_features.shape[1], device=component_features.device).scatter_(1, torch.LongTensor([[index]]), 1)

    # * updated node insertion
    @staticmethod
    def node_insertion(data, k, aggr, p):
        '''
        input a graph, randomly select two strongly-connected sub-graphs S1 and S2, add a dummy node n whose feature is the average of the S2, and add an edge between n and some node in S1, return the processed graph
        torch tensor:
        :param data.x: [num_nodes,num_node_features]
        :param data.edge_index: [2,num_edges]
        :param data.edge_attr: [num_edges, num_edge_features] for now we do not consider edge features
        :param data.y: [graph_label_dimension]
        :return: torch_geometric.data.Data
        '''
        if data.num_nodes <= 2:
            return data
        
        num_nodes = data.num_nodes
        G = to_networkx(data)
        
        # * instead of finding strongly connected components, find none-overlapped subgraph
        nodes_index = range(data.num_nodes)
        # * n = 0.5|V|
        s1 = random.sample(nodes_index, max(2, data.num_nodes//max(2, k)))
        s2 = list(set(nodes_index) - set(s1))
        
        # add all the nodes
        G.add_node(num_nodes)

        # create feature for the dummy node
        s1_features = data.x[s1,:]
        
        # ! does not check one hot
        # one_hot = Augmentation.check_one_hot(data.x[0])
        # if one_hot:
        #     dummy_node = Augmentation.get_one_hot(s2_features)
        # else:
        #     dummy_node = torch.mean(s2_features, dim=0).unsqueeze(dim=0)
        if aggr == 'mean':
            dummy_node = torch.mean(s1_features, dim=0).unsqueeze(dim=0)
        elif aggr == 'sum':
            dummy_node = torch.sum(s1_features, dim=0).unsqueeze(dim=0)
        # new nodes embeddings
        x = torch.cat((data.x, dummy_node))
        
        # * add edges btw the newly inserted dummy node and random sampled nodes from s2
        selected_nodes = random.sample(list(s2), max(1, int(len(s2)//p)))
        for node in selected_nodes:
            G.add_edge(node, num_nodes)
            G.add_edge(num_nodes, node)
            
        # label for the dummy node is -1
        inserted_labels = torch.zeros((1, data.dummy_label.size(1)), device=data.dummy_label.device) - 1
        dummy_label = torch.cat((data.dummy_label, inserted_labels), dim=0)
        
        # pos for dummy node
        inserted_pos = torch.mean(data.pos[s1,:], 0, True, dtype=data.pos.dtype)
        pos = torch.cat((data.pos, inserted_pos), dim=0)

        edge_index = torch.tensor(list(G.edges), device=data.edge_index.device).t().contiguous().long()
        hyperedge_index = Augmentation.get_hyperedge_index(num_nodes + 1, edge_index)
        return Data(x=x, edge_index=edge_index, edge_attr=data.edge_attr,
                    hyperedge_index=hyperedge_index, dummy_label=dummy_label,
                    pos=pos)

    # controlled by hyperparameters p_ni
    @staticmethod
    def nodes_insertion(data, p_ni, k, aggr, p):
        # assert 0 <= p_ni <= 1
        
        if data.num_nodes <= 2:
            data.dummy_label = torch.cat((data.dummy_label, data.dummy_label), dim=1)
            return data
    
        # * insert random number of nodes
        num_insertion = np.random.randint(1, math.ceil(data.num_nodes * p_ni) + 1)
        for _ in range(num_insertion):
            data = Augmentation.node_insertion(data, k, aggr, p)
        data.dummy_label = torch.cat((data.dummy_label, data.dummy_label), dim=1)
        return data

    # controlled by hyperparameters p_sub
    @staticmethod
    def sub_sampling(data, p_sub):
        # assert 0 <= p_sub[0] <= 1
        # assert 0 <= p_sub[1] <= 1
        # assert p_sub[0] <= p_sub[1]
        if data.num_nodes == 0:
            data.dummy_label = torch.cat((data.dummy_label, data.dummy_label), dim=1)
            return data
        
        # * keep random number of nodes
        nodes_kept = min(max(2, math.ceil(data.num_nodes * random.uniform(*p_sub))), data.num_nodes - 1)
        # assert 2 <= nodes_kept <= data.num_nodes - 1
        indices = random.sample(range(data.num_nodes), k=nodes_kept)
        # * specify the number of nodes in the original graph to avoid the error of index out of range
        edge_index = subgraph(indices, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)[0]
        edge_attr = data.edge_attr[edge_index]
        hyperedge_index = Augmentation.get_hyperedge_index(nodes_kept, edge_index)
        return Data(x=data.x[indices,:], edge_index=edge_index, edge_attr=data.edge_attr,
                    hyperedge_index = hyperedge_index, dummy_label=torch.cat((data.dummy_label, data.dummy_label), dim=1)[indices, :],
                    pos=data.pos[indices,:])

    # refer: @Reinforcement Learning with Augmented Data
    # logseq://graph/noteshelf?page=%40Reinforcement%20Learning%20with%20Augmented%20Data
    # apply the perturbation to all the nodes 

    # s' = s + z and z \sim \mathcal{N}(0, I)
    @staticmethod
    def gaussian_noise(data):
        if Augmentation.check_one_hot(data.x[0]):
            return data
        z = torch.empty((data.x.size(1),),
                dtype=torch.float32,
                device=data.x.device).normal_(0, 1)
        x = torch.add(data.x, z)
        return Data(x=x, edge_index=data.edge_index, edge_attr=data.edge_attr, 
                    hyperedge_index=data.hyperedge_index, dummy_label=torch.cat((data.dummy_label, data.dummy_label), dim=1),
                    pos=data.pos)

    # s' = s * z and z \sim \mathcal{U}(\alpha, \beta)
    # z could be either one dimensional or multi dimensional
    # \alpha \in {0.6, 0.8}
    # \beta \in {1.2, 1.4}
    @staticmethod
    def random_amplitude_scaling_multivariate_(data):
        if Augmentation.check_one_hot(data.x[0]):
            return data
        alpha = torch.empty((data.x.size(1),),
                dtype=torch.float32,
                device=data.x.device).uniform_(0.6, 0.8)
        beta = torch.empty((data.x.size(1),),
                dtype=torch.float32,
                device=data.x.device).uniform_(1.2, 1.4)
        z = Uniform(alpha, beta).sample()
        x = torch.mul(data.x, z)
        return Data(x=x, edge_index=data.edge_index, edge_attr=data.edge_attr, 
                    hyperedge_index=data.hyperedge_index, dummy_label=torch.cat((data.dummy_label, data.dummy_label), dim=1),
                    pos=data.pos)
        
    @staticmethod
    def random_amplitude_scaling_multivariate(data, alpha, beta):
        """ alpha \in [0, 0.8] and beta \in [1, 1.8]
        """
        if Augmentation.check_one_hot(data.x[0]):
            return data
        alpha = torch.empty((data.x.size(1),),
                dtype=torch.float32,
                device=data.x.device).uniform_(alpha, alpha + 0.2)
        beta = torch.empty((data.x.size(1),),
                dtype=torch.float32,
                device=data.x.device).uniform_(beta, beta + 0.2)
        z = Uniform(alpha, beta).sample()
        x = torch.mul(data.x, z)
        return Data(x=x, edge_index=data.edge_index, edge_attr=data.edge_attr, 
                    hyperedge_index=data.hyperedge_index, dummy_label=torch.cat((data.dummy_label, data.dummy_label), dim=1),
                    pos=data.pos)

    @staticmethod
    def random_amplitude_scaling_single_(data):
        if Augmentation.check_one_hot(data.x[0]):
            return data
        alpha = torch.empty((1,),
                dtype=torch.float32,
                device=data.x.device).uniform_(0.6, 0.8)
        beta = torch.empty((1,),
                dtype=torch.float32,
                device=data.x.device).uniform_(1.2, 1.4)
        z = Uniform(alpha, beta).sample()
        x = torch.mul(data.x, z)
        return Data(x=x, edge_index=data.edge_index, edge_attr=data.edge_attr, 
                hyperedge_index=data.hyperedge_index, dummy_label=torch.cat((data.dummy_label, data.dummy_label), dim=1),
                pos=data.pos)
    
    @staticmethod
    def random_amplitude_scaling_single(data, alpha, beta):
        if Augmentation.check_one_hot(data.x[0]):
            return data
        alpha = torch.empty((1,),
                dtype=torch.float32,
                device=data.x.device).uniform_(alpha, alpha + 0.2)
        beta = torch.empty((1,),
                dtype=torch.float32,
                device=data.x.device).uniform_(beta, beta + 0.2)
        z = Uniform(alpha, beta).sample()
        x = torch.mul(data.x, z)
        return Data(x=x, edge_index=data.edge_index, edge_attr=data.edge_attr, 
                hyperedge_index=data.hyperedge_index, dummy_label=torch.cat((data.dummy_label, data.dummy_label), dim=1),
                pos=data.pos)
        
    # refer: @Deep Graph Contrastive Representation Learning
    # logseq://graph/noteshelf?page=%40Deep%20Graph%20Contrastive%20Representation%20Learning
    # controlled by hyperparameters p_r
    # https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.dropout_adj
    # * keep the isolated nodes
    # ~~TODO~~: try out discarding the isolated nodes
    # torch_geometric.utils.isolated
    @staticmethod
    def edge_removing(data, p_er):
        # assert 0 <= p_er <= 1
        if data.num_edges == 0:
            data.dummy_label = torch.cat((data.dummy_label, data.dummy_label), dim=1)
            return data
        
        edge_index, _ = dropout_adj(data.edge_index, p=p_er)
        hyperedge_index = Augmentation.get_hyperedge_index(data.num_nodes, edge_index)
        return Data(x=data.x, edge_index=edge_index, edge_attr=data.edge_attr, 
                hyperedge_index=hyperedge_index, dummy_label=torch.cat((data.dummy_label, data.dummy_label), dim=1),
                pos=data.pos)

    # controlled by hyperparameters p_m
    @staticmethod
    def node_feature_masking(data, p_nm):
        # assert 0 <= p_nm <= 1
        
        if Augmentation.check_one_hot(data.x[0]):
            return data
        
        drop_mask = torch.empty((data.x.size(1),),
                                dtype=torch.float32,
                                device=data.x.device).uniform_(0, 1) < p_nm
        x = data.x.clone()
        x[:, drop_mask] = 0
        return Data(x=x, edge_index=data.edge_index, edge_attr=data.edge_attr, 
                hyperedge_index=data.hyperedge_index, dummy_label=torch.cat((data.dummy_label, data.dummy_label), dim=1),
                pos=data.pos)

    # randomly remove one edge
    @staticmethod
    def edge_deletion(data): 
        if data.num_edges==0:
            return data

        # random edge to be deleted
        index = random.randint(0, data.num_edges - 1)
        edge_index = data.edge_index.t().detach().cpu().numpy().tolist()

        u, v = edge_index[index]
        edge_index.remove([u, v])
        edge_index.remove([v, u])

        edge_index = torch.tensor(edge_index, device=data.edge_index.device).t().contiguous().long()
        hyperedge_index = Augmentation.get_hyperedge_index(data.num_nodes, edge_index)
        return Data(x=data.x, edge_index=edge_index, edge_attr=data.edge_attr, 
                    hyperedge_index=hyperedge_index, dummy_label=data.dummy_label,
                    pos=data.pos)

    @staticmethod
    def edges_deletion(data, p_ed):
        # assert 0 <= p_ed <= 1
        
        if data.num_edges==0:
            data.dummy_label = torch.cat((data.dummy_label, data.dummy_label), dim=1)
            return data
        
        # * delete at most data.num_edges - 2 edges
        num_deletion = min(np.random.randint(1, math.ceil(data.num_nodes * p_ed) + 1), data.num_nodes - 2)
        for _ in range(num_deletion):
            data = Augmentation.edge_deletion(data)
            if data.num_edges<=2:
                data.dummy_label = torch.cat((data.dummy_label, data.dummy_label), dim=1)
                return data
            data.dummy_label = torch.cat((data.dummy_label, data.dummy_label), dim=1)
        return data

    # randomly  select two nodes, if they are not directly connected but there is a path between them
    # then add an edge between these two nodes, return the processed graph
    @staticmethod
    def edge_insertion(data, rescale):
        G = to_networkx(data)
        A = nx.to_numpy_matrix(G)
        P = data.pos.detach().cpu().numpy()
        inserted = False
        # * reduce max_iter to speed up the augmentation
        max_iter = data.num_nodes
        
        while not inserted and max_iter >= 1:
            u, v = random.sample(range(data.num_nodes), k=2)
            edge_index_list = data.edge_index.t().detach().cpu().numpy().tolist()

            if nx.has_path(G, source=u, target=v) and nx.has_path(G, source=v, target=u):
                if not [u, v] in edge_index_list:
                    # * update edge_attr
                    # * refer to src.dataset.data_loader.to_pyg_graph
                    A[u, v], A[v, u] = 1, 1
                    edge_index = np.nonzero(A)
                    edge_feat = 0.5 * (np.expand_dims(P, axis=1) - np.expand_dims(P, axis=0)) / rescale + 0.5
                    edge_attr = edge_feat[edge_index]
                    edge_attr = np.clip(edge_attr, 0, 1)
                    edge_attr=torch.tensor(edge_attr, device=data.edge_attr.device).to(torch.float32)
                    edge_index = torch.tensor(np.array(edge_index), dtype=torch.long, device=data.edge_index.device)
                    hyperedge_index = Augmentation.get_hyperedge_index(data.num_nodes, edge_index)
                    inserted = True
                    return Data(x=data.x, edge_index=edge_index, edge_attr=edge_attr, 
                                hyperedge_index=hyperedge_index, dummy_label=data.dummy_label, pos=data.pos), inserted
                    
                    # break
            max_iter -= 1
        return Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, 
                    hyperedge_index=data.hyperedge_index, dummy_label=data.dummy_label, pos=data.pos), inserted

    @staticmethod
    def edges_insertion(data, rescale, p_ei):
        # assert 0 <= p_ei <= 1
        
        data.dummy_label = torch.cat((data.dummy_label, data.dummy_label), dim=1)
        
        if data.num_nodes <= 1:
            return data
        
        # * insert random number of edges
        num_insertion = np.random.randint(1, math.ceil(data.num_edges * p_ei) + 1)
        success = False
        for _ in range(num_insertion):
            # data, inserted = Augmentation.edge_insertion(data, P, rescale)
            data, inserted = Augmentation.edge_insertion(data, rescale)
            if not inserted:
                return data, success
            success = True
        return data, success

    @staticmethod
    def parse_para(params):
        """Parse the input string to a list of float, int, string, and tuple."""
        if not isinstance(params, str):
            return [params]

        # Define a regex pattern to match each input type
        patterns = [r'^\d+\.\d+$', r'^\d+$', r'^\w+$', r'\(([\w\d.]+),\s*([\w\d.]+)\)']

        # Split the input string using regex
        split_params = re.split(',\s*(?![^()]*\))', params)

        # Parse each input using regex and list comprehensions
        parsed_params = []
        for param in split_params:
            if re.match(patterns[0], param):
                parsed_params.append(float(param))
            elif re.match(patterns[1], param):
                parsed_params.append(int(param))
            elif re.match(patterns[2], param):
                parsed_params.append(param)
            else:
                tuple_match = re.match(patterns[3], param)
                if tuple_match:
                    values = tuple_match.groups()
                    parsed_values = []
                    for value in values:
                        if value.replace('.', '').isdigit():  # Check if the value is an int or float
                            if '.' in value:
                                parsed_values.append(float(value))
                            else:
                                parsed_values.append(int(value))
                        else:
                            parsed_values.append(value)
                    parsed_params.append(tuple(parsed_values))
        return parsed_params
    
    @staticmethod
    def random_aug_(params):
        """ generate random augmentation hyperparameters based on the input params
        ['random', n, (float, float), (int, int), (str, str)]
        represent a list of n random float, int, str in the range of float, int, str
        return n random augmentation hyperparameters
        """
        assert params[0] == 'random'
        
        random_params = []
        n = params[1]
        
        for i in range(n):
            para = []
            for p in params[2:]: # p is tuple of float, int, or str
                if isinstance(p[0], float):
                    para.append(round(np.random.uniform(p[0], p[1]), 2))
                elif isinstance(p[0], int):
                    para.append(np.random.randint(p[0], p[1]))
                else:
                    para.append(np.random.choice(p))
            if para not in random_params:
                random_params.append(para)
        return len(random_params), random_params

    @staticmethod
    def random_aug(params):
        """
        Generate random augmentation hyperparameters based on the input params.

        params: ['random', n, (float, float), (int, int), (str, str)] or ['random', n, (float, float), (int, int)] or ['random', n, (float, float), (str, str)]
        Represents a list of n random floats, ints, or strings within the specified ranges.

        Returns the number of random augmentation hyperparameters generated, the list of hyperparameters, and the count for each value.
        """
        assert params[0] == 'random'

        random_params = []
        n = params[1]
        param_ranges = params[2:]
        param_types = [type(param_range[0]) for param_range in param_ranges]

        for i, param_range in enumerate(param_ranges):
            param_type = param_types[i]
            
            if n == 1:
                if param_type == float:
                    samples = [round(random.uniform(*param_range), 2)]
                elif param_type == int:
                    samples = [random.randint(*param_range)]
                elif param_type == str:
                    samples = [random.choice(param_range)]
                else:
                    raise ValueError(f"Unsupported parameter type: {param_type}")
            else:
                if param_type == float:
                    start, end = param_range
                    step = (end - start) / n
                    samples = [round(start + step * j, 2) for j in range(n)]
                elif param_type == int:
                    start, end = param_range
                    values_per_range = n // (end - start + 1)
                    samples = [num for num in range(start, end + 1) for _ in range(values_per_range)]
                    samples += random.sample(samples, n % (end - start + 1))  # Handle the case when n is not divisible by (max - min + 1)
                    random.shuffle(samples)
                elif param_type == str:
                    str_range = param_range
                    total_strings = len(str_range)
                    values_per_string = n // total_strings
                    samples = [random.choice(str_range) for _ in range(values_per_string * total_strings)]
                    samples += random.sample(samples, n - len(samples))  # Handle the case when n is not divisible by total_strings
                    random.shuffle(samples)
                else:
                    raise ValueError(f"Unsupported parameter type: {param_type}")

            random_params.append(samples)

        random_params = list(zip(*random_params))
        random_params = [list(params) for params in random_params]
        random.shuffle(random_params)
        return len(random_params), random_params

    @staticmethod
    def reweight(weights, alpha=0.5):
        """ takes a list of weights and adjusts them using the log method to make the distribution less skewed while preserving the dominance of the largest weight
        $$ score_b=\alpha \cdot \frac{1}{\operatorname{var}\left\{w_1^{\prime}, w_2^{\prime}, \ldots, w_n^{\prime}\right\}}+ (1 - \alpha) \cdot \frac{\sum_{i=1}^n w_i^{\prime}}{\max _{i=1}^n w_i^{\prime}}$$
        $\alpha$ is the weight that determine the importance of each criterion
        """
        # compute logarithmic transformation of weights
        log_weights = np.log(weights + 1)
        # * coefficient of variation (CV) is the ratio of the standard deviation to the mean
        if np.std(log_weights) / np.mean(log_weights) <= 1.e-2:
            return weights
        
        # iterate over possible bases and compute weighted score for each base
        base_scores = {}
        for base in np.arange(2, 10):
            transformed_weights = np.log(weights + 1) / np.log(base)
            var = np.var(transformed_weights)
            score = alpha / var + (1 - alpha) * np.sum(transformed_weights) / np.max(transformed_weights)
            base_scores[base] = score
        
        # find base with maximum score
        max_base = max(base_scores, key=base_scores.get)
        # compute adjusted weights based on selected base
        adjusted_weights = np.log(weights + 1) / np.log(max_base)
        return adjusted_weights
    
    @staticmethod
    def update_moving_avg(avg_f1, count, new_f1, new_count):
        """ update moving average
        (count - 1) is the number of times the moving average has been updated
        """
        return (avg_f1 * count + new_f1) / new_count

    @staticmethod
    def update_counts(prev_counts, cur_indices):
        """ count the number of times each augmentation has been used in the current batch
        """
        unique_indices, counts = np.unique(cur_indices, return_counts=True)
        cur_counts = prev_counts.copy()
        for index, count in zip(unique_indices, counts):
            cur_counts[index] += count
        return cur_counts
            
    ### sample certain number of augmentations from the whole augmentation list
    def uniform_sample(self, num_augs):
        """ uniform sample num_augs augmentation from whole augmentation list
        """
        augs_with_params = list(zip(self.augs, self.params))
        random.shuffle(augs_with_params)
        sample = random.sample(augs_with_params, num_augs)
        sampled_augs, sampled_params = zip(*sample)
        self.augs, self.params = list(sampled_augs), list(sampled_params)

    def stratified_sample(self, num_augs):
        """ stratified sample num_augs augmentation from whole augmentation list
        """
        # create a dictionary where the keys are the category names
        # and the values are lists of augs in each category and corresponding params
        category_dict = defaultdict(list)
        for aug, param in zip(self.augs, self.params):
            category_dict[aug.__name__.upper()].append((aug, param))

        categories = list(category_dict.keys())
        samples_per_category = num_augs // len(categories)
        remainder = num_augs % len(categories)
        sample = []

        for category in categories:
            augs = category_dict[category]
            if len(augs) < samples_per_category:
                sample += augs
            else:
                sample += random.sample(augs, samples_per_category)
        
        # if num_augs is not a multiple of len(categories), sample additional items
        if remainder > 0:
            additional_categories = random.sample(categories, remainder)
            for category in additional_categories:
                augs = [aug for aug in category_dict[category] if aug not in sample]
                if augs:
                    sample += random.sample(augs, 1)
                    
        random.shuffle(sample)
        sampled_augs, sampled_params = zip(*sample)
        self.augs, self.params = list(sampled_augs), list(sampled_params)
        self.categories = categories.copy()
        
    ### create augmentation pairs
    def create_pairs(self):
        augs_with_params = list(zip(self.augs, self.indices, self.params))
        
        # create all possible pairs of augs including self-pairs
        pairs = list(combinations_with_replacement(augs_with_params, 2))
        
        # order the pair with Noop as the second element if it exists
        # and order the pair with Mixup as the first element if it exists
        if 'MIXUP' in self.categories or 'NOOP' in self.categories:
            for i, p in enumerate(pairs):
                if p[0][0].__name__.upper() == 'NOOP':
                    pairs[i] = (p[1], p[0])
                if p[1][0].__name__.upper() == 'MIXUP':
                    pairs[i] = (p[1], p[0])
        
        # remove pairs with (MIXUP, MIXUP) and (NOOP, NOOP)
        pairs = list(filter(lambda x: x[1][0].__name__.upper() != 'MIXUP' and x[0][0].__name__.upper() != 'NOOP', pairs))
        return pairs
    
    ### sample certain number of pairs from the whole pair list
    def uniform_pairs(self, num_pairs):
        """ uniformly sample num_pairs pairs from whole pair list
        """
        pairs = self.create_pairs()
        random.shuffle(pairs)
        sampled_pairs = list(random.sample(pairs, num_pairs))
        return sampled_pairs
        
    def stratified_pairs(self, num_pairs):
        """ stratified sample num_pairs pairs based on type of augmentation from whole pair list
        """
        # create a dictionary where the keys are the (aug name, aug_name) tuple
        # and the values are lists of aug pair in each category
        pairs = self.create_pairs()
        if num_pairs < len(pairs):
            category_dict = defaultdict(list)
            for pair in pairs:
                category_dict[tuple(sorted(pair[i][0].__name__.upper() for i in range(len(pair))))].append(pair)

            categories = list(category_dict.keys())
            samples_per_category = num_pairs // len(categories)
            remainder = num_pairs % len(categories)
            sampled_pairs = []

            for category in categories:
                pairs = category_dict[category]
                if len(pairs) < samples_per_category:
                    sampled_pairs += pairs
                else:
                    sampled_pairs += random.sample(pairs, samples_per_category)
                        
            # if num_pairs is not a multiple of len(categories), sample additional items
            if remainder > 0:
                # Shuffle categories for random selection
                random.shuffle(categories)
                
                # Create a dictionary to hold unsampled pairs for each category
                unsampled_dict = {}
                for category in categories:
                    unsampled_dict[category] = [pair for pair in category_dict[category] if pair not in sampled_pairs]
                
                i = 0  # Initialize category index
                while remainder > 0 and any(unsampled_dict.values()):  # While there are still pairs to add and unsampled pairs exist
                    category = categories[i % len(categories)]  # Select a category
                    
                    # If unsampled pairs exist for the category, pop and add a pair to sampled_pairs
                    if unsampled_dict[category]:
                        pair = random.choice(unsampled_dict[category])
                        unsampled_dict[category].remove(pair)
                        sampled_pairs.append(pair)
                        remainder -= 1
                    i += 1  # Proceed to next category
                        
            random.shuffle(sampled_pairs)
            return sampled_pairs
        return pairs