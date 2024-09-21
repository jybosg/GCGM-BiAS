import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import math

from src.lap_solvers.sinkhorn import Sinkhorn, GumbelSinkhorn
from src.build_graphs import reshape_edge_feature
from src.feature_align import feature_align
from src.factorize_graph_matching import construct_aff_mat
from models.NGM.gnn import GNNLayer
from models.NGM.geo_edge_feature import geo_edge_feature
from models.GMN.affinity_layer import InnerpAffinity, GaussianAffinity
from src.evaluation_metric import objective_score
from src.lap_solvers.hungarian import hungarian
from src.utils.gpu_memory import gpu_free_memory
# * GCL encoder and projection head
from models.GCGM.encoder import Encoder
# * boosting
from models.GCGM.augmentation import Augmentation
# * data loader
from torch_geometric.loader import DataLoader
from src.utils.data_to_cuda import data_to_cuda
# * updated affinity learning
from models.GCGM.affinity_layer import InnerProductWithWeightsAffinity, GaussianWithWeightAffinity
from src.utils.pad_tensor import pad_tensor
# * Kronecker product
from src.factorize_graph_matching import kronecker_sparse, kronecker_torch
from src.sparse_torch import CSRMatrix3d
import numpy as np
# * traditional GM solver
from lpmp_py import GraphMatchingModule
from lpmp_py import MultiGraphMatchingModule
# * SplineConv
from models.BBGM.sconv_archs import SiameseSConvOnNodes

from src.utils.config import cfg

from src.backbone import *
CNN = eval(cfg.BACKBONE)

from src.utils.model_sl import load_model

class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        # * affinity layer
        self.global_state_dim = 2 * cfg.GCGM.OUT_CHANNELS # concatenation of global representation of two graphs
        # * vertex affinity
        self.vertex_affinity = InnerProductWithWeightsAffinity(self.global_state_dim, 
                                                               cfg.GCGM.OUT_CHANNELS)
        # * edge affinity
        if cfg.GCGM.EDGE_FEATURE == 'cat': # edge feature is the concatenation of the two ending nodes
            self.edge_affinity = InnerProductWithWeightsAffinity(self.global_state_dim,
                                                                 2 * cfg.GCGM.OUT_CHANNELS)
        elif cfg.GCGM.EDGE_FEATURE == 'geo':
            self.edge_affinity = GaussianWithWeightAffinity(self.global_state_dim,
                                                            1,
                                                            cfg.GCGM.GAUSSIAN_SIGMA)
        else:
            raise ValueError('Unknown edge feature type {}'.format(cfg.GCGM.EDGE_FEATURE))
        
        if cfg.GCGM.get('SOLVER_NAME', '') != "LPMP":
            # * sinkhorn normalization
            self.tau = cfg.GCGM.SK_TAU
            self.sinkhorn = Sinkhorn(max_iter=cfg.GCGM.SK_ITER_NUM, tau=self.tau, epsilon=cfg.GCGM.SK_EPSILON)
            self.gumbel_sinkhorn = GumbelSinkhorn(max_iter=cfg.GCGM.SK_ITER_NUM, tau=self.tau * 10, epsilon=cfg.GCGM.SK_EPSILON, batched_operation=True)
            
            # affinity learning
            self.gnn_layer = cfg.GCGM.GNN_LAYER
            for i in range(self.gnn_layer):
                tau = cfg.GCGM.SK_TAU
                if i == 0:
                    gnn_layer = GNNLayer(1, 1, cfg.GCGM.GNN_FEAT[i] + (1 if cfg.GCGM.SK_EMB else 0), cfg.GCGM.GNN_FEAT[i],
                                        sk_channel=cfg.GCGM.SK_EMB, sk_tau=tau, edge_emb=cfg.GCGM.EDGE_EMB)
                else:
                    gnn_layer = GNNLayer(cfg.GCGM.GNN_FEAT[i - 1] + (1 if cfg.GCGM.SK_EMB else 0), cfg.GCGM.GNN_FEAT[i - 1],
                                        cfg.GCGM.GNN_FEAT[i] + (1 if cfg.GCGM.SK_EMB else 0), cfg.GCGM.GNN_FEAT[i],
                                        sk_channel=cfg.GCGM.SK_EMB, sk_tau=tau, edge_emb=cfg.GCGM.EDGE_EMB)
                self.add_module('gnn_layer_{}'.format(i), gnn_layer)

            self.classifier = nn.Linear(cfg.GCGM.GNN_FEAT[-1] + (1 if cfg.GCGM.SK_EMB else 0), 1)
        
        self.rescale = cfg.PROBLEM.RESCALE
        # feature normalization
        self.l2norm = nn.LocalResponseNorm(cfg.GCGM.IN_CHANNELS, alpha=cfg.GCGM.IN_CHANNELS, beta=0.5, k=0)
        
        # * SplineConv
        self.message_pass_node_features = SiameseSConvOnNodes(input_node_dim=cfg.GCGM.IN_CHANNELS)
        self.sconv_params = list(self.message_pass_node_features.parameters())
        
        # * GCN encoder
        # * GCGM.IN_CHANNELS == 2 * GCGM.FEATURE_CHANNEL for visual datasets
        self.encoder = Encoder(cfg.GCGM.IN_CHANNELS, cfg.GCGM.INTER_CHANNELS, cfg.GCGM.OUT_CHANNELS, cfg.GCGM.ACTIVATION,
                               base_model=cfg.GCGM.BASE_MODEL, aggr=cfg.GCGM.AGGR, dropout=cfg.GCGM.DROPOUT,
                               skip=cfg.GCGM.SKIP, project=cfg.GCGM.PROJECT,
                               k=cfg.GCGM.K, global_readout=cfg.GCGM.GLOBAL_READOUT,
                               structure=cfg.GCGM.ENCODER_STRUCTURE)
        
        self.augmentation = Augmentation(cfg)
                    
        # * prepare for the two-stage training
        self.encoder_params = list(self.encoder.parameters())
        
    def forward(self, data_dict, **kwargs):
        # * data_dict['pyg_graphs'] is a list of two batched graph
        # * data_dict['pyg_graphs'][0] is 'torch_geometric.data.batch.DataBatch'
        # refer to https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        batch_size = data_dict['batch_size'] // len(cfg.GPUS)
        if 'images' in data_dict:
            # real image data
            src, tgt = data_dict['images']
            P_src, P_tgt = data_dict['Ps']
            ns_src, ns_tgt = data_dict['ns']
            G_src, G_tgt = data_dict['Gs']
            H_src, H_tgt = data_dict['Hs']
            K_G, K_H = data_dict['KGHs'] 

            # extract feature
            # * src_node.shape = b \times GCGM.FEATURE_CHANNEL \times H \times W
            src_node = self.node_layers(src)
            src_edge = self.edge_layers(src_node)
            tgt_node = self.node_layers(tgt)
            tgt_edge = self.edge_layers(tgt_node)

            # feature normalization
            src_node = self.l2norm(src_node)
            src_edge = self.l2norm(src_edge)
            tgt_node = self.l2norm(tgt_node)
            tgt_edge = self.l2norm(tgt_edge)
            
            # arrange features
            # * U_src.shape = b \times GCGM.FEATURE_CHANNEL \times n
            # * n = max(ns_src)
            U_src = feature_align(src_node, P_src, ns_src, self.rescale)
            F_src = feature_align(src_edge, P_src, ns_src, self.rescale)
            U_tgt = feature_align(tgt_node, P_tgt, ns_tgt, self.rescale)
            F_tgt = feature_align(tgt_edge, P_tgt, ns_tgt, self.rescale)
            
            # * save features into the data_dict
            data_dict.update({'U_src': U_src,
                              'F_src': F_src,
                              'U_tgt': U_tgt,
                              'F_tgt': F_tgt
                              })
            
        elif 'features' in data_dict:
            # synthetic data
            src, tgt = data_dict['features']
            # * P_src.shape = b \times n \times 2
            P_src, P_tgt = data_dict['Ps']
            ns_src, ns_tgt = data_dict['ns']
            G_src, G_tgt = data_dict['Gs']
            H_src, H_tgt = data_dict['Hs']
            K_G, K_H = data_dict['KGHs']
            
            # * normalize over channels
            src, tgt = Net.normalize_over_channels(src), Net.normalize_over_channels(tgt)

            # * U_src.shape = b \times GCGM.FEATURE_CHANNEL \times n
            U_src = src[:, :src.shape[1] // 2, :]
            F_src = src[:, src.shape[1] // 2:, :]
            U_tgt = tgt[:, :tgt.shape[1] // 2, :]
            F_tgt = tgt[:, tgt.shape[1] // 2:, :]
            
            # save features into the data_dict
            data_dict.update({'U_src': U_src,
                              'F_src': F_src,
                              'U_tgt': U_tgt,
                              'F_tgt': F_tgt
                              })
            
        elif 'aff_mat' in data_dict:
            K = data_dict['aff_mat']
            ns_src, ns_tgt = data_dict['ns']
        else:
            raise ValueError('Unknown data type for this model.')
        
        # * update source graphs and target graphs with the features extracted from the backbone network
        data_dict = self.update_graphs(data_dict)
        
        # * pretrain encoder
        if cfg.PROBLEM.TYPE == 'GCL':
            if self.training:
                data_dict = self.encode_corrupted(data_dict)
                data_dict = self.get_perm_mat(data_dict)
                data_dict = self.get_connectivity_matrices(data_dict)
                data_dict = self.get_kronecker_product(data_dict)
            else:
                data_dict = self.encode(data_dict)
            
        # * two-stage training with encoder.parameters() loaded
        elif cfg.PROBLEM.TYPE == 'GCLTS':
            data_dict = self.encode(data_dict)
        
        # * train the whole model end-to-end with both contrastive loss and permutation loss
        elif cfg.PROBLEM.TYPE == 'GCLE2E':
            data_dict = self.encode(data_dict)
            data_dict = self.encode_corrupted(data_dict)
        
        if cfg.PROBLEM.TYPE != 'GCL' or not self.training:
            if 'images' in data_dict:
                tgt_len = P_tgt.shape[1]
                if cfg.GCGM.EDGE_FEATURE == 'cat':
                    X = reshape_edge_feature(data_dict['UF_src_updt'], G_src, H_src)
                    Y = reshape_edge_feature(data_dict['UF_tgt_updt'], G_tgt, H_tgt)
                elif cfg.GCGM.EDGE_FEATURE == 'geo':
                    X = geo_edge_feature(P_src, G_src, H_src)[:, :1, :]
                    Y = geo_edge_feature(P_tgt, G_tgt, H_tgt)[:, :1, :]
                else:
                    raise ValueError('Unknown edge feature type {}'.format(cfg.GCGM.EDGE_FEATURE))

                ns_src_cum = torch.cat((torch.zeros(1, dtype=ns_src.dtype, device=ns_src.device), torch.cumsum(ns_src, dim=0)), dim=0)
                ns_tgt_cum = torch.cat((torch.zeros(1, dtype=ns_tgt.dtype, device=ns_tgt.device), torch.cumsum(ns_tgt, dim=0)), dim=0)
                # * unary_affs is a list of vertex affinity matrix [(ns_src[0], ns_tgt[0]), ...]
                unary_affs = self.vertex_affinity([data_dict['UF_src_enc'][ns_src_cum[b - 1].item(): ns_src_cum[b].item(),:] for b in range(1, ns_src_cum.size(0))], 
                                                  [data_dict['UF_tgt_enc'][ns_tgt_cum[b - 1].item(): ns_tgt_cum[b].item(),:] for b in range(1, ns_tgt_cum.size(0))], 
                                                  [data_dict['Glob'][b] for b in range(batch_size)],
                                                  use_global=cfg.GCGM.USE_GLOBAL
                                                  )
                
                quadratic_affs = self.edge_affinity([X[b, :, :].t() for b in range(batch_size)], 
                                                    [Y[b, :, :].t() for b in range(batch_size)], 
                                                    [data_dict['Glob'][b] for b in range(batch_size)],
                                                    use_global=cfg.GCGM.USE_GLOBAL
                                                    )
                
                # * NGM-v2
                quadratic_affs = [0.5 * x for x in quadratic_affs]

                Kp = torch.stack(pad_tensor(unary_affs), dim=0)
                Ke = torch.stack(pad_tensor(quadratic_affs), dim=0)
                K = construct_aff_mat(Ke, Kp, K_G, K_H)

                # A = (K > 0).to(K.dtype)
                if cfg.GCGM.POSITIVE_EDGES:
                    A = (K > 0).to(K.dtype)
                else:
                    A = (K != 0).to(K.dtype)

                if cfg.GCGM.FIRST_ORDER:
                    emb = Kp.transpose(1, 2).contiguous().view(Kp.shape[0], -1, 1)
                else:
                    emb = torch.ones(K.shape[0], K.shape[1], 1, device=K.device)
            
            elif 'features' in data_dict:
                tgt_len = P_tgt.shape[1]
                if cfg.GCGM.EDGE_FEATURE == 'cat':
                    X = reshape_edge_feature(data_dict['UF_src_updt'], G_src, H_src)
                    Y = reshape_edge_feature(data_dict['UF_tgt_updt'], G_tgt, H_tgt)
                elif cfg.GCGM.EDGE_FEATURE == 'geo':
                    X = geo_edge_feature(P_src, G_src, H_src)[:, :1, :]
                    Y = geo_edge_feature(P_tgt, G_tgt, H_tgt)[:, :1, :]
                else:
                    raise ValueError('Unknown edge feature type {}'.format(cfg.GCGM.EDGE_FEATURE))

                # * affinity layer
                ns_src_cum = torch.cat((torch.zeros(1, dtype=ns_src.dtype, device=ns_src.device), torch.cumsum(ns_src, dim=0)), dim=0)
                ns_tgt_cum = torch.cat((torch.zeros(1, dtype=ns_tgt.dtype, device=ns_tgt.device), torch.cumsum(ns_tgt, dim=0)), dim=0)
                # * unary_affs is a list of vertex affinity matrix [(ns_src[0], ns_tgt[0]), ...]
                unary_affs = self.vertex_affinity([data_dict['UF_src_enc'][ns_src_cum[b - 1].item(): ns_src_cum[b].item(),:] for b in range(1, ns_src_cum.size(0))], 
                                                  [data_dict['UF_tgt_enc'][ns_tgt_cum[b - 1].item(): ns_tgt_cum[b].item(),:] for b in range(1, ns_tgt_cum.size(0))], 
                                                  [data_dict['Glob'][b] for b in range(batch_size)],
                                                  use_global=cfg.GCGM.USE_GLOBAL)
                
                quadratic_affs = self.edge_affinity([X[b, :, :].t() for b in range(batch_size)], 
                                                    [Y[b, :, :].t() for b in range(batch_size)],
                                                    [data_dict['Glob'][b] for b in range(batch_size)],
                                                    use_global=cfg.GCGM.USE_GLOBAL)
                
                # * NGM-v2
                quadratic_affs = [0.5 * x for x in quadratic_affs]
                
                Kp = torch.stack(pad_tensor(unary_affs), dim=0)
                Ke = torch.stack(pad_tensor(quadratic_affs), dim=0)

                K = construct_aff_mat(Ke, Kp, K_G, K_H)

                # A = (K > 0).to(K.dtype)
                if cfg.GCGM.POSITIVE_EDGES:
                    A = (K > 0).to(K.dtype)
                else:
                    A = (K != 0).to(K.dtype)

                if cfg.GCGM.FIRST_ORDER:
                    emb = Kp.transpose(1, 2).contiguous().view(Kp.shape[0], -1, 1)
                else:
                    emb = torch.ones(K.shape[0], K.shape[1], 1, device=K.device)
            else:
                tgt_len = int(math.sqrt(K.shape[2]))
                dmax = (torch.max(torch.sum(K, dim=2, keepdim=True), dim=1, keepdim=True).values + 1e-5)
                K = K / dmax * 1000
                # A = (K > 0).to(K.dtype)
                if cfg.GCGM.POSITIVE_EDGES:
                    A = (K > 0).to(K.dtype)
                else:
                    A = (K != 0).to(K.dtype)
                emb = torch.ones(K.shape[0], K.shape[1], 1, device=K.device)

            emb_K = K.unsqueeze(-1)
            
            # * traditional solver
            if cfg.GCGM.get('SOLVER_NAME', '') == "LPMP":
                gm_solver = GraphMatchingModule(
                        [data_dict['src_graphs'][b].edge_index for b in range(batch_size)],
                        [data_dict['tgt_graphs'][b].edge_index for b in range(batch_size)],
                        ns_src,
                        ns_tgt,
                        cfg.GCGM.LAMBDA_VAL,
                        cfg.GCGM.SOLVER_PARAMS,
                    )

                # * normalize the affinity matrices
                unary_affs_normalized = []
                for tensor in unary_affs:
                    range_val = tensor.max() - tensor.min()
                    if range_val != 0:
                        normalized_tensor = (tensor - tensor.min()) / range_val
                    else:
                        normalized_tensor = tensor - tensor.min()
                    unary_affs_normalized.append(normalized_tensor)
                
                quadratic_affs_normalized = []
                for tensor in quadratic_affs:
                    range_val = tensor.max() - tensor.min()
                    if range_val != 0:
                        normalized_tensor = (tensor - tensor.min()) / range_val
                    else:
                        normalized_tensor = tensor - tensor.min()
                    quadratic_affs_normalized.append(normalized_tensor)
                    
                if self.training:
                    unary_affs_list = [-unary_affs_normalized[b] + 0.5 * data_dict['new_perm_mat'][b, :data_dict['ns_1'][b], :data_dict['ns_2'][b]] for b in range(batch_size)]
                else:
                    unary_affs_list = [-unary_affs_normalized[b] for b in range(batch_size)]
                quadratic_affs_list = [-0.1 * quadratic_affs_normalized[b][:data_dict['src_graphs'][b].num_edges, :data_dict['tgt_graphs'][b].num_edges] for b in range(batch_size)]
                
                matching = gm_solver(unary_affs_list, quadratic_affs_list)
                data_dict.update({'ds_mat': None,
                                  'perm_mat': matching,
                                  'aff_mat': K
                                  })
                
            else:
                # * NGM qap solver
                for i in range(self.gnn_layer):
                    gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                    emb_K, emb = gnn_layer(A, emb_K, emb, ns_src, ns_tgt) #, norm=False)

                v = self.classifier(emb)
                s = v.view(v.shape[0], tgt_len, -1).transpose(1, 2)

                if self.training or cfg.GCGM.GUMBEL_SK <= 0:
                # if cfg.GCGM.GUMBEL_SK <= 0:
                    ss = self.sinkhorn(s, ns_src, ns_tgt, dummy_row=True)
                    x = hungarian(ss, ns_src, ns_tgt)
                else:
                    gumbel_sample_num = cfg.GCGM.GUMBEL_SK
                    if self.training:
                        gumbel_sample_num //= 10
                    ss_gumbel = self.gumbel_sinkhorn(s, ns_src, ns_tgt, sample_num=gumbel_sample_num, dummy_row=True)

                    repeat = lambda x, rep_num=gumbel_sample_num: torch.repeat_interleave(x, rep_num, dim=0)
                    if not self.training:
                        ss_gumbel = hungarian(ss_gumbel, repeat(ns_src), repeat(ns_tgt))
                    ss_gumbel = ss_gumbel.reshape(batch_size, gumbel_sample_num, ss_gumbel.shape[-2], ss_gumbel.shape[-1])

                    if ss_gumbel.device.type == 'cuda':
                        dev_idx = ss_gumbel.device.index
                        free_mem = gpu_free_memory(dev_idx) - 100 * 1024 ** 2 # 100MB as buffer for other computations
                        K_mem_size = K.element_size() * K.nelement()
                        max_repeats = free_mem // K_mem_size
                        if max_repeats <= 0:
                            print('Warning: GPU may not have enough memory')
                            max_repeats = 1
                    else:
                        max_repeats = gumbel_sample_num

                    obj_score = []
                    for idx in range(0, gumbel_sample_num, max_repeats):
                        if idx + max_repeats > gumbel_sample_num:
                            rep_num = gumbel_sample_num - idx
                        else:
                            rep_num = max_repeats
                        obj_score.append(
                            objective_score(
                                ss_gumbel[:, idx:(idx+rep_num), :, :].reshape(-1, ss_gumbel.shape[-2], ss_gumbel.shape[-1]),
                                repeat(K, rep_num)
                            ).reshape(batch_size, -1)
                        )
                    obj_score = torch.cat(obj_score, dim=1)
                    min_obj_score = obj_score.min(dim=1)
                    ss = ss_gumbel[torch.arange(batch_size), min_obj_score.indices.cpu(), :, :]
                    x = hungarian(ss, repeat(ns_src), repeat(ns_tgt))
                    
                data_dict.update({'ds_mat': ss,
                                    'perm_mat': x,
                                    'aff_mat': K
                                    })
        
        # * GCL with graph matching loss
        else:
            batch_size = data_dict['batch_size'] // len(cfg.GPUS)
            if 'images' in data_dict:
                tgt_len = torch.max(data_dict['ns_2'])
                if cfg.GCGM.EDGE_FEATURE == 'cat':
                    X = reshape_edge_feature(data_dict['UF_1_updt'], data_dict['G_1'], data_dict['H_1'])
                    Y = reshape_edge_feature(data_dict['UF_2_updt'], data_dict['G_2'], data_dict['H_2'])
                elif cfg.GCGM.EDGE_FEATURE == 'geo':
                    X = geo_edge_feature(data_dict['Pos_1'], data_dict['G_1'], data_dict['H_1'])[:, :1, :]
                    Y = geo_edge_feature(data_dict['Pos_2'], data_dict['G_2'], data_dict['H_2'])[:, :1, :]
                
                ns_1_cum = torch.cat((torch.zeros(1, dtype=data_dict['ns_1'].dtype, device=data_dict['ns_1'].device), torch.cumsum(data_dict['ns_1'], dim=0)), dim=0)
                ns_2_cum = torch.cat((torch.zeros(1, dtype=data_dict['ns_2'].dtype, device=data_dict['ns_2'].device), torch.cumsum(data_dict['ns_2'], dim=0)), dim=0)
                # * unary_affs is a list of vertex affinity matrix [(ns_src[0], ns_tgt[0]), ...]
                unary_affs = self.vertex_affinity([data_dict['UF_1_enc'][ns_1_cum[b - 1].item(): ns_1_cum[b].item(),:] for b in range(1, ns_1_cum.size(0))], 
                                                  [data_dict['UF_2_enc'][ns_2_cum[b - 1].item(): ns_2_cum[b].item(),:] for b in range(1, ns_2_cum.size(0))], 
                                                  [data_dict['Glob'][b] for b in range(batch_size)],
                                                  use_global=cfg.GCGM.USE_GLOBAL)
                
                quadratic_affs = self.edge_affinity([X[b, :, :].t() for b in range(batch_size)], 
                                                    [Y[b, :, :].t() for b in range(batch_size)], 
                                                    [data_dict['Glob'][b] for b in range(batch_size)],
                                                    use_global=cfg.GCGM.USE_GLOBAL)
                
                # * NGM-v2
                quadratic_affs = [0.5 * x for x in quadratic_affs]
                
                Kp = torch.stack(pad_tensor(unary_affs), dim=0)
                Ke = torch.stack(pad_tensor(quadratic_affs), dim=0)

                # * Kronecker product
                K = construct_aff_mat(Ke, Kp, data_dict['K1G'],  data_dict['K1H'])

                if cfg.GCGM.POSITIVE_EDGES:
                    A = (K > 0).to(K.dtype)
                else:
                    A = (K != 0).to(K.dtype)

                if cfg.GCGM.FIRST_ORDER:
                    emb = Kp.transpose(1, 2).contiguous().view(Kp.shape[0], -1, 1)
                else:
                    emb = torch.ones(K.shape[0], K.shape[1], 1, device=K.device)
            
            elif 'features' in data_dict:
                tgt_len = torch.max(data_dict['ns_2'])
                if cfg.GCGM.EDGE_FEATURE == 'cat':               
                    X = reshape_edge_feature(data_dict['UF_1_updt'], data_dict['G_1'], data_dict['H_1'])
                    Y = reshape_edge_feature(data_dict['UF_2_updt'], data_dict['G_2'], data_dict['H_2'])
                # * update P_src and P_tgt according to the augmentations
                elif cfg.GCGM.EDGE_FEATURE == 'geo':
                    X = geo_edge_feature(data_dict['Pos_1'], data_dict['G_1'], data_dict['H_1'])[:, :1, :]
                    Y = geo_edge_feature(data_dict['Pos_2'], data_dict['G_2'], data_dict['H_2'])[:, :1, :]
                
                ns_1_cum = torch.cat((torch.zeros(1, dtype=data_dict['ns_1'].dtype, device=data_dict['ns_1'].device), torch.cumsum(data_dict['ns_1'], dim=0)), dim=0)
                ns_2_cum = torch.cat((torch.zeros(1, dtype=data_dict['ns_2'].dtype, device=data_dict['ns_2'].device), torch.cumsum(data_dict['ns_2'], dim=0)), dim=0)
                # * unary_affs is a list of vertex affinity matrix [(ns_src[0], ns_tgt[0]), ...]
                unary_affs = self.vertex_affinity([data_dict['UF_1_enc'][ns_1_cum[b - 1].item(): ns_1_cum[b].item(),:] for b in range(1, ns_1_cum.size(0))], 
                                                  [data_dict['UF_2_enc'][ns_2_cum[b - 1].item(): ns_2_cum[b].item(),:] for b in range(1, ns_2_cum.size(0))], 
                                                  [data_dict['Glob'][b] for b in range(batch_size)],
                                                  use_global=cfg.GCGM.USE_GLOBAL)
                
                quadratic_affs = self.edge_affinity([X[b, :, :].t() for b in range(batch_size)], 
                                                    [Y[b, :, :].t() for b in range(batch_size)], 
                                                    [data_dict['Glob'][b] for b in range(batch_size)],
                                                    # * for synthetic data, use_gloabl=False else it results in NaN
                                                    use_global=cfg.GCGM.USE_GLOBAL)
                
                # * NGM-v2
                quadratic_affs = [0.5 * x for x in quadratic_affs]
                
                Kp = torch.stack(pad_tensor(unary_affs), dim=0)
                Ke = torch.stack(pad_tensor(quadratic_affs), dim=0)
                K = construct_aff_mat(Ke, Kp, data_dict['K1G'],  data_dict['K1H'])

                if cfg.GCGM.POSITIVE_EDGES:
                    A = (K > 0).to(K.dtype)
                else:
                    A = (K != 0).to(K.dtype)

                if cfg.GCGM.FIRST_ORDER:
                    emb = Kp.transpose(1, 2).contiguous().view(Kp.shape[0], -1, 1)
                else:
                    emb = torch.ones(K.shape[0], K.shape[1], 1, device=K.device)
            else:
                tgt_len = int(math.sqrt(K.shape[2]))
                dmax = (torch.max(torch.sum(K, dim=2, keepdim=True), dim=1, keepdim=True).values + 1e-5)
                K = K / dmax * 1000
                if cfg.GCGM.POSITIVE_EDGES:
                    A = (K > 0).to(K.dtype)
                else:
                    A = (K != 0).to(K.dtype)
                emb = torch.ones(K.shape[0], K.shape[1], 1, device=K.device)

            emb_K = K.unsqueeze(-1)
            
             # * traditional solver
            if cfg.GCGM.get('SOLVER_NAME', '') == "LPMP":
                gm_solver = GraphMatchingModule(
                        [data_dict['views_1'][b].edge_index for b in range(batch_size)],
                        [data_dict['views_2'][b].edge_index for b in range(batch_size)],
                        data_dict['ns_1'],
                        data_dict['ns_2'],
                        cfg.GCGM.LAMBDA_VAL,
                        cfg.GCGM.SOLVER_PARAMS,
                    )
                
                # * normalize the affinity matrices
                unary_affs_normalized = []
                for tensor in unary_affs:
                    range_val = tensor.max() - tensor.min()
                    if range_val != 0:
                        normalized_tensor = (tensor - tensor.min()) / range_val
                    else:
                        normalized_tensor = tensor - tensor.min()  # Or some other fallback mechanism
                    unary_affs_normalized.append(normalized_tensor)
                
                quadratic_affs_normalized = []
                for tensor in quadratic_affs:
                    range_val = tensor.max() - tensor.min()
                    if range_val != 0:
                        normalized_tensor = (tensor - tensor.min()) / range_val
                    else:
                        normalized_tensor = tensor - tensor.min()  # Or some other fallback mechanism
                    quadratic_affs_normalized.append(normalized_tensor)
                
                if self.training:
                    unary_affs_list = [-unary_affs_normalized[b] + 0.5 * data_dict['new_perm_mat'][b, :data_dict['ns_1'][b], :data_dict['ns_2'][b]] for b in range(batch_size)]
                else:
                    unary_affs_list = [-unary_affs_normalized[b] for b in range(batch_size)]
                quadratic_affs_list = [-0.1 * quadratic_affs_normalized[b][:data_dict['views_1'][b].num_edges, :data_dict['views_2'][b].num_edges] for b in range(batch_size)]
                # quadratic_affs_list = [-quadratic_affs_normalized[b][:data_dict['views_1'][b].num_edges, :data_dict['views_2'][b].num_edges] for b in range(batch_size)]
                
                matching = gm_solver(unary_affs_list, quadratic_affs_list)
                
                data_dict.update({'ds_mat': None,
                                  'perm_mat': matching,
                                  'aff_mat': K
                                  })
            else:
                # * NGM qap solver
                for i in range(self.gnn_layer):
                    gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                    emb_K, emb = gnn_layer(A, emb_K, emb, data_dict['ns_1'], data_dict['ns_2']) #, norm=False)

                v = self.classifier(emb)
                s = v.view(v.shape[0], tgt_len, -1).transpose(1, 2)

                if self.training or cfg.GCGM.GUMBEL_SK <= 0:
                    ss = self.sinkhorn(s, data_dict['ns_1'], data_dict['ns_2'], dummy_row=True)
                    # ! print(ss)
                    x = hungarian(ss, data_dict['ns_1'], data_dict['ns_2'])
                else:
                    gumbel_sample_num = cfg.GCGM.GUMBEL_SK
                    if self.training:
                        gumbel_sample_num //= 10
                    ss_gumbel = self.gumbel_sinkhorn(s, data_dict['ns_1'], data_dict['ns_2'], sample_num=gumbel_sample_num, dummy_row=True)

                    repeat = lambda x, rep_num=gumbel_sample_num: torch.repeat_interleave(x, rep_num, dim=0)
                    if not self.training:
                        ss_gumbel = hungarian(ss_gumbel, repeat(data_dict['ns_1']), repeat(data_dict['ns_2']))
                    ss_gumbel = ss_gumbel.reshape(batch_size, gumbel_sample_num, ss_gumbel.shape[-2], ss_gumbel.shape[-1])

                    if ss_gumbel.device.type == 'cuda':
                        dev_idx = ss_gumbel.device.index
                        free_mem = gpu_free_memory(dev_idx) - 100 * 1024 ** 2 # 100MB as buffer for other computations
                        K_mem_size = K.element_size() * K.nelement()
                        max_repeats = free_mem // K_mem_size
                        if max_repeats <= 0:
                            print('Warning: GPU may not have enough memory')
                            max_repeats = 1
                    else:
                        max_repeats = gumbel_sample_num

                    obj_score = []
                    for idx in range(0, gumbel_sample_num, max_repeats):
                        if idx + max_repeats > gumbel_sample_num:
                            rep_num = gumbel_sample_num - idx
                        else:
                            rep_num = max_repeats
                        obj_score.append(
                            objective_score(
                                ss_gumbel[:, idx:(idx+rep_num), :, :].reshape(-1, ss_gumbel.shape[-2], ss_gumbel.shape[-1]),
                                repeat(K, rep_num)
                            ).reshape(batch_size, -1)
                        )
                    obj_score = torch.cat(obj_score, dim=1)
                    min_obj_score = obj_score.min(dim=1)
                    ss = ss_gumbel[torch.arange(batch_size), min_obj_score.indices.cpu(), :, :]
                    x = hungarian(ss, repeat(data_dict['ns_1']), repeat(data_dict['ns_2']))

                data_dict.update({
                    'ds_mat': ss,
                    'perm_mat': x,
                    'aff_mat': K
                })
        keys_to_remove = ['G_1', 'H_1', 'G_2', 'H_2', 'K1G', 'K1H']    
        for key in keys_to_remove:
            data_dict.pop(key, None)
        return data_dict
    
    def update_graphs(self, data_dict):
        """ update source graphs and target graphs with the features extracted from the backbone network
        """
        batch_size = data_dict['batch_size'] // len(cfg.GPUS)
        ns_src, ns_tgt = data_dict['ns']
        
        src_graphs, tgt_graphs = data_dict['pyg_graphs']
        if not isinstance(src_graphs, pyg.data.Batch): # list of graphs
            src_graphs = next(iter(DataLoader(src_graphs, batch_size=batch_size, shuffle=False)))
            tgt_graphs = next(iter(DataLoader(tgt_graphs, batch_size=batch_size, shuffle=False)))
        
        # * save new graphs with nodes features extracted from the source image via backbone
        src_graphs, tgt_graphs = src_graphs.to(data_dict['U_src'].device), tgt_graphs.to(data_dict['U_tgt'].device)
        # * U_src_rshpd.shape = n \times GCGM.FEATURE_CHANNEL
        U_src_rshpd, U_tgt_rshpd = torch.cat([data_dict['U_src'][b, :, :ns_src[b]].squeeze().t() for b in range(batch_size)], dim=0), torch.cat([data_dict['U_tgt'][b, :, :ns_tgt[b]].squeeze().t() for b in range(batch_size)], dim=0)
        F_src_rshpd, F_tgt_rshpd = torch.cat([data_dict['F_src'][b, :, :ns_src[b]].squeeze().t() for b in range(batch_size)], dim=0), torch.cat([data_dict['F_tgt'][b, :, :ns_tgt[b]].squeeze().t() for b in range(batch_size)], dim=0)
        # * UF_src_rshpd.shape = n \times (2 * GCGM.FEATURE_CHANNEL)
        try:
            U_src_rshpd.size(1)
        except:
            # * for synthetic data
            U_src_rshpd, F_src_rshpd = U_src_rshpd.unsqueeze(1), F_src_rshpd.unsqueeze(1)
            U_tgt_rshpd, F_tgt_rshpd = U_tgt_rshpd.unsqueeze(1), F_tgt_rshpd.unsqueeze(1)
        UF_src, UF_tgt = torch.cat([U_src_rshpd, F_src_rshpd], dim=1), torch.cat([U_tgt_rshpd, F_tgt_rshpd], dim=1)
        src_graphs.x, tgt_graphs.x = UF_src, UF_tgt
        if self.message_pass_node_features is not None:
            # * SplineConv
            src_graphs, tgt_graphs = self.message_pass_node_features(src_graphs), self.message_pass_node_features(tgt_graphs)
        data_dict.update({'src_graphs': src_graphs, 
                          'tgt_graphs': tgt_graphs,
                          'UF_src': UF_src,
                          'UF_tgt': UF_tgt
                        })
        return data_dict
    
    def encode(self, data_dict):
        """ encoder original source graph and target graph
        """
        batch_size = data_dict['batch_size'] // len(cfg.GPUS)
        ns_src, ns_tgt = data_dict['ns']
        
        UF_src_enc, Glob_src, _ = self.encoder(data_dict['src_graphs'].x, data_dict['src_graphs'].edge_index, data_dict['src_graphs'])
        UF_tgt_enc, Glob_tgt, _ = self.encoder(data_dict['tgt_graphs'].x, data_dict['tgt_graphs'].edge_index, data_dict['tgt_graphs'])

        UF_src_updt = torch.stack([F.pad(UF_src_enc[torch.sum(ns_src[:b]): torch.sum(ns_src[:b+1])], pad=(0, 0, 0, torch.max(ns_src) - ns_src[b]), mode='constant', value=0).t() for b in range(batch_size)])
        UF_tgt_updt = torch.stack([F.pad(UF_tgt_enc[torch.sum(ns_tgt[:b]): torch.sum(ns_tgt[:b+1])], pad=(0, 0, 0, torch.max(ns_tgt) - ns_tgt[b]), mode='constant', value=0).t() for b in range(batch_size)])
        
        Glob = torch.cat([Glob_src, Glob_tgt], dim=-1)

        data_dict.update({'UF_src_enc': UF_src_enc,
                          'UF_tgt_enc': UF_tgt_enc,
                          'UF_src_updt': UF_src_updt,
                          'UF_tgt_updt': UF_tgt_updt,
                          'Glob_src': Glob_src,
                          'Glob_tgt': Glob_tgt,
                          'Glob': Glob})
        
        return data_dict

    def encode_corrupted(self, data_dict):
        """ encode corrupted source graph and target graphs
        """
        
        data_dict = self.augmentation.apply(data_dict)
        batch_size = data_dict['batch_size'] // len(cfg.GPUS)
                
        # * X: feature matrix
        # * x_list: feature embedding of each layer
        UF_1_enc, Glob_1, _ = self.encoder(data_dict['views_1'].x, data_dict['views_1'].edge_index, data_dict['views_1'])
        UF_2_enc, Glob_2, _ = self.encoder(data_dict['views_2'].x, data_dict['views_2'].edge_index, data_dict['views_2'])
        
        # * UF_1_updated.shape = b \times GCGM.OUT_CHANNEL \times torch.max(data_dcit['ns_1'])
        ns_1 = torch.tensor([data_dict['views_1'][i].num_nodes for i in range(batch_size)], dtype=data_dict['ns'][0].dtype, device=data_dict['ns'][0].device)
        ns_2 = torch.tensor([data_dict['views_2'][i].num_nodes for i in range(batch_size)], dtype=data_dict['ns'][1].dtype, device=data_dict['ns'][1].device)
        UF_1_updt = torch.stack([F.pad(UF_1_enc[torch.sum(ns_1[:b]): torch.sum(ns_1[:b+1])], pad=(0, 0, 0, torch.max(ns_1) - ns_1[b]), mode='constant', value=0).t() for b in range(batch_size)])
        UF_2_updt = torch.stack([F.pad(UF_2_enc[torch.sum(ns_2[:b]): torch.sum(ns_2[:b+1])], pad=(0, 0, 0, torch.max(ns_2) - ns_2[b]), mode='constant', value=0).t() for b in range(batch_size)])
        Glob = torch.cat([Glob_1, Glob_2], dim=-1)
        
        # pos
        # * data_dict['views_1'].pos.shape = (b * torch.max(ns_1)) \times 2
        # * Pos_1.shape = b \times torch.max(ns_1) \times 2
        Pos_1 = torch.stack([F.pad(data_dict['views_1'].pos[torch.sum(ns_1[:b]): torch.sum(ns_1[:b+1])], pad=(0, 0, 0, torch.max(ns_1) - ns_1[b]), mode='constant', value=0) for b in range(batch_size)])
        Pos_2 = torch.stack([F.pad(data_dict['views_2'].pos[torch.sum(ns_2[:b]): torch.sum(ns_2[:b+1])], pad=(0, 0, 0, torch.max(ns_2) - ns_2[b]), mode='constant', value=0) for b in range(batch_size)])
        
        # * update the data_dict with the new features: X and X_list
        data_dict.update({'UF_1_enc': UF_1_enc,
                          'UF_2_enc': UF_2_enc,
                          'hidden_1': UF_1_enc,
                          'hidden_2': UF_2_enc,
                          'Glob_1': Glob_1,
                          'Glob_2': Glob_2,
                          'UF_1_updt': UF_1_updt,
                          'UF_2_updt': UF_2_updt,
                          'ns_1': ns_1,
                          'ns_2': ns_2,
                          'Glob': Glob,
                          'Pos_1': Pos_1,
                          'Pos_2': Pos_2})
        
        return data_dict
    
    def get_perm_mat(self, data_dict):
        """ get the permutation matrix of the two corrupted views
        """
        batch_size = data_dict['batch_size'] // len(cfg.GPUS)
        perm_mat = torch.zeros((batch_size, max(data_dict['ns_1']), max(data_dict['ns_2'])), device=data_dict['gt_perm_mat'].device)
        # * perm_mat after mixing up
        perm_mat_ = torch.zeros((batch_size, max(data_dict['ns_1']), max(data_dict['ns_2'])), device=data_dict['gt_perm_mat'].device)
        for i in range(batch_size):
            for j, d in enumerate(data_dict['views_1'][i].dummy_label[:, 0]):
                for k, d_ in enumerate(data_dict['views_2'][i].dummy_label[:, 0]):
                    if d.item() == d_.item() and d.item() != -1: # * exclude the dummy nodes
                        perm_mat[i, j, k] = 1
                            
            # * only apply mixup to src graphs
            for j, d in enumerate(data_dict['views_1'][i].dummy_label[:, 1]):
                for k, d_ in enumerate(data_dict['views_2'][i].dummy_label[:, 0]):
                    if d.item() == d_.item() and d.item() != -1: # * exclude the dummy nodes
                        perm_mat_[i, j, k] = 1
                            
        data_dict.update({'new_perm_mat': perm_mat,
                          'mix_perm_mat': perm_mat_})
        return data_dict
    
    def get_connectivity_matrices(self, data_dict):
        """ get the connectivity matrices G and H of two batches of corrupted views
        """
        batch_size = data_dict['batch_size'] // len(cfg.GPUS)
        # * num of edges in each view
        es_1, es_2 = [data_dict['views_1'][i].num_edges for i in range(batch_size)], [data_dict['views_2'][i].num_edges for i in range(batch_size)]
        
        G_1 = torch.zeros((batch_size, torch.max(data_dict['ns_1']), max(es_1)), dtype=torch.float32, device=data_dict['Gs'][0].device)
        H_1 = torch.zeros((batch_size, torch.max(data_dict['ns_1']), max(es_1)), dtype=torch.float32, device=data_dict['Hs'][0].device)
        
        G_2 = torch.zeros((batch_size, torch.max(data_dict['ns_2']), max(es_2)), dtype=torch.float32, device=data_dict['Gs'][1].device)
        H_2 = torch.zeros((batch_size, torch.max(data_dict['ns_2']), max(es_2)), dtype=torch.float32, device=data_dict['Hs'][1].device)
                
        for b in range(batch_size):
            g_1, g_2 = data_dict['views_1'][b], data_dict['views_2'][b]
            A_view_1 = Net.get_adjacency_matrix(g_1.num_nodes, g_1.edge_index)
            A_view_2 = Net.get_adjacency_matrix(g_2.num_nodes, g_2.edge_index)
            
            # * view_1
            edge_idx = 0
            n = g_1.num_nodes
            for i in range(n):
                if cfg.GRAPH.SYM_ADJACENCY:
                    range_j = range(n)
                else:
                    range_j = range(i, n)
                for j in range_j:
                    if A_view_1[i, j] == 1:
                        G_1[b, i, edge_idx] = 1
                        H_1[b, j, edge_idx] = 1
                        edge_idx += 1
            
            # * view_2
            edge_idx = 0
            n = g_2.num_nodes
            for i in range(n):
                if cfg.GRAPH.SYM_ADJACENCY:
                    range_j = range(n)
                else:
                    range_j = range(i, n)
                for j in range_j:
                    if A_view_2[i, j] == 1:
                        G_2[b, i, edge_idx] = 1
                        H_2[b, j, edge_idx] = 1
                        edge_idx += 1
            
        data_dict.update({'G_1': G_1,
                          'H_1': H_1,
                          'G_2': G_2,
                          'H_2': H_2})
        return data_dict
    
    def get_kronecker_product(self, data_dict):
        
        G1, G2 = data_dict['G_1'], data_dict['G_2']
        H1, H2 = data_dict['H_1'], data_dict['H_2']
        G1, G2 = G1.cpu().detach().numpy(), G2.cpu().detach().numpy()
        H1, H2 = H1.cpu().detach().numpy(), H2.cpu().detach().numpy()
        if cfg.FP16:
            sparse_dtype = np.float16
        else:
            sparse_dtype = np.float32
        K1G = [kronecker_sparse(x, y).astype(sparse_dtype) for x, y in zip(G2, G1)]  # 1 as source graph, 2 as target graph
        K1H = [kronecker_sparse(x, y).astype(sparse_dtype) for x, y in zip(H2, H1)]
        K1G = CSRMatrix3d(K1G)
        K1H = CSRMatrix3d(K1H).transpose()
        data_dict.update({'K1G': K1G.to(data_dict['KGHs'][0].device),
                          'K1H': K1H.to(data_dict['KGHs'][1].device)})
        return data_dict
    
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
    def normalize_over_channels(x, epsilon=1e-8):
        channel_norms = torch.norm(x, dim=1, keepdim=True)
        channel_norms = torch.where(channel_norms < epsilon, torch.tensor(1.0, device=x.device), channel_norms)
        return x / channel_norms
    # def normalize_over_channels(x):
    #     channel_norms = torch.norm(x, dim=1, keepdim=True)
    #     return x / channel_norms
