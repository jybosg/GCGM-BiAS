import torch
import torch.nn as nn
import torch.nn.functional as functional

import numpy as np

from src.feature_align import feature_align
from src.lap_solvers.hungarian import hungarian
from src.utils.pad_tensor import pad_tensor
from src.build_graphs import reshape_edge_feature
from models.NGM.geo_edge_feature import geo_edge_feature

from src.utils.config import cfg

# * learning-free solver
import pygmtools as pygm
pygm.BACKEND = 'pytorch'
import functools

from src.backbone import *
CNN = eval(cfg.BACKBONE)

class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        self.l2norm = nn.LocalResponseNorm(cfg.LF.FEATURE_CHANNEL * 2, alpha=cfg.LF.FEATURE_CHANNEL * 2, beta=0.5, k=0)
        self.univ_size = torch.tensor(cfg.LF.UNIV_SIZE)
        self.rescale = cfg.PROBLEM.RESCALE

    def forward(self, data_dict, **kwargs):

        if 'images' in data_dict:
            # real image data
            src, tgt = data_dict['images']
            P_src, P_tgt = data_dict['Ps']
            ns_src, ns_tgt = data_dict['ns']
            A_src, A_tgt = data_dict['As']
            G_src, G_tgt = data_dict['Gs']
            H_src, H_tgt = data_dict['Hs']
            K_G, K_H = data_dict['KGHs']

            # extract feature
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
            U_src = feature_align(src_node, P_src, ns_src, self.rescale)
            F_src = feature_align(src_edge, P_src, ns_src, self.rescale)
            U_tgt = feature_align(tgt_node, P_tgt, ns_tgt, self.rescale)
            F_tgt = feature_align(tgt_edge, P_tgt, ns_tgt, self.rescale)
            
            # node features
            UF_src = torch.cat([U_src, F_src], dim=1) 
            UF_tgt = torch.cat([U_tgt, F_tgt], dim=1)
            
        elif 'features' in data_dict:
            # * non-visible data
            UF_src, UF_tgt = data_dict['features']
            P_src, P_tgt = data_dict['Ps']
            ns_src, ns_tgt = data_dict['ns']
            A_src, A_tgt = data_dict['As']
            G_src, G_tgt = data_dict['Gs']
            H_src, H_tgt = data_dict['Hs']
            K_G, K_H = data_dict['KGHs']
            
            # src, tgt = data_dict['features']
            # U_src = src[:, :src.shape[1] // 2, :]
            # F_src = src[:, src.shape[1] // 2:, :]
            # U_tgt = tgt[:, :tgt.shape[1] // 2, :]
            # F_tgt = tgt[:, tgt.shape[1] // 2:, :]
            # UF_src = torch.cat([U_src, F_src], dim=1) 
            # UF_tgt = torch.cat([U_tgt, F_tgt], dim=1)
            UF_src = Net.normalize_over_channels(UF_src)
            UF_tgt = Net.normalize_over_channels(UF_tgt)
            
        else:
            raise ValueError('Unknown data type for this model.')
        
        if cfg.LF.EDGE_FEATURE == 'cat':
            X = reshape_edge_feature(UF_src, G_src, H_src)
            Y = reshape_edge_feature(UF_tgt, G_tgt, H_tgt)
        elif cfg.LF.EDGE_FEATURE == 'geo':
            X = geo_edge_feature(P_src, G_src, H_src)[:, :1, :]
            Y = geo_edge_feature(P_tgt, G_tgt, H_tgt)[:, :1, :]
            
        # Build affinity matrix
        conn_src, _, ne_src = pygm.utils.dense_to_sparse(A_src)
        conn_tgt, _, ne_tgt = pygm.utils.dense_to_sparse(A_tgt)
        
        UF_src = torch.transpose(UF_src, 1, 2)
        UF_tgt = torch.transpose(UF_tgt, 1, 2)
        X = torch.transpose(X, 1, 2)
        Y = torch.transpose(Y, 1, 2)
        
        gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=cfg.LF.GAUSSIAN_SIGMA) # set affinity function
        # https://pygmtools.readthedocs.io/en/latest/api/_autosummary/pygmtools.utils.build_aff_mat.html
        K = pygm.utils.build_aff_mat(UF_src, X, conn_src, UF_tgt, Y, conn_tgt, ns_src, ne_src, ns_tgt, ne_tgt, edge_aff_fn=gaussian_aff)
        
        if cfg.LF.SOLVER == 'IPFP':
            perm_mat = pygm.ipfp(K, ns_src, ns_tgt)
        elif cfg.LF.SOLVER == 'RRWM':
            ss = pygm.rrwm(K, ns_src, ns_tgt)
            perm_mat = pygm.hungarian(ss)
        elif cfg.LF.SOLVER == 'SM':
            ss = pygm.sm(K, ns_src, ns_tgt)
            perm_mat = pygm.hungarian(ss)
            
        data_dict.update({'ds_mat': None,
                          'perm_mat': perm_mat,
                          'aff_mat': K})
        return data_dict
    
    @staticmethod
    def normalize_over_channels(x, epsilon=1e-8):
        channel_norms = torch.norm(x, dim=1, keepdim=True)
        channel_norms = torch.where(channel_norms < epsilon, torch.tensor(1.0, device=x.device), channel_norms)
        return x / channel_norms
