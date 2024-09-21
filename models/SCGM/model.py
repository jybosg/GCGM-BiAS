import copy
import torch
import itertools

from models.NGM.geo_edge_feature import geo_edge_feature
from models.GCGM.affinity_layer import InnerProductWithWeightsAffinity, GaussianWithWeightAffinity
from models.BBGM.sconv_archs import SiameseSConvOnNodes, SiameseNodeFeaturesToEdgeFeatures
from lpmp_py import GraphMatchingModule
from lpmp_py import MultiGraphMatchingModule
from src.feature_align import feature_align

from src.utils.config import cfg
from src.utils.c_loss import simclr_loss

from torch_geometric.loader import DataLoader
from src.utils.pad_tensor import pad_tensor
from src.factorize_graph_matching import construct_aff_mat

from src.backbone import *
CNN = eval(cfg.BACKBONE)

def lexico_iter(lex):
    return itertools.combinations(lex, 2)

def normalize_over_channels(x, epsilon=1e-8):
    channel_norms = torch.norm(x, dim=1, keepdim=True)
    channel_norms = torch.where(channel_norms < epsilon, torch.tensor(1.0, device=x.device), channel_norms)
    return x / channel_norms

def concat_features(embeddings, num_vertices):
    res = torch.cat([embedding[:, :num_v] for embedding, num_v in zip(embeddings, num_vertices)], dim=-1)
    return res.transpose(0, 1)

class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        # * SplineConv
        self.message_pass_node_features = SiameseSConvOnNodes(input_node_dim=cfg.BBGM.FEATURE_CHANNEL)
        self.sconv_params = list(self.message_pass_node_features.parameters())
            
        self.build_edge_features_from_node_features = SiameseNodeFeaturesToEdgeFeatures(total_num_nodes=cfg.BBGM.FEATURE_CHANNEL)
        
        self.global_state_dim = cfg.BBGM.FEATURE_CHANNEL
        self.vertex_affinity = InnerProductWithWeightsAffinity(self.global_state_dim, cfg.BBGM.FEATURE_CHANNEL)
        if cfg.BBGM.EDGE_FEATURE == 'cat':
            self.edge_affinity = InnerProductWithWeightsAffinity(self.global_state_dim,
                                                                 self.build_edge_features_from_node_features.num_edge_features)
        elif cfg.BBGM.EDGE_FEATURE == 'geo':
            self.edge_affinity = GaussianWithWeightAffinity(self.global_state_dim,
                                                            1,
                                                            cfg.BBGM.GAUSSIAN_SIGMA)
            
        self.rescale = cfg.PROBLEM.RESCALE
        if cfg.PROBLEM.SSL and cfg.SSL.C_LOSS:
            # self.mlp = nn.Sequential(
            #     nn.Linear(1024, 1024, bias=False),
            #     nn.ReLU(),
            #     nn.Linear(1024, 256, bias=False)
            # )
            # * for visual datasets, cfg.SSL.IN_DIM == cfg.BBGM.FEATURE_CHANNEL == cfg.NGM.FEATURE_CHANNEL * 2
            self.mlp = nn.Sequential(
                nn.Linear(cfg.SSL.IN_DIM, cfg.SSL.IN_DIM, bias=False),
                nn.ReLU(),
                nn.Linear(cfg.SSL.IN_DIM, cfg.SSL.OUT_DIM, bias=False)
            )

    def forward(self, data_dict):
        
        batch_size = data_dict['batch_size'] // len(cfg.GPUS)
        # * visual dataset
        if 'images' in data_dict:
            # print('\nmodel - pyg_graphs: {}\n'.format(data_dict['pyg_graphs']))
            # print('\nmodel - Ps: {}\n'.format(data_dict['Ps']))
            images = data_dict['images']
            points = data_dict['Ps']
            n_points = data_dict['ns']
            graphs = data_dict['pyg_graphs'] 
            num_graphs = len(images)
        # * non-visual dataset
        elif 'features' in data_dict:
            features = data_dict['features']
            points = data_dict['Ps']
            n_points = data_dict['ns']
            graphs = data_dict['pyg_graphs']
            Ps = data_dict['Ps']
            # * connectivity matrices
            Gs = data_dict['Gs']
            Hs = data_dict['Hs']
            num_graphs = len(features)
        K_G, K_H = data_dict['KGHs']

        if cfg.PROBLEM.TYPE in ['2GM', 'GCL', 'GCLTS'] and 'gt_perm_mat' in data_dict:
            gt_perm_mats = [data_dict['gt_perm_mat']]
        elif cfg.PROBLEM.TYPE == 'MGM' and 'gt_perm_mat' in data_dict:
            perm_mat_list = data_dict['gt_perm_mat']
            gt_perm_mats = [torch.bmm(pm_src, pm_tgt.transpose(1, 2)) for pm_src, pm_tgt in lexico_iter(perm_mat_list)]
        else:
            raise ValueError('Ground truth information is required during training.')

        global_list = []
        orig_graph_list = []
        old_shape = data_dict['gt_perm_mat'][0].shape
        max_shape = max(data_dict['gt_perm_mat'][0].shape)
        perm_old = torch.zeros([len(data_dict['gt_perm_mat']), max_shape, max_shape],
                               device=data_dict['gt_perm_mat'].device)
        perm_old[:, 0:old_shape[0], 0:old_shape[1]] = data_dict['gt_perm_mat']
        perm_list = []
        node_feature_cl = []
        idx = 0
        perms = []
        
        if 'images' in data_dict:
            for image, p, n_p, graph in zip(images, points, n_points, graphs):
                if type(graph) == list:
                    graph = next(iter(DataLoader(graph, batch_size=batch_size, shuffle=False)))
                    
                idx += 1
                # extract feature
                nodes = self.node_layers(image)
                edges = self.edge_layers(nodes)

                global_list.append(self.final_layers(edges).reshape((nodes.shape[0], -1)))
                nodes = normalize_over_channels(nodes)
                edges = normalize_over_channels(edges)

                # arrange features
                U = concat_features(feature_align(nodes, p, n_p, self.rescale), n_p)
                F = concat_features(feature_align(edges, p, n_p, self.rescale), n_p)
                node_features = torch.cat((U, F), dim=1)

                # if cfg.PROBLEM.SSL:
                if cfg.PROBLEM.SSL and self.training:
                    n_p_c = n_p.cpu().detach().numpy()

                    if cfg.SSL.C_LOSS:
                        node_features_format = torch.zeros([perm_old.shape[0], perm_old.shape[1], node_features.shape[1]],
                                                        device=perm_old.device)  # B x N x D
                        pre = 0
                        for i in range(len(n_p_c)):
                            node_features_format[i][0: n_p_c[i]] = node_features[pre: pre + n_p_c[i]]
                            pre += int(n_p_c[i])
                        node_feature_cl.append(node_features_format)

                    if idx == 2: # * only apply mix-up to target view
                        pre = 0
                        rate = cfg.SSL.MIX_RATE
                        for i in range(len(n_p_c)):
                            perm = torch.randperm(int(n_p_c[i]))
                            perms.append(perm)
                            if not cfg.SSL.MIX_DETACH:
                                node_features[pre: pre + n_p_c[i]] = node_features[pre: pre + n_p_c[i]] * (
                                            1 - rate) + rate * \
                                                                    node_features[pre + perm]
                            else:
                                node_features[pre: pre + n_p_c[i]] = node_features[pre: pre + n_p_c[i]] * (
                                            1 - rate) + rate * \
                                                                    node_features[pre + perm].detach()
                            pre += int(n_p_c[i])

                graph.x = node_features
                graph = self.message_pass_node_features(graph)
                orig_graph = self.build_edge_features_from_node_features(graph)
                orig_graph_list.append(orig_graph)
        
        # * non-visual dataset 
        elif 'features' in data_dict:
            for feature, p, n_p, graph, P, G, H in zip(features, points, n_points, graphs, Ps, Gs, Hs):
                
                if type(graph) == list:
                    graph = next(iter(DataLoader(graph, batch_size=batch_size, shuffle=False)))
                    
                idx += 1
                # * normalize feature
                feature = normalize_over_channels(feature)
                node_features = torch.cat([feature[b, :, :n_p[b]].squeeze().t() for b in range(batch_size)], dim=0)

                # if cfg.PROBLEM.SSL:
                if cfg.PROBLEM.SSL and self.training:
                    n_p_c = n_p.cpu().detach().numpy()

                    if cfg.SSL.C_LOSS:
                        node_features_format = torch.zeros([perm_old.shape[0], perm_old.shape[1], node_features.shape[1]],
                                                        device=perm_old.device)  # B x N x D
                        pre = 0
                        for i in range(len(n_p_c)):
                            node_features_format[i][0: n_p_c[i]] = node_features[pre: pre + n_p_c[i]]
                            pre += int(n_p_c[i])
                        node_feature_cl.append(node_features_format)

                    if idx == 2: # * only apply mix-up to target view
                        pre = 0
                        rate = cfg.SSL.MIX_RATE
                        for i in range(len(n_p_c)):
                            perm = torch.randperm(int(n_p_c[i]))
                            perms.append(perm)
                            if not cfg.SSL.MIX_DETACH:
                                node_features[pre: pre + n_p_c[i]] = node_features[pre: pre + n_p_c[i]] * (
                                            1 - rate) + rate * \
                                                                    node_features[pre + perm]
                            else:
                                node_features[pre: pre + n_p_c[i]] = node_features[pre: pre + n_p_c[i]] * (
                                            1 - rate) + rate * \
                                                                    node_features[pre + perm].detach()
                            pre += int(n_p_c[i])
                
                # * global readout
                ns_cum = torch.cat((torch.zeros(1, dtype=n_p.dtype, device=n_p.device), torch.cumsum(n_p, dim=0)), dim=0)
                if cfg.BBGM.GLOBAL_READOUT == 'mean':
                    global_list.append(torch.cat([torch.mean(node_features[ns_cum[b]: ns_cum[b+1]], dim=0, keepdim=True) for b in range(batch_size)], dim=0))
                elif cfg.BBGM.GLOBAL_READOUT == 'max':
                    global_list.append(torch.cat([torch.max(node_features[ns_cum[b]: ns_cum[b+1]], dim=0, keepdim=True) for b in range(batch_size)], dim=0))
                
                graph.x = node_features
                if self.message_pass_node_features is not None:
                    graph = self.message_pass_node_features(graph)
                    
                if cfg.BBGM.EDGE_FEATURE == 'cat':
                    orig_graph = self.build_edge_features_from_node_features(graph)
                elif cfg.BBGM.EDGE_FEATURE == 'geo':
                    edge_feature = geo_edge_feature(P, G, H)[:, :1, :]
                    graph.edge_attr = torch.cat([edge_feature[b, :, :graph[b].edge_index.size(1)].view(graph[b].edge_index.size(1), 1) for b in range(batch_size)], dim=0)
                    orig_graph = [graph[b] for b in range(batch_size)]
                orig_graph_list.append(orig_graph)

        if self.training:
            if cfg.PROBLEM.SSL and cfg.SSL.C_LOSS:
                z1 = node_feature_cl[0]
                z2 = node_feature_cl[1]
                z2_ = torch.bmm(perm_old, z2)
                z1_cross = torch.zeros(z1.shape, device=z1.device)
                z2_cross = torch.zeros(z1.shape, device=z1.device)
                for i in range(len(z2_)):
                    non_zeros = z2_[i].sum(axis=1).nonzero()[:, 0]
                    z2_cross[i][0: len(non_zeros)] = z2_[i][non_zeros]
                    z1_cross[i][0: len(non_zeros)] = z1[i][non_zeros]
                c_loss = simclr_loss(torch.nn.functional.normalize(self.mlp(z1_cross), dim=-1),
                                    torch.nn.functional.normalize(self.mlp(z2_cross), dim=-1))
                data_dict['c_loss'] = c_loss

            if cfg.PROBLEM.SSL and not cfg.SSL.MIX_DETACH:
                perm_new = torch.zeros(perm_old.shape).to(perm_old.device)
                for i in range(len(perm_old)):
                    perm = perms[i]
                    for j in range(len(perm)):
                        perm_new[i][j, perm[j]] = 1
                perm_new_ = torch.bmm(perm_old, perm_new)[:, 0: old_shape[0], 0: old_shape[1]]
                data_dict['gt_perm_mat_old'] = copy.deepcopy(data_dict['gt_perm_mat'])
                data_dict['gt_perm_mat_new'] = perm_new_

        # * for non-visual datasets: cfg.SSL.USE_GLOBAL = False
        global_weights_list = [
            torch.cat([global_src, global_tgt], axis=-1) for global_src, global_tgt in lexico_iter(global_list)
        ]

        global_weights_list = [normalize_over_channels(g) for g in global_weights_list]

        unary_affs_list = [
            self.vertex_affinity([item.x for item in g_1], [item.x for item in g_2], global_weights,
                                use_global=cfg.SSL.USE_GLOBAL)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]

        # Similarities to costs
        unary_costs_list = [[-x for x in unary_costs] for unary_costs in unary_affs_list]

        if self.training:
            unary_costs_list = [
                [
                    x + 1.0 * gt[:dim_src, :dim_tgt]  # Add margin with alpha = 1.0
                    for x, gt, dim_src, dim_tgt in zip(unary_costs, gt_perm_mat, ns_src, ns_tgt)
                ]
                for unary_costs, gt_perm_mat, (ns_src, ns_tgt) in
                zip(unary_costs_list, gt_perm_mats, lexico_iter(n_points))
            ]
            # ! check the scale of unary_costs_list
            # print('unary_costs_list: {}'.format(unary_costs_list))
            
        if not cfg.BBGM.FIRST_ORDER:
            # set unary costs to zero
            unary_costs_list = [[torch.zeros_like(x) for x in unary_costs] for unary_costs in unary_costs_list]

        quadratic_costs_list = [
            self.edge_affinity([item.edge_attr for item in g_1], [item.edge_attr for item in g_2], global_weights,
                                use_global=cfg.SSL.USE_GLOBAL)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]
        
        # Similarities to costs
        quadratic_costs_list = [[-0.5 * x for x in quadratic_costs] for quadratic_costs in quadratic_costs_list]
        
        # ! check the scale of unary_costs_list
        # print('quadratic_costs_list: {}'.format(quadratic_costs_list))
        
        # * affinity matrix
        Kp = torch.stack(pad_tensor(unary_costs_list[0]), dim=0)
        Ke = torch.stack(pad_tensor(quadratic_costs_list[0]), dim=0)
        K = construct_aff_mat(Ke, Kp, K_G, K_H)

        if cfg.BBGM.SOLVER_NAME == "LPMP":
            all_edges = [[item.edge_index for item in graph] for graph in orig_graph_list]
            gm_solvers = [
                GraphMatchingModule(
                    all_left_edges,
                    all_right_edges,
                    ns_src,
                    ns_tgt,
                    cfg.BBGM.LAMBDA_VAL,
                    cfg.BBGM.SOLVER_PARAMS,
                )
                for (all_left_edges, all_right_edges), (ns_src, ns_tgt) in zip(
                    lexico_iter(all_edges), lexico_iter(n_points)
                )
            ]
            matchings = [
                gm_solver(unary_costs, quadratic_costs)
                for gm_solver, unary_costs, quadratic_costs in zip(gm_solvers, unary_costs_list, quadratic_costs_list)
            ]
        elif cfg.BBGM.SOLVER_NAME == "LPMP_MGM":
            all_edges = [[item.edge_index for item in graph] for graph in orig_graph_list]
            gm_solver = MultiGraphMatchingModule(
                all_edges, n_points, cfg.BBGM.LAMBDA_VAL, cfg.BBGM.SOLVER_PARAMS)
            matchings = gm_solver(unary_costs_list, quadratic_costs_list)
        else:
            raise ValueError("Unknown solver {}".format(cfg.BBGM.SOLVER_NAME))

        if cfg.PROBLEM.TYPE in ['2GM', 'GCL', 'GCLTS']:
            data_dict.update({
                'ds_mat': None,
                'perm_mat': matchings[0],
                'aff_mat': K
            })
        elif cfg.PROBLEM.TYPE == 'MGM':
            indices = list(lexico_iter(range(num_graphs)))
            data_dict.update({
                'perm_mat_list': matchings,
                'graph_indices': indices,
                'gt_perm_mat_list': gt_perm_mats
            })

        return data_dict
