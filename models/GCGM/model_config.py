from easydict import EasyDict as edict
import torch.nn.functional as F

__C = edict()

model_cfg = __C

# GCGm model options
__C.GCGM = edict()
__C.GCGM.FEATURE_CHANNEL = 512
__C.GCGM.SK_ITER_NUM = 10
__C.GCGM.SK_EPSILON = 1e-10
__C.GCGM.SK_TAU = 0.005
__C.GCGM.MGM_SK_TAU = 0.005
__C.GCGM.GNN_FEAT = [16, 16, 16]
__C.GCGM.GNN_LAYER = 3
__C.GCGM.GAUSSIAN_SIGMA = 1.
__C.GCGM.SIGMA3 = 1.
__C.GCGM.WEIGHT2 = 1.
__C.GCGM.WEIGHT3 = 1.
__C.GCGM.EDGE_FEATURE = 'cat' # 'cat' or 'geo'
__C.GCGM.ORDER3_FEATURE = 'none' # 'cat' or 'geo' or 'none'
__C.GCGM.FIRST_ORDER = True
__C.GCGM.EDGE_EMB = False
__C.GCGM.SK_EMB = 1
__C.GCGM.GUMBEL_SK = 0 # 0 for no gumbel, other wise for number of gumbel samples
__C.GCGM.UNIV_SIZE = -1
__C.GCGM.POSITIVE_EDGES = True