from easydict import EasyDict as edict

__C = edict()

dataset_cfg = __C
# Pascal VOC 2011 dataset with keypoint annotations
# * train and val split
__C.PascalVOCSplit = edict()
__C.PascalVOCSplit.KPT_ANNO_DIR = 'data/PascalVOC/annotations/'  # keypoint annotation
__C.PascalVOCSplit.ROOT_DIR = 'data/PascalVOC/TrainVal/VOCdevkit/VOC2011/'  # original VOC2011 dataset
__C.PascalVOCSplit.SET_SPLIT = 'data/PascalVOC/voc2011_pairs.npz'  # set split path
__C.PascalVOCSplit.CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                         'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                         'tvmonitor']


# Willow ObjectClass dataset with keypoint annotations
# * train and val split
__C.WillowObjectSplit = edict()
__C.WillowObjectSplit.ROOT_DIR = 'data/WillowObject/WILLOW-ObjectClass'
__C.WillowObjectSplit.CLASSES = ['Car', 'Duck', 'Face', 'Motorbike', 'Winebottle']
__C.WillowObjectSplit.KPT_LEN = 10
__C.WillowObjectSplit.TRAIN_NUM = 20
__C.WillowObjectSplit.SPLIT_OFFSET = 0
__C.WillowObjectSplit.TRAIN_SAME_AS_TEST = False
__C.WillowObjectSplit.RAND_OUTLIER = 0

# Synthetic dataset
__C.MixedSynthetic = edict()
__C.SYNTHETIC = edict()
__C.SYNTHETIC.DIM = 1024
__C.SYNTHETIC.TRAIN_NUM = 100  # training graphs
__C.SYNTHETIC.TEST_NUM = 100  # testing graphs
__C.SYNTHETIC.MIXED_DATA_NUM = 10  # num of samples in mixed Synthetic test
__C.SYNTHETIC.RANDOM_EXP_ID = 0  # id of random experiment
__C.SYNTHETIC.EDGE_DENSITY = 0.3  # edge_num = X * node_num^2 / 4
__C.SYNTHETIC.KPT_NUM = 10  # number of nodes (inliers)
__C.SYNTHETIC.OUT_NUM = 0 # number of outliers
__C.SYNTHETIC.FEAT_GT_UNIFORM = 1.  # reference node features in uniform(-X, X) for each dimension
__C.SYNTHETIC.FEAT_NOISE_STD = 0.1  # corresponding node features add a random noise ~ N(0, X^2)
__C.SYNTHETIC.POS_GT_UNIFORM = 256.  # reference keypoint position in image: uniform(0, X)
__C.SYNTHETIC.POS_AFFINE_DXY = 50.  # corresponding position after affine transform: t_x, t_y ~ uniform(-X, X)
__C.SYNTHETIC.POS_AFFINE_S_LOW = 0.8  # corresponding position after affine transform: s ~ uniform(S_LOW, S_HIGH)
__C.SYNTHETIC.POS_AFFINE_S_HIGH = 1.2
__C.SYNTHETIC.POS_AFFINE_DTHETA = 60.  # corresponding position after affine transform: theta ~ uniform(-X, X)
__C.SYNTHETIC.POS_NOISE_STD = 10.  # corresponding position add a random noise ~ N(0, X^2) after affine transform

#SPair71k dataset
__C.SPair71k = edict()
__C.SPair71k.ROOT_DIR = 'data/SPair-71k'
__C.SPair71k.TRAIN_DIFF_PARAMS = {}
__C.SPair71k.EVAL_DIFF_PARAMS = {}
__C.SPair71k.COMB_CLS = False
__C.SPair71k.SIZE = 'large'