from easydict import EasyDict as edict

__C = edict()

model_cfg = __C

# LF model options
__C.LF = edict()
__C.LF.FEATURE_CHANNEL = 1024
__C.LF.UNIV_SIZE = 10
