import copy
import pickle
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
# * split train and val
from sklearn.model_selection import train_test_split
from src.dataset.base_dataset import BaseDataset
from src.utils.config import cfg


class MixedSynthetic(BaseDataset):
    def __init__(self, sets, rate_1, rate_2, obj_resize):
        super(MixedSynthetic, self).__init__()
        self.rand_size = cfg.SYNTHETIC.MIXED_DATA_NUM
        self.classes = list(range(self.rand_size))
        self.obj_resize = obj_resize

        self.data_list = [Synthetic(sets, rate_1, rate_2, obj_resize, i) for i in self.classes]

    def get_pair(self, cls=None, problem=cfg.PROBLEM.TYPE, shuffle=True, tgt_outlier=False, src_outlier=False):
        if cls is None:
            cls = random.choice(self.classes)
        return self.data_list[cls].get_pair(cls, problem, shuffle, tgt_outlier, src_outlier)

    def get_multi(self, cls=None, num=2, shuffle=True):
        if cls is None:
            cls = random.choice(self.classes)
        return self.data_list[cls].get_multi(num=num, shuffle=shuffle)

class Synthetic(BaseDataset):
    def __init__(self, sets, rate_1, rate_2, obj_resize, exp_id=None):
        super(Synthetic, self).__init__()
        self.sets = sets
        self.rate_1 = rate_1
        self.rate_2 = rate_2
        self.val_split_size = 1 - cfg.TRAIN.SPLIT
        self.classes = ['synthetic']
        self.obj_resize = obj_resize

        self.train_num = cfg.SYNTHETIC.TRAIN_NUM
        self.test_num = cfg.SYNTHETIC.TEST_NUM
        self.exp_id = exp_id if exp_id is not None else cfg.SYNTHETIC.RANDOM_EXP_ID

        self.inlier_num = cfg.SYNTHETIC.KPT_NUM
        self.outlier_num = cfg.SYNTHETIC.OUT_NUM
        self.dimension = cfg.SYNTHETIC.DIM
        self.gt_feat_high = cfg.SYNTHETIC.FEAT_GT_UNIFORM
        self.gt_feat_low = - cfg.SYNTHETIC.FEAT_GT_UNIFORM
        self.feat_noise = cfg.SYNTHETIC.FEAT_NOISE_STD
        self.gt_pos_high = cfg.SYNTHETIC.POS_GT_UNIFORM
        self.gt_pos_low = 0
        #self.edge_density = cfg.Synthetic.EDGE_DENSITY
        self.affine_dxy_high = cfg.SYNTHETIC.POS_AFFINE_DXY
        self.affine_dxy_low = - cfg.SYNTHETIC.POS_AFFINE_DXY
        # * delta_s
        self.affine_s_high = cfg.SYNTHETIC.POS_AFFINE_S_HIGH
        self.affine_s_low = cfg.SYNTHETIC.POS_AFFINE_S_LOW
        self.affine_theta_high = cfg.SYNTHETIC.POS_AFFINE_DTHETA
        self.affine_theta_low = - cfg.SYNTHETIC.POS_AFFINE_DTHETA
        # * delta_n
        self.pos_noise = cfg.SYNTHETIC.POS_NOISE_STD

        self.cache_name = 'synthetic_train{}_test{}_in{}_out{}_dim{}_feat{:.2f}n{:.2f}_xy{:.2f},{:.2f}_s{:.2f},{:.2f}_theta{:.0f},{:.0f}_pos{:.2f}n{:.2f}_id{}.pkl'.format(
            self.train_num, self.test_num, self.inlier_num, self.outlier_num,
            self.dimension, self.gt_feat_high, self.feat_noise,
            self.affine_dxy_high, self.affine_dxy_low, self.affine_s_high, self.affine_s_low, self.affine_theta_high, self.affine_theta_low,
            self.gt_pos_high, self.pos_noise,
            self.exp_id
        )
        self.cache_path = Path(cfg.CACHE_PATH) / 'synthetic' / self.cache_name

        if not self.cache_path.parent.exists():
            self.cache_path.parent.mkdir(parents=True)

        if self.cache_path.exists():
            print('loading dataset from {}'.format(self.cache_path))
            with self.cache_path.open(mode='rb') as f:
                data_dict = pickle.load(f)
        else:
            print('caching dataset to {}'.format(self.cache_path))
            self.data_feat = np.random.uniform(self.gt_feat_low, self.gt_feat_high, (self.dimension, self.inlier_num))
            self.data_pos = np.random.uniform(self.gt_pos_low, self.gt_pos_high, (2, self.inlier_num))
            data_dict = self.__gen_data()
            with self.cache_path.open(mode='wb') as f:
                pickle.dump(data_dict, f)
                
        # if self.sets == 'train':
        #     num = int(len(self.data_list) * self.rate_1)
        #     num = max(2, num)
        #     if self.rate_1 > 0.5:
        #         self.data_list = self.data_list[:num]
        #     else:
        #         self.data_list = self.data_list[-num:]
        #         self.sets = 'validation'
        if self.sets in ['train', 'validation']:
            if self.val_split_size > 0:
                train, val = train_test_split(data_dict['train'], test_size=self.val_split_size, random_state=cfg.RANDOM_SEED)
                if self.sets == 'train':
                    self.data_list = train
                else:
                    self.data_list = val
        else:
            self.data_list = data_dict[self.sets]

    def __gen_data(self):
        """
        Generate random data and cache them into files
        """
        data_dict = dict()
        for period, sample_num in zip(['train', 'test'], [self.train_num, self.test_num]):
            data_lst = []
            for i in range(sample_num):
                data_lst.append(self.__gen_anno_dict())
            data_dict[period] = data_lst
        return data_dict

    def __gen_anno_dict(self, is_src=False):
        """
        Generate an annotation dict according to is_src True or False
        :param is_src: get a source point (i.e. ground truth point w/o noise and outlier) or not
        """
        outlier_num = random.randint(0, self.outlier_num)
        # ourlier_num = self.outlier_num
        if is_src:
            data_feat = self.data_feat.copy()
            data_pos = np.concatenate((self.data_pos, np.ones((1, self.inlier_num))), axis=0)

        else:
            outlier_feat = np.random.uniform(self.gt_feat_low, self.gt_feat_high, (self.dimension, outlier_num))
            outlier_pos = np.random.uniform(self.gt_pos_low, self.gt_pos_high, (2, outlier_num))
            
            data_feat = np.concatenate((self.data_feat, outlier_feat), axis=1)
            data_pos = np.concatenate(
                (
                    np.concatenate((self.data_pos, outlier_pos), axis=1),
                    np.ones((1, self.inlier_num + outlier_num))
                ), 
                axis=0)
            
            # feature distortion
            data_feat = data_feat + np.random.normal(0, self.feat_noise, data_feat.shape)

            # position distortion
            tx = np.random.uniform(self.affine_dxy_low, self.affine_dxy_high)
            ty = np.random.uniform(self.affine_dxy_low, self.affine_dxy_high)
            # * delta_s: scaling factor
            s = np.random.uniform(self.affine_s_low, self.affine_s_high)
            theta = np.random.uniform(self.affine_theta_low, self.affine_theta_high) * np.pi / 180
            aff = np.array(
                [[s * np.cos(theta), -s * np.sin(theta), tx],
                 [s * np.sin(theta), s * np.cos(theta), ty],
                 [0, 0, 1]]
            )
            data_pos = np.matmul(aff, data_pos)[:2, :]
            # * sigma_n
            data_pos = data_pos + np.random.normal(0, self.pos_noise, data_pos.shape)
            
        keypoint_list = []
        for idx in range(self.inlier_num + (0 if is_src else outlier_num)):
            keypoint = data_pos[:, idx]
            attr = dict()
            attr['name'] = idx
            attr['x'] = float(keypoint[0])
            attr['y'] = float(keypoint[1])
            attr['feat'] = data_feat[:, idx]
            keypoint_list.append(attr)

        anno_dict = dict()
        anno_dict['keypoints'] = keypoint_list
        # the following keys are of no use, but kept for dataloader interface
        anno_dict['image'] = None
        anno_dict['bounds'] = self.gt_pos_low, self.gt_pos_low, self.gt_pos_high, self.gt_pos_high
        anno_dict['ori_sizes'] = (self.gt_pos_high, self.gt_pos_high)
        anno_dict['cls'] = 'synthetic'

        return anno_dict
    
    def get_ssl_pair(self, cls=None, shuffle=True, tgt_outlier=True, src_outlier=True):
        anno_pair = []
        for anno_dict in random.sample(self.data_list, 1):
            anno_dict = deepcopy(anno_dict)
            if shuffle:
                random.shuffle(anno_dict['keypoints'])
            anno_pair.append(anno_dict)
        
        # * tgt
        new_dict = copy.deepcopy(anno_pair[0])
        if shuffle:
            index = torch.randperm(len(anno_pair[0]['keypoints']))
        else:
            index = torch.arange(len(anno_pair[0]['keypoints']))
        new_dict['index'] = index
        new_dict['keypoints'] = [{'name': new_dict['keypoints'][i]['name'],
                                  'x': new_dict['keypoints'][i]['x'], 
                                  'y': new_dict['keypoints'][i]['y'],
                                  'feat': new_dict['keypoints'][i]['feat']} for i in index]
        anno_pair.append(new_dict)

        perm_mat = np.zeros([len(_['keypoints']) for _ in anno_pair], dtype=np.float32)
        row_list = []
        col_list = []
        for i, keypoint in enumerate(anno_pair[0]['keypoints']):
            for j, _keypoint in enumerate(anno_pair[1]['keypoints']):
                if keypoint['name'] == _keypoint['name']:
                    # if keypoint['name'] < self.inlier_num:
                    perm_mat[i, j] = 1
                    row_list.append(i)
                    col_list.append(j)
                    break
        row_list.sort()
        col_list.sort()
        # perm_mat = perm_mat[row_list, :]
        # perm_mat = perm_mat[:, col_list]
        # anno_pair[0]['keypoints'] = [anno_pair[0]['keypoints'][i] for i in row_list]
        # anno_pair[1]['keypoints'] = [anno_pair[1]['keypoints'][j] for j in col_list]
        # if not src_outlier:
        #     perm_mat = perm_mat[row_list, :]
        #     anno_pair[0]['keypoints'] = [anno_pair[0]['keypoints'][i] for i in row_list]
        # if not tgt_outlier:
        #     perm_mat = perm_mat[:, col_list]
        #     anno_pair[1]['keypoints'] = [anno_pair[1]['keypoints'][j] for j in col_list]

        return anno_pair, perm_mat

    def get_pair(self, cls=None, problem=cfg.PROBLEM.TYPE, shuffle=True, tgt_outlier=False, src_outlier=False):
        """
        Randomly get a pair of objects from synthetic data
        :param cls: no use here
        :param shuffle: random shuffle the keypoints
        :return: (pair of data, groundtruth permutation matrix)
        """
        # * SSL
        if self.sets == "train" and problem == 'GCL':
            return self.get_ssl_pair(cls, shuffle)
        
        anno_pair = []
        for anno_dict in random.sample(self.data_list, 2):
            anno_dict = deepcopy(anno_dict)
            if shuffle:
                random.shuffle(anno_dict['keypoints'])
            anno_pair.append(anno_dict)

        perm_mat = np.zeros([len(_['keypoints']) for _ in anno_pair], dtype=np.float32)
        row_list = []
        col_list = []
        for i, keypoint in enumerate(anno_pair[0]['keypoints']):
            for j, _keypoint in enumerate(anno_pair[1]['keypoints']):
                if keypoint['name'] == _keypoint['name']:
                    if keypoint['name'] < self.inlier_num:
                        perm_mat[i, j] = 1
                        row_list.append(i)
                        col_list.append(j)
                        break
        row_list.sort()
        col_list.sort()
        # perm_mat = perm_mat[row_list, :]
        # perm_mat = perm_mat[:, col_list]
        # anno_pair[0]['keypoints'] = [anno_pair[0]['keypoints'][i] for i in row_list]
        # anno_pair[1]['keypoints'] = [anno_pair[1]['keypoints'][j] for j in col_list]
        if not src_outlier:
            perm_mat = perm_mat[row_list, :]
            anno_pair[0]['keypoints'] = [anno_pair[0]['keypoints'][i] for i in row_list]
        if not tgt_outlier:
            perm_mat = perm_mat[:, col_list]
            anno_pair[1]['keypoints'] = [anno_pair[1]['keypoints'][j] for j in col_list]
        
        return anno_pair, perm_mat

    def get_multi(self, cls=None, num=2, shuffle=True):
        """
        Randomly get multiple objects from synthetic dataset for multi-matching.
        The first data is fetched with all appeared keypoints, and the rest images are fetched with only inliers.
        :param cls: None for random class, or specify for a certain set
        :param num: number of objects to be fetched
        :param shuffle: random shuffle the keypoints
        :return: (list of data, list of permutation matrices)
        """
        anno_list = []
        for anno_dict in random.sample(self.data_list, num):
            anno_dict = deepcopy(anno_dict)
            if shuffle:
                random.shuffle(anno_dict['keypoints'])
            anno_list.append(anno_dict)

        perm_mat = [np.zeros([len(anno_list[0]['keypoints']), len(x['keypoints'])], dtype=np.float32) for x in anno_list]
        row_list = []
        col_lists = []
        for i in range(num):
            col_lists.append([])
        for i, keypoint in enumerate(anno_list[0]['keypoints']):
            kpt_idx = []
            for anno_dict in anno_list:
                kpt_name_list = [x['name'] for x in anno_dict['keypoints']]
                if keypoint['name'] in kpt_name_list:
                    kpt_idx.append(kpt_name_list.index(keypoint['name']))
                else:
                    kpt_idx.append(-1)
            row_list.append(i)
            for k in range(num):
                j = kpt_idx[k]
                if j != -1:
                    col_lists[k].append(j)
                    perm_mat[k][i, j] = 1

        row_list.sort()
        for col_list in col_lists:
            col_list.sort()

        for k in range(num):
            perm_mat[k] = perm_mat[k][row_list, :]
            perm_mat[k] = perm_mat[k][:, col_lists[k]]
            anno_list[k]['keypoints'] = [anno_list[k]['keypoints'][j] for j in col_lists[k]]
            perm_mat[k] = perm_mat[k].transpose()

        return anno_list, perm_mat
