import copy
import random
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
# * split train and val
from sklearn.model_selection import train_test_split
from src.dataset.base_dataset import BaseDataset
from src.ssl.augmentation import augmentation
from src.utils.config import cfg

'''
Important Notice: Face image 160 contains only 8 labeled keypoints (should be 10)
'''

class WillowObjectSplit(BaseDataset):
    def __init__(self, sets, rate_1, rate_2, obj_resize):
        """
        :param sets: 'train' or 'test'
        :param obj_resize: resized object size
        """
        super(WillowObjectSplit, self).__init__()
        self.classes = cfg.WillowObjectSplit.CLASSES
        self.kpt_len = [cfg.WillowObjectSplit.KPT_LEN for _ in cfg.WillowObjectSplit.CLASSES]

        self.root_path = Path(cfg.WillowObjectSplit.ROOT_DIR)
        self.obj_resize = obj_resize

        assert sets in ('train', 'validation', 'test'), 'No match found for dataset {}'.format(sets)
        self.sets = sets
        self.split_offset = cfg.WillowObjectSplit.SPLIT_OFFSET
        self.train_len = cfg.WillowObjectSplit.TRAIN_NUM
        if not sets == 'train':
            self.rate_1 = 1
        self.rate_1 = rate_1
        self.val_split_size = 1 - cfg.TRAIN.SPLIT
        self.train_len = int(self.train_len * rate_1)
        self.rand_outlier = cfg.WillowObjectSplit.RAND_OUTLIER
        self.rate_2 = rate_2

        self.mat_list = []
        for cls_name in self.classes:
            assert type(cls_name) is str
            cls_mat_list = [p for p in (self.root_path / cls_name).glob('*.mat')]
            if cls_name == 'Face':
                cls_mat_list.remove(self.root_path / cls_name / 'image_0160.mat')
                assert not self.root_path / cls_name / 'image_0160.mat' in cls_mat_list
            ori_len = len(cls_mat_list)
            num = int(rate_1 * ori_len)
            if num < 2:
                num = 2
            cls_mat_list = cls_mat_list[0: num]
            ori_len = len(cls_mat_list)
            if self.split_offset % ori_len + self.train_len <= ori_len:
                # if sets == 'train' and not cfg.WillowObjectSplit.TRAIN_SAME_AS_TEST:
                #     lst = cls_mat_list[self.split_offset % ori_len: (self.split_offset + self.train_len) % ori_len]
                #     if self.rate_1 > 0.5:
                #         random.seed(cfg.RANDOM_SEED)
                #         random.shuffle(lst)
                #         self.mat_list.append(lst[:int(len(lst)*self.rate_1)])
                #     else:
                #         random.seed(cfg.RANDOM_SEED)
                #         random.shuffle(lst)
                #         self.mat_list.append(lst[-int(len(lst)*self.rate_1):])
                #         self.sets = 'validation'
                if sets in ['train', 'validation'] and not cfg.WillowObjectSplit.TRAIN_SAME_AS_TEST:
                    lst = cls_mat_list[self.split_offset % ori_len: (self.split_offset + self.train_len) % ori_len]
                    if self.val_split_size > 0:
                        train_cls_mat_list, val_cls_mat_list = train_test_split(lst, test_size=self.val_split_size, random_state=cfg.RANDOM_SEED)
                        if sets == 'train':
                            self.mat_list.append(train_cls_mat_list)
                        else:
                            self.mat_list.append(val_cls_mat_list)
                    else:
                        self.mat_list.append(lst)
                else:
                    self.mat_list.append(
                        cls_mat_list[:self.split_offset % ori_len] +
                        cls_mat_list[(self.split_offset + self.train_len) % ori_len:]
                    )
            else:
                # if sets == 'train' and not cfg.WillowObjectSplit.TRAIN_SAME_AS_TEST:
                #     lst = cls_mat_list[:(self.split_offset + self.train_len) % ori_len - ori_len] + cls_mat_list[self.split_offset % ori_len:]
                #     if self.rate_1 > 0.5:
                #         random.seed(cfg.RANDOM_SEED)
                #         random.shuffle(lst)
                #         self.mat_list.append(lst[:int(len(lst)*self.rate_1)])
                #     else:
                #         random.seed(cfg.RANDOM_SEED)
                #         random.shuffle(lst)
                #         self.mat_list.append(lst[-int(len(lst)*self.rate_1):])
                #         self.sets = 'validation'
                if sets == 'train' and not cfg.WillowObjectSplit.TRAIN_SAME_AS_TEST:
                    lst = cls_mat_list[:(self.split_offset + self.train_len) % ori_len - ori_len] + cls_mat_list[self.split_offset % ori_len:]
                    if self.val_split_size != 0:
                        train_cls_mat_list, val_cls_mat_list = train_test_split(lst, test_size=self.val_split_size, random_state=cfg.RANDOM_SEED)
                        if sets == 'train':
                            self.mat_list.append(train_cls_mat_list)
                        else:
                            self.mat_list.append(val_cls_mat_list)
                    else:
                        self.mat_list.append(lst)
                else:
                    self.mat_list.append(
                        cls_mat_list[
                        (self.split_offset + self.train_len) % ori_len - ori_len: self.split_offset % ori_len]
                    )

    def get_ssl_pair(self, cls=None, shuffle=True, tgt_outlier=True, src_outlier=True):
        """
        Randomly get one object from WILLOW-object dataset, and then argument it two times to get a pair
        :param cls: None for random class, or specify for a certain set
        :param shuffle: random shuffle the keypoints
        :param double_argumentation:
        :return: (pair of data, groundtruth permutation matrix)
        """
        if cls is None:
            cls = random.randrange(0, len(self.classes))
        elif type(cls) == str:
            cls = self.classes.index(cls)
        assert type(cls) == int and 0 <= cls < len(self.classes)

        anno_pair = []
        for mat_name in random.sample(self.mat_list[cls], 1):
            anno_dict = self.__get_anno_dict(mat_name, cls)
            if shuffle:
                random.shuffle(anno_dict['keypoints'])
            anno_pair.append(anno_dict)

        # * control whether to enable image augmentations
        if cfg.SSL.IMAGE_AUGMENTATION:
            for n in range(2 if cfg.SSL.DOUBLE else 1):
                ps = anno_pair[0]['keypoints']
                pset = []
                for i in range(len(ps)):
                    pset.append((ps[i]['x'], ps[i]['y']))
                im = anno_pair[0]['image']
                trans, qset, coord = augmentation(im, pset, cfg.SSL.CROP_RATE_LB, cfg.SSL.CROP_RATE_UB,
                                                cfg.SSL.SCALE_RATIO_LB, cfg.SSL.SCALE_RATIO_UB,
                                                cfg.SSL.VERTICAL_FLIP_RATE, cfg.SSL.HORIZONTAL_FLIP_RATE,
                                                cfg.SSL.COLOR_JITTER, cfg.SSL.COLOR_JITTER_RATE, cfg.SSL.GRAY_SCALE,
                                                cfg.SSL.GAUSSIAN_BLUR_RATE, cfg.SSL.GAUSSIAN_BLUR_SIGMA)
                new_dict = copy.deepcopy(anno_pair[0])
                new_dict['image'] = trans
                qs = []
                for i in range(len(qset)):
                    if qset[i] is not None:
                        qs.append({'name': ps[i]['name'], 'x': qset[i][0], 'y': qset[i][1]})
                    elif random.random() < cfg.SSL.PADDING_RATE:
                        qs.append({'name': 'outlier', 'x': random.random() * im.size[0], 'y': random.random() * im.size[1]})
                while len(qs) <= 9:
                    qs.append({'name': 'outlier', 'x': random.random() * im.size[0], 'y': random.random() * im.size[1]})
                if shuffle:
                    random.shuffle(qs)
                new_dict['keypoints'] = qs
                anno_pair.append(new_dict)
            if len(anno_pair) > 2:
                anno_pair.pop(0)
        else:
            new_dict = copy.deepcopy(anno_pair[0])
            if shuffle:
                index = torch.randperm(len(anno_pair[0]['keypoints']))
            else:
                index = torch.arange(len(anno_pair[0]['keypoints']))
            new_dict['index'] = index
            new_dict['keypoints'] = [{'name': new_dict['keypoints'][i]['name'],
                                      'x': new_dict['keypoints'][i]['x'], 
                                      'y': new_dict['keypoints'][i]['y']} for i in index]
            anno_pair.append(new_dict)
            
        perm_mat = np.zeros([len(_['keypoints']) for _ in anno_pair], dtype=np.float32)
        row_list = []
        col_list = []
        for i, keypoint in enumerate(anno_pair[0]['keypoints']):
            for j, _keypoint in enumerate(anno_pair[1]['keypoints']):
                if keypoint['name'] == _keypoint['name'] and keypoint['name'] != 'outlier':
                    perm_mat[i, j] = 1
                    row_list.append(i)
                    col_list.append(j)
                    break
        row_list.sort()
        col_list.sort()
        # if not src_outlier:
        #     perm_mat = perm_mat[row_list, :]
        #     anno_pair[0]['keypoints'] = [anno_pair[0]['keypoints'][i] for i in row_list]
        # if not tgt_outlier:
        #     perm_mat = perm_mat[:, col_list]
        #     anno_pair[1]['keypoints'] = [anno_pair[1]['keypoints'][j] for j in col_list]

        return anno_pair, perm_mat

    def get_pair(self, cls=None, problem=cfg.PROBLEM.TYPE, shuffle=True, tgt_outlier=False, src_outlier=False):
        """
        Randomly get a pair of objects from WILLOW-object dataset
        :param cls: None for random class, or specify for a certain set
        :param shuffle: random shuffle the keypoints
        :return: (pair of data, groundtruth permutation matrix)
        """
        if self.sets == "train" and cfg.PROBLEM.TYPE == 'GCL':
            return self.get_ssl_pair(cls, shuffle)
        if cls is None:
            cls = random.randrange(0, len(self.classes))
        elif type(cls) == str:
            cls = self.classes.index(cls)
        assert type(cls) == int and 0 <= cls < len(self.classes)

        anno_pair = []
        for mat_name in random.sample(self.mat_list[cls][0: int(len(self.mat_list[cls]) * self.rate_2)], 2):
            anno_dict = self.__get_anno_dict(mat_name, cls)
            if shuffle:
                random.shuffle(anno_dict['keypoints'])
            anno_pair.append(anno_dict)

        perm_mat = np.zeros([len(_['keypoints']) for _ in anno_pair], dtype=np.float32)
        row_list = []
        col_list = []
        for i, keypoint in enumerate(anno_pair[0]['keypoints']):
            for j, _keypoint in enumerate(anno_pair[1]['keypoints']):
                if keypoint['name'] == _keypoint['name']:
                    if keypoint['name'] != 'outlier':
                        perm_mat[i, j] = 1
                    row_list.append(i)
                    col_list.append(j)
                    break
        row_list.sort()
        col_list.sort()
        perm_mat = perm_mat[row_list, :]
        perm_mat = perm_mat[:, col_list]
        anno_pair[0]['keypoints'] = [anno_pair[0]['keypoints'][i] for i in row_list]
        anno_pair[1]['keypoints'] = [anno_pair[1]['keypoints'][j] for j in col_list]

        return anno_pair, perm_mat

    def get_multi(self, cls=None, num=2, shuffle=True):
        """
        Randomly get multiple objects from Willow Object Class dataset for multi-matching.
        :param cls: None for random class, or specify for a certain set
        :param num: number of objects to be fetched
        :param shuffle: random shuffle the keypoints
        :return: (list of data, list of permutation matrices)
        """
        if cls is None:
            cls = random.randrange(0, len(self.classes))
        elif type(cls) == str:
            cls = self.classes.index(cls)
        assert type(cls) == int and 0 <= cls < len(self.classes)

        anno_list = []
        for mat_name in random.sample(self.mat_list[cls], num):
            anno_dict = self.__get_anno_dict(mat_name, cls)
            if shuffle:
                random.shuffle(anno_dict['keypoints'])
            anno_list.append(anno_dict)

        perm_mat = [np.zeros([len(anno_list[0]['keypoints']), len(x['keypoints'])], dtype=np.float32) for x in
                    anno_list]
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
                    if keypoint['name'] != 'outlier':
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

    def __get_anno_dict(self, mat_file, cls):
        """
        Get an annotation dict from .mat annotation
        """
        assert mat_file.exists(), '{} does not exist.'.format(mat_file)

        img_name = mat_file.stem + '.png'
        img_file = mat_file.parent / img_name

        struct = sio.loadmat(mat_file.open('rb'))
        kpts = struct['pts_coord']

        with Image.open(str(img_file)) as img:
            ori_sizes = img.size
            obj = img.resize(self.obj_resize, resample=Image.BICUBIC)
            xmin = 0
            ymin = 0
            w = ori_sizes[0]
            h = ori_sizes[1]

        keypoint_list = []
        for idx, keypoint in enumerate(np.split(kpts, kpts.shape[1], axis=1)):
            attr = {
                'name': idx,
                'x': float(keypoint[0]) * self.obj_resize[0] / w,
                'y': float(keypoint[1]) * self.obj_resize[1] / h
            }
            keypoint_list.append(attr)

        for idx in range(self.rand_outlier):
            attr = {
                'name': 'outlier',
                'x': random.uniform(0, self.obj_resize[0]),
                'y': random.uniform(0, self.obj_resize[1])
            }
            keypoint_list.append(attr)

        anno_dict = dict()
        anno_dict['image'] = obj
        anno_dict['keypoints'] = keypoint_list
        anno_dict['bounds'] = xmin, ymin, w, h
        anno_dict['ori_sizes'] = ori_sizes
        anno_dict['cls'] = cls
        anno_dict['univ_size'] = 10

        return anno_dict

    def len(self, cls):
        if type(cls) == int:
            cls = self.classes[cls]
        assert cls in self.classes
        return len(self.mat_list[self.classes.index(cls)])


if __name__ == '__main__':
    cfg.WillowObjectSplit.ROOT_DIR = 'WILLOW-ObjectClass'
    cfg.WillowObjectSplit.SPLIT_OFFSET = 0
    train = WillowObjectSplit('train', (256, 256))
    test = WillowObjectSplit('test', (256, 256))
    for train_cls_list, test_cls_list in zip(train.mat_list, test.mat_list):
        for t in train_cls_list:
            assert t not in test_cls_list
    pass
