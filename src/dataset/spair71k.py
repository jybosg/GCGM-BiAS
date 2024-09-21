import copy
import glob
import json
import os
import pickle
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

VOC2011_KPT_NAMES = {
    'cat': ['L_B_Elbow', 'L_B_Paw', 'L_EarBase', 'L_Eye', 'L_F_Elbow',
            'L_F_Paw', 'Nose', 'R_B_Elbow', 'R_B_Paw', 'R_EarBase', 'R_Eye',
            'R_F_Elbow', 'R_F_Paw', 'TailBase', 'Throat', 'Withers'],
    'bottle': ['L_Base', 'L_Neck', 'L_Shoulder', 'L_Top', 'R_Base', 'R_Neck',
               'R_Shoulder', 'R_Top'],
    'horse': ['L_B_Elbow', 'L_B_Paw', 'L_EarBase', 'L_Eye', 'L_F_Elbow',
              'L_F_Paw', 'Nose', 'R_B_Elbow', 'R_B_Paw', 'R_EarBase', 'R_Eye',
              'R_F_Elbow', 'R_F_Paw', 'TailBase', 'Throat', 'Withers'],
    'motorbike': ['B_WheelCenter', 'B_WheelEnd', 'ExhaustPipeEnd',
                  'F_WheelCenter', 'F_WheelEnd', 'HandleCenter', 'L_HandleTip',
                  'R_HandleTip', 'SeatBase', 'TailLight'],
    'boat': ['Hull_Back_Bot', 'Hull_Back_Top', 'Hull_Front_Bot',
             'Hull_Front_Top', 'Hull_Mid_Left_Bot', 'Hull_Mid_Left_Top',
             'Hull_Mid_Right_Bot', 'Hull_Mid_Right_Top', 'Mast_Top', 'Sail_Left',
             'Sail_Right'],
    'tvmonitor': ['B_Bottom_Left', 'B_Bottom_Right', 'B_Top_Left',
                  'B_Top_Right', 'F_Bottom_Left', 'F_Bottom_Right', 'F_Top_Left',
                  'F_Top_Right'],
    'cow': ['L_B_Elbow', 'L_B_Paw', 'L_EarBase', 'L_Eye', 'L_F_Elbow',
            'L_F_Paw', 'Nose', 'R_B_Elbow', 'R_B_Paw', 'R_EarBase', 'R_Eye',
            'R_F_Elbow', 'R_F_Paw', 'TailBase', 'Throat', 'Withers'],
    'chair': ['BackRest_Top_Left', 'BackRest_Top_Right', 'Leg_Left_Back',
              'Leg_Left_Front', 'Leg_Right_Back', 'Leg_Right_Front',
              'Seat_Left_Back', 'Seat_Left_Front', 'Seat_Right_Back',
              'Seat_Right_Front'],
    'car': ['L_B_RoofTop', 'L_B_WheelCenter', 'L_F_RoofTop', 'L_F_WheelCenter',
            'L_HeadLight', 'L_SideviewMirror', 'L_TailLight', 'R_B_RoofTop',
            'R_B_WheelCenter', 'R_F_RoofTop', 'R_F_WheelCenter', 'R_HeadLight',
            'R_SideviewMirror', 'R_TailLight'],
    'person': ['B_Head', 'HeadBack', 'L_Ankle', 'L_Ear', 'L_Elbow', 'L_Eye',
               'L_Foot', 'L_Hip', 'L_Knee', 'L_Shoulder', 'L_Toes', 'L_Wrist', 'Nose',
               'R_Ankle', 'R_Ear', 'R_Elbow', 'R_Eye', 'R_Foot', 'R_Hip', 'R_Knee',
               'R_Shoulder', 'R_Toes', 'R_Wrist'],
    'diningtable': ['Bot_Left_Back', 'Bot_Left_Front', 'Bot_Right_Back',
                    'Bot_Right_Front', 'Top_Left_Back', 'Top_Left_Front', 'Top_Right_Back',
                    'Top_Right_Front'],
    'dog': ['L_B_Elbow', 'L_B_Paw', 'L_EarBase', 'L_Eye', 'L_F_Elbow',
            'L_F_Paw', 'Nose', 'R_B_Elbow', 'R_B_Paw', 'R_EarBase', 'R_Eye',
            'R_F_Elbow', 'R_F_Paw', 'TailBase', 'Throat', 'Withers'],
    'bird': ['Beak_Base', 'Beak_Tip', 'Left_Eye', 'Left_Wing_Base',
             'Left_Wing_Tip', 'Leg_Center', 'Lower_Neck_Base', 'Right_Eye',
             'Right_Wing_Base', 'Right_Wing_Tip', 'Tail_Tip', 'Upper_Neck_Base'],
    'bicycle': ['B_WheelCenter', 'B_WheelEnd', 'B_WheelIntersection',
                'CranksetCenter', 'F_WheelCenter', 'F_WheelEnd', 'F_WheelIntersection',
                'HandleCenter', 'L_HandleTip', 'R_HandleTip', 'SeatBase'],
    'train': ['Base_Back_Left', 'Base_Back_Right', 'Base_Front_Left',
              'Base_Front_Right', 'Roof_Back_Left', 'Roof_Back_Right',
              'Roof_Front_Middle'],
    'sheep': ['L_B_Elbow', 'L_B_Paw', 'L_EarBase', 'L_Eye', 'L_F_Elbow',
              'L_F_Paw', 'Nose', 'R_B_Elbow', 'R_B_Paw', 'R_EarBase', 'R_Eye',
              'R_F_Elbow', 'R_F_Paw', 'TailBase', 'Throat', 'Withers'],
    'aeroplane': ['Bot_Rudder', 'Bot_Rudder_Front', 'L_Stabilizer',
                  'L_WingTip', 'Left_Engine_Back', 'Left_Engine_Front',
                  'Left_Wing_Base', 'NoseTip', 'Nose_Bottom', 'Nose_Top',
                  'R_Stabilizer', 'R_WingTip', 'Right_Engine_Back',
                  'Right_Engine_Front', 'Right_Wing_Base', 'Top_Rudder'],
    'sofa': ['Back_Base_Left', 'Back_Base_Right', 'Back_Top_Left',
             'Back_Top_Right', 'Front_Base_Left', 'Front_Base_Right',
             'Handle_Front_Left', 'Handle_Front_Right', 'Handle_Left_Junction',
             'Handle_Right_Junction', 'Left_Junction', 'Right_Junction'],
    'pottedplant': ['Bottom_Left', 'Bottom_Right', 'Top_Back_Middle',
                    'Top_Front_Middle', 'Top_Left', 'Top_Right'],
    'bus': ['L_B_Base', 'L_B_RoofTop', 'L_F_Base', 'L_F_RoofTop', 'R_B_Base',
            'R_B_RoofTop', 'R_F_Base', 'R_F_RoofTop']
}


class SPair71k:
    def __init__(self, sets, rate_1, rate_2, obj_resize):
        """
        :param sets: 'train', 'validation', 'test'
        :param obj_resize: resized object size
        """
        super(SPair71k, self).__init__()
        sets_translation_dict = dict(train="trn", validation='val', test="test")
        diff_params_dict = dict(trn=cfg.SPair71k.TRAIN_DIFF_PARAMS, val=cfg.SPair71k.EVAL_DIFF_PARAMS, test=cfg.SPair71k.EVAL_DIFF_PARAMS)
        
        self.sets = sets_translation_dict[sets]
        self.diff_params = diff_params_dict[self.sets]
        self.pair_anno_path = Path(cfg.SPair71k.ROOT_DIR + "/PairAnnotation")
        self.image_anno_path = Path(cfg.SPair71k.ROOT_DIR + "/ImageAnnotation")
        self.image_path = Path(cfg.SPair71k.ROOT_DIR + "/JPEGImages")
        self.layout_path = Path(cfg.SPair71k.ROOT_DIR + "/Layout")
        self.dataset_size = cfg.SPair71k.SIZE
        self.anno_files = open(os.path.join(self.layout_path, self.dataset_size, self.sets + ".txt"), "r").read().split("\n")
        self.anno_files = self.anno_files[: len(self.anno_files) - 1]
        self.classes = list(map(lambda x: os.path.basename(x), glob.glob("%s/*" % self.image_path)))
        self.classes.sort()
        self.obj_resize = obj_resize
        self.comb_cls = cfg.SPair71k.COMB_CLS
        self.anno_files_filtered, self.anno_files_filtered_cls_dict, self.classes = self.filter_annotations(self.anno_files, self.diff_params)
        self.total_size = len(self.anno_files_filtered)
        self.size_by_cls = {cls: len(anno_list) for cls, anno_list in self.anno_files_filtered_cls_dict.items()}

    def filter_annotations(self, anno_files, diff_params):
        if len(diff_params) > 0:
            basepath = os.path.join(self.pair_anno_path, "pickled", self.sets)
            if not os.path.exists(basepath):
                os.makedirs(basepath)
            diff_paramas_str = self.diff_dict_to_str(diff_params)
            try:
                filepath = os.path.join(basepath, diff_paramas_str + ".pickle")
                anno_files_filtered = pickle.load(open(filepath, "rb"))
                print(
                    f"Found filtered annotations for difficulty parameters {diff_params} and {self.sets}-set at {filepath}"
                )
            except (OSError, IOError) as e:
                print(
                    f"No pickled annotations found for difficulty parameters {diff_params} and {self.sets}-set. Filtering..."
                )
                anno_files_filtered_dict = {}

                for anno_file in anno_files:
                    with open(os.path.join(self.pair_anno_path, self.sets, anno_file + ".json")) as f:
                        annotation = json.load(f)
                    diff = {key: annotation[key] for key in self.diff_params.keys()}
                    diff_str = self.diff_dict_to_str(diff)
                    if diff_str in anno_files_filtered_dict:
                        anno_files_filtered_dict[diff_str].append(anno_file)
                    else:
                        anno_files_filtered_dict[diff_str] = [anno_file]
                total_l = 0
                for diff_str, file_list in anno_files_filtered_dict.items():
                    total_l += len(file_list)
                    filepath = os.path.join(basepath, diff_str + ".pickle")
                    pickle.dump(file_list, open(filepath, "wb"))
                assert total_l == len(anno_files)
                print(f"Done filtering. Saved filtered annotations to {basepath}.")
                anno_files_filtered = anno_files_filtered_dict[diff_paramas_str]
        else:
            print(f"No difficulty parameters for {self.sets}-set. Using all available data.")
            anno_files_filtered = anno_files

        anno_files_filtered_cls_dict = {
            cls: list(filter(lambda x: cls in x, anno_files_filtered)) for cls in self.classes
        }
        class_len = {cls: len(anno_list) for cls, anno_list in anno_files_filtered_cls_dict.items()}
        print(f"Number of annotation pairs matching the difficulty params in {self.sets}-set: {class_len}")
        if self.comb_cls:
            cls_name = "combined"
            anno_files_filtered_cls_dict = {cls_name: anno_files_filtered}
            filtered_classes = [cls_name]
            print(f"Combining {self.sets}-set classes. Total of {len(anno_files_filtered)} image pairs used.")
        else:
            filtered_classes = []
            for cls, anno_f in anno_files_filtered_cls_dict.items():
                if len(anno_f) > 0:
                    filtered_classes.append(cls)
                else:
                    print(f"Excluding class {cls} from {self.sets}-set.")
        return anno_files_filtered, anno_files_filtered_cls_dict, filtered_classes

    def diff_dict_to_str(self, diff):
        diff_str = ""
        keys = ["mirror", "viewpoint_variation", "scale_variation", "truncation", "occlusion"]
        for key in keys:
            if key in diff.keys():
                diff_str += key
                diff_str += str(diff[key])
        return diff_str

    def get_ssl_pair(self, cls=None, shuffle=True, tgt_outlier=False, src_outlier=False):
        
        if cls is None:
            cls = self.classes[random.randrange(0, len(self.classes))]
            anno_files = self.anno_files_filtered_cls_dict[cls]
        elif type(cls) == int:
            cls = self.classes[cls]
            anno_files = self.anno_files_filtered_cls_dict[cls]
        else:
            assert type(cls) == str
            anno_files = self.anno_files_filtered_cls_dict[cls]

        assert len(anno_files) > 0
        anno_file = random.choice(anno_files) + ".json"
        with open(os.path.join(self.pair_anno_path, self.sets, anno_file)) as f:
            annotation = json.load(f)

        category = annotation["category"]
        if cls is not None and not self.comb_cls:
            assert cls == category
        assert all(annotation[key] == value for key, value in self.diff_params.items())
        
        # * anno_file
        anno_file = self.image_anno_path / category / random.choice([annotation["src_imname"][:-4] + '.json', annotation["trg_imname"][:-4] + '.json'])
        # * annot_dict
        anno_dict = self.__get_anno_dict(anno_file, category)
        if shuffle:
            random.shuffle(anno_dict['keypoints'])
        anno_pair = [anno_dict]
        
        # * control whether to enable image augmentations
        if cfg.SSL.IMAGE_AUGMENTATION:
            # cfg.SSL.DOUBLE = False
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
                # while all([q is None for q in qset]):
                #     trans, qset, coord = argumentation(im, pset)
                new_dict = copy.deepcopy(anno_pair[0])
                new_dict['image'] = trans
                qs = []
                for i in range(len(qset)):
                    if qset[i] is not None:
                        qs.append({'name': ps[i]['name'], 'x': qset[i][0], 'y': qset[i][1]})
                    elif random.random() < cfg.SSL.PADDING_RATE:
                        qs.append({'name': 'outlier', 'x': random.random() * im.size[0], 'y': random.random() * im.size[1]})
                while len(qs) <= 2:
                    qs.append({'name': 'outlier', 'x': random.random() * im.size[0], 'y': random.random() * im.size[1]})
                if shuffle:
                    random.shuffle(qs)
                new_dict['keypoints'] = qs
                anno_pair.append(new_dict)
            if len(anno_pair) > 2:
                anno_pair.pop(0)
            # anno_pair.reverse()
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
                if keypoint['name'] == _keypoint['name']:
                    if keypoint['name'] != 'outlier':
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
        Randomly get a pair of objects from SPair dataset
        :param cls: None for random class, or specify for a certain set
        :param shuffle: random shuffle the keypoints
        :param src_outlier: allow outlier in the source graph (first graph)
        :param tgt_outlier: allow outlier in the target graph (second graph)
        :return: (pair of data, groundtruth permutation matrix)
        """
        # * 'trn' instead of 'train'
        if self.sets == "trn" and problem == 'GCL':
            return self.get_ssl_pair(cls, shuffle)
        if cls is None:
            cls = self.classes[random.randrange(0, len(self.classes))]
            anno_files = self.anno_files_filtered_cls_dict[cls]
        elif type(cls) == int:
            cls = self.classes[cls]
            anno_files = self.anno_files_filtered_cls_dict[cls]
        else:
            assert type(cls) == str
            anno_files = self.anno_files_filtered_cls_dict[cls]

        # get pre-processed images
        assert len(anno_files) > 0
        anno_file = random.choice(anno_files) + ".json"
        with open(os.path.join(self.pair_anno_path, self.sets, anno_file)) as f:
            annotation = json.load(f)

        category = annotation["category"]
        if cls is not None and not self.comb_cls:
            assert cls == category
        assert all(annotation[key] == value for key, value in self.diff_params.items())
        
        # * anno_file
        src_anno_file = self.image_anno_path / category / str(annotation["src_imname"][:-4] + '.json')
        tgt_anno_file = self.image_anno_path / category / str(annotation["trg_imname"][:-4] + '.json')
        # * annot_dict
        src_anno_dict = self.__get_anno_dict(src_anno_file, category)
        tgt_anno_dict = self.__get_anno_dict(tgt_anno_file, category)
        if shuffle:
            random.shuffle(src_anno_dict['keypoints'])
            random.shuffle(tgt_anno_dict['keypoints'])
        anno_pair = [src_anno_dict, tgt_anno_dict]

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
        if not src_outlier:
            perm_mat = perm_mat[row_list, :]
            anno_pair[0]['keypoints'] = [anno_pair[0]['keypoints'][i] for i in row_list]
        if not tgt_outlier:
            perm_mat = perm_mat[:, col_list]
            anno_pair[1]['keypoints'] = [anno_pair[1]['keypoints'][j] for j in col_list]

        return anno_pair, perm_mat
    
    def __get_anno_dict(self, anno_file, cls):
        assert anno_file.exists(), '{} does not exist.'.format(anno_file)
        img_file = self.image_path / cls / str(anno_file.stem + '.jpg')
        assert img_file.exists(), '{} does not exist.'.format(img_file)
        
        with Image.open(img_file) as img:
            ori_sizes = img.size
            obj = img.resize(self.obj_resize, resample=Image.BICUBIC)

        with open(anno_file) as f:
            annotations = json.load(f)
            h = float(annotations['image_height'])
            w = float(annotations['image_width'])

        keypoint_list = []
        for key, value in annotations['kps'].items():
            if not value == None:
                x = value[0] * self.obj_resize[0] / w
                y = value[1] * self.obj_resize[1] / h
                kpts_anno = dict()
                kpts_anno['name'] = key
                kpts_anno['x'] = x
                kpts_anno['y'] = y
                keypoint_list.append(kpts_anno)

        anno_dict = dict()
        anno_dict['image'] = obj
        anno_dict['keypoints'] = keypoint_list
        anno_dict['path'] = img_file
        anno_dict['cls'] = cls
        anno_dict['bounds'] = annotations['bndbox']
        anno_dict['univ_size'] = len(VOC2011_KPT_NAMES[cls])
        return anno_dict