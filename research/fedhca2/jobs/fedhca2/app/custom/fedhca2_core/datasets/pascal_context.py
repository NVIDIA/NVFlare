# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import cv2
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from skimage.morphology import thin

from .utils.mypath import MyPath


class PASCALContext(torch.utils.data.Dataset):
    HUMAN_PART = {
        'hair': 1,
        'head': 1,
        'lear': 1,
        'lebrow': 1,
        'leye': 1,
        'lfoot': 6,
        'lhand': 4,
        'llarm': 4,
        'llleg': 6,
        'luarm': 3,
        'luleg': 5,
        'mouth': 1,
        'neck': 2,
        'nose': 1,
        'rear': 1,
        'rebrow': 1,
        'reye': 1,
        'rfoot': 6,
        'rhand': 4,
        'rlarm': 4,
        'rlleg': 6,
        'ruarm': 3,
        'ruleg': 5,
        'torso': 2,
    }

    def __init__(self, root=MyPath.db_root_dir('PASCALContext'), train=True, tasks=None, transform=None, dataidxs=None):
        """
        Initialize the PASCALContext dataset
        :param str root: Root directory of dataset
        :param bool train: True for training set, False for validation set
        :param list tasks: Tasks to be loaded
        :param Compose transform: Data augmentation
        :param list dataidxs: Indexes of the data to be loaded (for small dataset loading, default is None for all data)
        """
        self.root = root
        self.transform = transform
        self.dataidxs = dataidxs
        self.area_thres = 0

        if not os.path.exists(self.root):
            raise RuntimeError('Dataset Not Found!')

        images_dir = os.path.join(self.root, 'JPEGImages')

        # Edge Detection
        self.do_edge = 'edge' in tasks
        self.edges = []
        edge_gt_dir = os.path.join(self.root, 'pascal-context', 'trainval')
        self.edge_gt_dir = edge_gt_dir

        # Semantic Segmentation
        self.do_semseg = 'semseg' in tasks
        self.semsegs = []

        # Human Part Segmentation
        self.do_human_parts = 'human_parts' in tasks
        part_gt_dir = os.path.join(self.root, 'human_parts')
        self.cat_part = json.load(open(os.path.join(self.root, 'json/pascal_part.json'), 'r'))
        self.human_parts_category = 15  # The category represents human parts
        self.cat_part["15"] = self.HUMAN_PART  # Rules of parts segmentation of human
        # Preprocessed file, implies which image has human object
        if train:
            self.parts_file = os.path.join(self.root, 'ImageSets/Parts/train.txt')
        else:
            self.parts_file = os.path.join(self.root, 'ImageSets/Parts/val.txt')
        self.parts = []

        # Saliency Detection
        self.do_sal = 'sal' in tasks
        sal_gt_dir = os.path.join(self.root, 'sal_distill')
        self.sals = []

        # Surface Normal Estimation
        self.do_normals = 'normals' in tasks
        normal_gt_dir = os.path.join(self.root, 'normals_distill')
        self.normals = []
        if self.do_normals:
            # Find common classes between the two datasets to use for normals (Dataset setting)
            with open(os.path.join(self.root, 'json/nyu_classes.json')) as f:
                cls_nyu = json.load(f)
            with open(os.path.join(self.root, 'json/context_classes.json')) as f:
                cls_context = json.load(f)

            self.normals_valid_classes = []
            for cl_nyu in cls_nyu:
                if cl_nyu in cls_context and cl_nyu != 'unknown':
                    self.normals_valid_classes.append(cls_context[cl_nyu])
            # Custom additions due to incompatibilities
            self.normals_valid_classes.append(cls_context['tvmonitor'])

        # Separation of training set and validation set
        splits_dir = os.path.join(self.root, 'ImageSets/Context')
        if train:
            split_f = os.path.join(splits_dir, 'train.txt')
        else:
            split_f = os.path.join(splits_dir, 'val.txt')

        with open(split_f, "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.im_ids = file_names
        self.images = []
        for x in file_names:
            _image = os.path.join(images_dir, x + ".jpg")
            assert os.path.isfile(_image)
            self.images.append(_image)

            _edge = os.path.join(edge_gt_dir, x + ".mat")
            assert os.path.isfile(_edge)
            self.edges.append(_edge)

            _semseg = self._get_semseg_fname(x)
            assert os.path.isfile(_semseg)
            self.semsegs.append(_semseg)

            _human_part = os.path.join(self.root, part_gt_dir, x + ".mat")
            assert os.path.isfile(_human_part)
            self.parts.append(_human_part)

            _sal = os.path.join(self.root, sal_gt_dir, x + ".png")
            assert os.path.isfile(_sal)
            self.sals.append(_sal)

            _normal = os.path.join(self.root, normal_gt_dir, x + ".png")
            assert os.path.isfile(_normal)
            self.normals.append(_normal)

        if self.do_edge:
            assert len(self.images) == len(self.edges)
        if self.do_semseg:
            assert len(self.images) == len(self.semsegs)
        if self.do_human_parts:
            assert len(self.images) == len(self.parts)
        if self.do_sal:
            assert len(self.images) == len(self.sals)
        if self.do_normals:
            assert len(self.images) == len(self.normals)

        # Preprocess for human parts
        if self.do_human_parts:
            if not self._check_preprocess_parts():
                print("Preprocessing PASCAL dataset for human parts, this will take long, but will be done only once.")
                self._preprocess_parts()
            # Find images which have human parts
            self.has_human_parts = []
            for i in range(len(self.im_ids)):
                # part_obj_dict implies parts object category contained in image labels
                if self.human_parts_category in self.part_obj_dict[self.im_ids[i]]:
                    self.has_human_parts.append(1)
                else:
                    self.has_human_parts.append(0)

        self.__build_truncated_dataset__()

        if self.do_human_parts:
            # If the other tasks are disabled, select only the images that contain human parts
            if not self.do_edge and not self.do_semseg and not self.do_sal and not self.do_normals:
                print("Ignoring images that do not contain human parts.")
                for i in range(len(self.parts) - 1, -1, -1):
                    if self.has_human_parts[i] == 0:
                        del self.im_ids[i]
                        del self.images[i]
                        del self.parts[i]
                        del self.has_human_parts[i]
                assert len(self.images) == len(self.parts)
                print("Number of images with human parts: %d" % (len(self.parts)))

    def __build_truncated_dataset__(self):
        if self.dataidxs is not None:
            self.images = [self.images[idx] for idx in self.dataidxs]
            self.im_ids = [self.im_ids[idx] for idx in self.dataidxs]
            self.edges = [self.edges[idx] for idx in self.dataidxs if self.do_edge]
            self.semsegs = [self.semsegs[idx] for idx in self.dataidxs if self.do_semseg]
            self.parts = [self.parts[idx] for idx in self.dataidxs if self.do_human_parts]
            self.sals = [self.sals[idx] for idx in self.dataidxs if self.do_sal]
            self.normals = [self.normals[idx] for idx in self.dataidxs if self.do_normals]

    def __getitem__(self, index):
        sample = {}

        _img = self._load_img(index)
        sample['image'] = _img

        # Load data, make sure the size of images equals the size of labels
        if self.do_edge:
            _edge = self._load_edge(index)
            assert _edge.shape == _img.shape[:2]
            sample['edge'] = np.expand_dims(_edge, -1)

        if self.do_semseg:
            _semseg = self._load_semseg(index)
            assert _semseg.shape == _img.shape[:2]
            sample['semseg'] = np.expand_dims(_semseg, -1)

        if self.do_human_parts:
            _human_parts = self._load_human_parts(index)
            assert _human_parts.shape == _img.shape[:2]
            sample['human_parts'] = np.expand_dims(_human_parts, -1)

        if self.do_sal:
            _sal = self._load_sal(index)
            assert _sal.shape == _img.shape[:2]
            sample['sal'] = np.expand_dims(_sal, -1)

        if self.do_normals:
            _normals = self._load_normals(index)
            assert _normals.shape[:2] == _img.shape[:2]
            sample['normals'] = _normals

        # Make transforms and augmentations
        if self.transform is not None:
            sample = self.transform(sample)

        sample['meta'] = {'file_name': str(self.im_ids[index]), 'size': (_img.shape[0], _img.shape[1])}
        return sample

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = np.asarray(Image.open(self.images[index]).convert('RGB'), dtype=np.float32)
        return _img

    def _load_edge(self, index):
        _tmp = sio.loadmat(self.edges[index])  # The label of Context
        _edge = cv2.Laplacian(_tmp['LabelMap'], cv2.CV_64F)  # Laplacian, detect edges of different context labels
        # Classify non-zero in _edge as True，then do morphology thin operation
        _edge = thin(np.abs(_edge) > 0).astype(np.float32)
        return _edge

    def _load_semseg(self, index):
        _semseg = np.asarray(Image.open(self.semsegs[index]), dtype=np.float32)
        return _semseg

    def _load_human_parts(self, index):
        if self.has_human_parts[index]:
            # Read Target object
            _part_mat = sio.loadmat(self.parts[index])['anno'][0][0][1][0]
            _target = _inst_mask = None

            for _obj_ii in range(len(_part_mat)):
                has_human = _part_mat[_obj_ii][1][0][0] == self.human_parts_category
                has_parts = len(_part_mat[_obj_ii][3]) != 0

                if has_human and has_parts:
                    if _inst_mask is None:  # The first human in the image
                        _inst_mask = _part_mat[_obj_ii][2].astype(np.float32)  # Mask of positions contains human
                        _target = np.zeros(_inst_mask.shape)
                    else:
                        # If _inst_mask is not None, means there are more than one human in the image
                        # Take union of the humans
                        _inst_mask = np.maximum(_inst_mask, _part_mat[_obj_ii][2].astype(np.float32))

                    n_parts = len(_part_mat[_obj_ii][3][0])  # Number of parts object
                    for part_i in range(n_parts):
                        cat_part = str(_part_mat[_obj_ii][3][0][part_i][0][0])  # Name of part
                        mask_id = self.cat_part[str(self.human_parts_category)][cat_part]
                        mask = _part_mat[_obj_ii][3][0][part_i][1].astype(bool)  # Position of part
                        _target[mask] = mask_id  # Label of part set as mask_id

            if _target is None:
                shape = np.shape(_part_mat[0][2])
                _target = np.zeros(shape, dtype=np.float32)
            return _target.astype(np.float32)
        else:
            shape = np.shape(sio.loadmat(self.parts[index])['anno'][0][0][1][0][0][2])
            # When the output has no humans, we use output as 255 (equal to loss function ignore index)
            return np.zeros(shape, dtype=np.float32)

    def _load_sal(self, index):
        _sal = np.asarray(Image.open(self.sals[index]), dtype=np.float32) / 255.0  # [0,255] => [0,1]
        _sal = (_sal > 0.5).astype(np.float32)  # Binary classification
        return _sal

    def _load_normals(self, index):
        _tmp = np.asarray(Image.open(self.normals[index]), dtype=np.float32)
        _tmp = 2.0 * _tmp / 255.0 - 1.0  # [0,255] => [-1，1]

        labels = sio.loadmat(os.path.join(self.edge_gt_dir, self.im_ids[index] + '.mat'))
        labels = labels['LabelMap']

        _normals = np.zeros(_tmp.shape, dtype=np.float32)
        for x in np.unique(labels):
            if x in self.normals_valid_classes:  # Normals are valid only in valid class
                _normals[labels == x, :] = _tmp[labels == x, :]
        return _normals

    def _get_semseg_fname(self, fname):
        fname_voc = os.path.join(self.root, 'semseg', 'VOC12', fname + '.png')
        fname_context = os.path.join(self.root, 'semseg', 'pascal-context', fname + '.png')
        if os.path.isfile(fname_voc):
            seg = fname_voc
        elif os.path.isfile(fname_context):
            seg = fname_context
        else:
            raise RuntimeError('Segmentation for im: {} was not found'.format(fname))

        return seg

    def _check_preprocess_parts(self):
        _obj_list_file = self.parts_file
        if not os.path.isfile(_obj_list_file):
            return False
        else:
            self.part_obj_dict = json.load(open(_obj_list_file, 'r'))
            # Check whether all data samples have been preprocessed
            return list(np.sort([str(x) for x in self.part_obj_dict.keys()])) == list(np.sort(self.im_ids))

    def _preprocess_parts(self):
        self.part_obj_dict = {}
        obj_counter = 0
        for i in range(len(self.im_ids)):
            # Read object masks and get number of objects
            if i % 100 == 0:
                print("Processing image: %d" % (i))
            part_mat = sio.loadmat(self.parts[i])
            n_obj = len(part_mat['anno'][0][0][1][0])

            # Get the categories from these objects
            _cat_ids = []
            for j in range(n_obj):
                obj_area = np.sum(part_mat['anno'][0][0][1][0][j][2])
                obj_cat = int(part_mat['anno'][0][0][1][0][j][1])
                if obj_area > self.area_thres:
                    _cat_ids.append(obj_cat)
                else:
                    _cat_ids.append(-1)
                obj_counter += 1

            self.part_obj_dict[self.im_ids[i]] = _cat_ids

        with open(self.parts_file, 'w') as outfile:
            outfile.write('{{\n\t"{:s}": {:s}'.format(self.im_ids[0], json.dumps(self.part_obj_dict[self.im_ids[0]])))
            for i in range(1, len(self.im_ids)):
                outfile.write(
                    ',\n\t"{:s}": {:s}'.format(self.im_ids[i], json.dumps(self.part_obj_dict[self.im_ids[i]]))
                )
            outfile.write('\n}\n')

        print('Preprocessing for parts finished.')
