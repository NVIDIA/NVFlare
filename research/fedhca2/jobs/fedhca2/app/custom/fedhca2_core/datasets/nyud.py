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

import os

import numpy as np
import torch
from PIL import Image

from .utils.mypath import MyPath


class NYUD(torch.utils.data.Dataset):

    def __init__(self, root=MyPath.db_root_dir('NYUDv2'), train=True, tasks=None, transform=None, dataidxs=None):
        """
        Initialize the NYUDv2 dataset
        :param str root: Root directory of dataset
        :param bool train: True for training set, False for validation set
        :param list tasks: Tasks to be loaded
        :param Compose transform: Data augmentation
        :param list dataidxs: Indexes of the data to be loaded (for small dataset loading, default is None for all data)
        """
        self.root = root
        self.transform = transform
        self.dataidxs = dataidxs

        if not os.path.exists(self.root):
            raise RuntimeError('Dataset not found!')

        # Original Images
        self.images = []
        images_dir = os.path.join(self.root, 'images')

        # Edge Detection
        self.do_edge = 'edge' in tasks
        self.edges = []
        edge_gt_dir = os.path.join(root, 'edge')

        # Semantic Segmentation
        self.do_semseg = 'semseg' in tasks
        self.semsegs = []
        semseg_gt_dir = os.path.join(root, 'segmentation')

        # Surface Normals Estimation
        self.do_normals = 'normals' in tasks
        self.normals = []
        normal_gt_dir = os.path.join(root, 'normals')

        # Depth Estimation
        self.do_depth = 'depth' in tasks
        self.depths = []
        depth_gt_dir = os.path.join(root, 'depth')

        # Separation of training set and validation set
        splits_dir = os.path.join(self.root, 'gt_sets')
        if train:
            split_f = os.path.join(splits_dir, 'train.txt')
        else:
            split_f = os.path.join(splits_dir, 'val.txt')

        with open(split_f, "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.im_ids = file_names
        for x in file_names:
            _image = os.path.join(images_dir, x + '.png')
            assert os.path.isfile(_image)
            self.images.append(_image)

            _edge = os.path.join(self.root, edge_gt_dir, x + '.png')
            assert os.path.isfile(_edge)
            self.edges.append(_edge)

            _semseg = os.path.join(self.root, semseg_gt_dir, x + '.png')
            assert os.path.isfile(_semseg)
            self.semsegs.append(_semseg)

            _normal = os.path.join(self.root, normal_gt_dir, x + '.png')
            assert os.path.isfile(_normal)
            self.normals.append(_normal)

            _depth = os.path.join(self.root, depth_gt_dir, x + '.npy')
            assert os.path.isfile(_depth)
            self.depths.append(_depth)

        if self.do_edge:
            assert len(self.images) == len(self.edges)
        if self.do_semseg:
            assert len(self.images) == len(self.semsegs)
        if self.do_normals:
            assert len(self.images) == len(self.normals)
        if self.do_depth:
            assert len(self.images) == len(self.depths)

        self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        if self.dataidxs is not None:
            self.images = [self.images[idx] for idx in self.dataidxs]
            self.edges = [self.edges[idx] for idx in self.dataidxs if self.do_edge]
            self.semsegs = [self.semsegs[idx] for idx in self.dataidxs if self.do_semseg]
            self.normals = [self.normals[idx] for idx in self.dataidxs if self.do_normals]
            self.depths = [self.depths[idx] for idx in self.dataidxs if self.do_depth]

    def __getitem__(self, index):
        sample = {}

        _img = self._load_img(index)
        sample['image'] = _img

        if self.do_edge:
            _edge = self._load_edge(index)
            assert _img.shape[0:2] == _edge.shape[0:2]
            sample['edge'] = np.expand_dims(_edge, axis=-1)

        if self.do_semseg:
            _semseg = self._load_semseg(index)
            assert _img.shape[0:2] == _semseg.shape[0:2]
            sample['semseg'] = np.expand_dims(_semseg, axis=-1)

        if self.do_normals:
            _normals = self._load_normals(index)
            assert _img.shape[0:2] == _normals.shape[0:2]
            sample['normals'] = _normals

        if self.do_depth:
            _depth = self._load_depth(index)
            assert _img.shape[0:2] == _depth.shape[0:2]
            sample['depth'] = np.expand_dims(_depth, axis=-1)

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
        _edge = np.asarray(Image.open(self.edges[index]), dtype=np.float32) / 255.0
        return _edge

    def _load_semseg(self, index):
        _semseg = np.asarray(Image.open(self.semsegs[index]), dtype=np.float32) - 1
        _semseg[_semseg == -1] = 255
        return _semseg

    def _load_normals(self, index):
        _tmp = np.asarray(Image.open(self.normals[index]), dtype=np.float32)
        _normals = 2.0 * _tmp / 255.0 - 1.0  # [0,255] => [-1，1]

        return _normals

    def _load_depth(self, index):
        _depth = np.load(self.depths[index]).astype(np.float32)
        return _depth
