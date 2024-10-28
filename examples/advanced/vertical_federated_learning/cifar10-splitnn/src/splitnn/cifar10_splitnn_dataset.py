# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
import torch
from torchvision import datasets


class CIFAR10SplitNN(object):  # TODO: use torch.utils.data.Dataset with batch sampling
    def __init__(self, root, train=True, transform=None, download=False, returns="all", intersect_idx=None):
        """CIFAR-10 dataset with index to extract a mini-batch based on given batch indices
        Useful for SplitNN training

        Args:
            root: data root
            data_idx: to specify the data for a particular client site.
                If index provided, extract subset, otherwise use the whole set
            train: whether to use the training or validation split (default: True)
            transform: image transforms
            download: whether to download the data (default: False)
            returns: specify which data the client has
            intersect_idx: indices of samples intersecting between both
                participating sites. Intersection indices will be sorted to
                ensure that data is aligned on both sites.
        Returns:
            A PyTorch dataset
        """
        self.root = root
        self.train = train
        self.transform = transform
        self.download = download
        self.returns = returns
        self.intersect_idx = intersect_idx
        self.orig_size = 0

        if self.intersect_idx is not None:
            self.intersect_idx = np.sort(self.intersect_idx).astype(np.int64)

        self.data, self.target = self.__build_cifar_subset__()

    def __build_cifar_subset__(self):
        # if intersect index provided, extract subset, otherwise use the whole
        # set
        cifar_dataobj = datasets.CIFAR10(self.root, self.train, self.transform, self.download)
        data = cifar_dataobj.data
        target = np.array(cifar_dataobj.targets)
        self.orig_size = len(data)
        if self.intersect_idx is not None:
            data = data[self.intersect_idx]
            target = target[self.intersect_idx]
        return data, target

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    # TODO: this can probably made more efficient using batch_sampler
    def get_batch(self, batch_indices):
        img_batch = []
        target_batch = []
        for idx in batch_indices:
            img, target = self.__getitem__(idx)
            img_batch.append(img)
            target_batch.append(torch.tensor(target, dtype=torch.long))
        img_batch = torch.stack(img_batch, dim=0)
        target_batch = torch.stack(target_batch, dim=0)
        if self.returns == "all":
            return img_batch, target_batch
        elif self.returns == "image":
            return img_batch
        elif self.returns == "label":
            return target_batch
        else:
            raise ValueError(f"Expected `returns` to be 'all', 'image', or 'label', but got '{self.returns}'")

    def __len__(self):
        return len(self.data)
