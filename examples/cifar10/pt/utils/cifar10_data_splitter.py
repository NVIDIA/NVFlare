# Copyright (c) 2022, NVIDIA CORPORATION.
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

# This Dirichlet sampling strategy for creating a heterogeneous partition is adopted
# from FedMA (https://github.com/IBM/FedMA).

# MIT License

# Copyright (c) 2020 International Business Machines

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import json
import os

import numpy as np
import torchvision.datasets as datasets

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.event_type import EventType

CIFAR10_ROOT = "/tmp/cifar10"   # will be used for all CIFAR-10 experiments


def _get_site_class_summary(train_label, site_idx):
    class_sum = {}

    for site, data_idx in site_idx.items():
        unq, unq_cnt = np.unique(train_label[data_idx], return_counts=True)
        tmp = {int(unq[i]): int(unq_cnt[i]) for i in range(len(unq))}
        class_sum[site] = tmp
    return class_sum


class Cifar10DataSplitter(FLComponent):
    def __init__(self,
                 split_dir: str = None,
                 num_sites: int = 8,
                 alpha: float = 0.5,
                 seed: int = 0
                 ):
        super().__init__()
        self.split_dir = split_dir
        self.num_sites = num_sites
        self.alpha = alpha
        self.seed = seed

        if alpha < 0.:
            raise ValueError(f"Alpha should be larger 0.0 but was {alpha}!")

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.split(fl_ctx)

    def split(self, fl_ctx: FLContext):
        np.random.seed(self.seed)

        self.log_info(fl_ctx, f"Partition CIFAR-10 dataset into {self.num_sites} sites with Dirichlet sampling under alpha {self.alpha}")
        site_idx, class_sum = self._partition_data()

        # write to files
        if self.split_dir is None:
            raise ValueError("You need to define a valid `split_dir` when splitting the data.")
        if not os.path.isdir(self.split_dir):
            os.makedirs(self.split_dir)
        sum_file_name = os.path.join(self.split_dir, "summary.txt")
        with open(sum_file_name, "w") as sum_file:
            sum_file.write(f"Number of clients: {self.num_sites} \n")
            sum_file.write(f"Dirichlet sampling parameter: {self.alpha} \n")
            sum_file.write("Class counts for each client: \n")
            sum_file.write(json.dumps(class_sum))

        site_file_path = os.path.join(self.split_dir, "site-")
        for site in range(self.num_sites):
            site_file_name = site_file_path + str(site + 1) + ".npy"
            np.save(site_file_name, np.array(site_idx[site]))

    def load_cifar10_data(self):
        # download data
        train_dataset = datasets.CIFAR10(root=CIFAR10_ROOT, train=True, download=True)

        # only training label is needed for doing split
        train_label = np.array(train_dataset.targets)
        return train_label

    def _partition_data(self):
        train_label = self.load_cifar10_data()

        min_size = 0
        K = 10
        N = train_label.shape[0]
        site_idx = {}

        # split
        while min_size < 10:
            idx_batch = [[] for _ in range(self.num_sites)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(train_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_sites))
                # Balance
                proportions = np.array([p * (len(idx_j) < N / self.num_sites) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        # shuffle
        for j in range(self.num_sites):
            np.random.shuffle(idx_batch[j])
            site_idx[j] = idx_batch[j]

        # collect class summary
        class_sum = _get_site_class_summary(train_label, site_idx)

        return site_idx, class_sum
