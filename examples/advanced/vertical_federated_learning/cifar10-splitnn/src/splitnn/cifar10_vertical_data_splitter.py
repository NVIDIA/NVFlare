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

import json
import os

import numpy as np
from pt.utils.cifar10_data_utils import get_site_class_summary, load_cifar10_data

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext


class Cifar10VerticalDataSplitter(FLComponent):
    def __init__(self, split_dir: str = None, overlap: int = 10_000, seed: int = 0):
        super().__init__()
        self.split_dir = split_dir
        self.overlap = overlap
        self.seed = seed

        if self.split_dir is None:
            raise ValueError("You need to define a valid `split_dir` when splitting the data.")
        if overlap <= 0:
            raise ValueError(f"Alpha should be larger 0 but was {overlap}!")

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.split(fl_ctx)

    def split(self, fl_ctx: FLContext):
        np.random.seed(self.seed)

        self.log_info(fl_ctx, f"Partition CIFAR-10 dataset into vertically with {self.overlap} overlapping samples.")
        site_idx, class_sum = self._split_data()

        # write to files
        if not os.path.isdir(self.split_dir):
            os.makedirs(self.split_dir)
        sum_file_name = os.path.join(self.split_dir, "summary.txt")
        with open(sum_file_name, "w") as sum_file:
            sum_file.write("Class counts for overlap: \n")
            sum_file.write(json.dumps(class_sum))

        for _site, _idx in site_idx.items():
            site_file_name = os.path.join(self.split_dir, f"{_site}.npy")
            self.log_info(fl_ctx, f"save {site_file_name}")
            np.save(site_file_name, _idx)

    def _split_data(self):
        train_label = load_cifar10_data()

        n_samples = len(train_label)

        if self.overlap > n_samples:
            raise ValueError(
                f"Chosen overlap of {self.overlap} is larger than " f"train dataset with {n_samples} entries."
            )

        sample_idx = np.arange(0, n_samples)

        overlap_idx = np.random.choice(sample_idx, size=np.int64(self.overlap), replace=False)

        remain_idx = list(set(sample_idx) - set(overlap_idx))

        idx_1 = np.concatenate((overlap_idx, np.array(remain_idx)))
        # adding n_samples to remain_idx of site-2 to make sure no overlap
        # with idx_1
        idx_2 = np.concatenate((overlap_idx, np.array(remain_idx) + n_samples))

        # shuffle indexes again for client sites to simulate real world
        # scenario
        np.random.shuffle(idx_1)
        np.random.shuffle(idx_2)

        site_idx = {"overlap": overlap_idx, "site-1": idx_1, "site-2": idx_2}

        # collect class summary
        class_sum = get_site_class_summary(train_label, {"overlap": overlap_idx})

        return site_idx, class_sum
