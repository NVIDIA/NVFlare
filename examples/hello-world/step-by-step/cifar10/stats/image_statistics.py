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
from typing import Dict, List

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.statistics_spec import Bin, DataType, Feature, Histogram, HistogramType, Statistics

# the dataset path
CIFAR10_ROOT = "/tmp/nvflare/data/cifar10"


class ImageStatistics(Statistics):
    def __init__(self, data_root: str = CIFAR10_ROOT, batch_size: int = 4):
        """local image intensity calculator.

        Args:
            data_root: directory with local image data.
         Returns:
            Histogram of local statistics`
        """
        super().__init__()
        self.dataset_path = data_root
        self.batch_size = batch_size

        # there are three color channels : RGB, each corresponding to each channel index
        # we are going treat each channel as one feature, the feature Ids are corresponding to tensor channel index.
        # The feature name is named "red", "gree", "blue" (RGB).

        self.features_ids = {"red": 0, "green": 1, "blue": 2}
        self.image_features = [
            Feature("red", DataType.FLOAT),
            Feature("green", DataType.FLOAT),
            Feature("blue", DataType.FLOAT),
        ]

        self.dataset_lengths = {}
        self.loaders = {}

        self.client_name = None
        self.fl_ctx = None

    def initialize(self, fl_ctx: FLContext):

        # FLContext is context information for the client side NVFLARE engine.
        # it includes many runtime information.
        # Here we only interested in client site name.
        # fl_ctx.get_identity_name() will return the client's name

        self.fl_ctx = fl_ctx
        self.client_name = "local_client" if fl_ctx is None else fl_ctx.get_identity_name()

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root=self.dataset_path, train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=self.dataset_path, train=False, download=True, transform=transform)
        self.dataset_lengths = {"train": len(trainset), "test": len(testset)}

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        self.loaders = {"train": trainloader, "test": testloader}

    def features(self) -> Dict[str, List[Feature]]:
        return {"train": self.image_features, "test": self.image_features}

    def count(self, dataset_name: str, feature_name: str) -> int:
        return self.dataset_lengths[dataset_name]

    def histogram(
        self, dataset_name: str, feature_name: str, num_of_bins: int, global_min_value: float, global_max_value: float
    ) -> Histogram:

        print(f"calculating image intensity histogram for client {self.client_name}")
        channel = self.features_ids[feature_name]

        # get the inputs; data is a list of [inputs, labels]
        histogram_bins: List[Bin] = []
        bin_edges = []
        histogram = np.zeros(num_of_bins, dtype=float)

        for inputs, _ in self.loaders[dataset_name]:
            for img in inputs:
                counts, bin_edges = np.histogram(
                    img[channel, :, :], bins=num_of_bins, range=(global_min_value, global_max_value)
                )
                histogram += counts

        for i in range(num_of_bins):
            low_value = bin_edges[i]
            high_value = bin_edges[i + 1]
            bin_sample_count = histogram[i]
            histogram_bins.append(Bin(low_value=low_value, high_value=high_value, sample_count=bin_sample_count))

        return Histogram(HistogramType.STANDARD, histogram_bins)
