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
import os.path
from typing import Dict, Optional, List

import numpy as np

from load_data_utils import get_app_paths
from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.statistics_spec import (
    Statistics,
    Histogram,
    HistogramType,
    Feature,
    DataType, Bin)


class ImageHistogram(Statistics):

    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.data: Optional[Dict[str, List[str]]] = None
        self.features_ids = {
            "density-red": 0,
            "density-green": 1,
            "density-blue": 2,
        }
        self.client_name = None

    def load_data(self, fl_ctx: FLContext) -> Dict[str, List[str]]:
        client_name = fl_ctx.get_prop(ReservedKey.CLIENT_NAME)
        self.client_name = client_name
        self.log_info(fl_ctx, f"load data for client {client_name}")

        workspace_dir, job_dir = get_app_paths(fl_ctx)
        data_path = f"{workspace_dir}/{self.data_path}"
        print(data_path)
        if not os.path.exists(data_path):
            raise ValueError(f"{data_path} doesn't exists")

        image_paths = []
        for root, dirs, files in os.walk(data_path, topdown=False, followlinks=True):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if file_path and file_name.endswith(".jpg"):
                    image_paths.append(file_path)

        return {"train": image_paths}

    def initialize(self, parts: dict, fl_ctx: FLContext):
        self.data = self.load_data(fl_ctx)
        if self.data is None:
            raise ValueError("data is not loaded. make sure the data is loaded")

    def features(self) -> Dict[str, List[Feature]]:
        return {"train": [Feature("density-red", DataType.FLOAT),
                          Feature("density-green", DataType.FLOAT),
                          Feature("density-blue", DataType.FLOAT)
                          ]}

    def count(self,
              dataset_name: str,
              feature_name: str) -> int:

        image_paths = self.data[dataset_name]
        print("client = ", self.client_name, f"count = ", len(image_paths))
        return len(image_paths)

    def histogram(self,
                  dataset_name: str,
                  feature_name: str,
                  num_of_bins: int,
                  global_min_value: float,
                  global_max_value: float) -> Histogram:

        print("client = ", self.client_name, "calculating histogram")

        num_of_bins: int = num_of_bins
        channel = self.features_ids[feature_name]

        import imageio.v2 as imageio
        image_paths = self.data[dataset_name]

        histogram_bins: List[Bin] = []

        bin_edges = []

        histogram = np.zeros(num_of_bins, dtype=np.float)

        for i, img_path in enumerate(image_paths):
            img = imageio.imread(img_path)
            counts, bin_edges = np.histogram(img[:, :, channel],
                                             bins=num_of_bins,
                                             range=(global_min_value, global_max_value))
            histogram += counts

        for i in range(num_of_bins):
            low_value = bin_edges[i]
            high_value = bin_edges[i + 1]
            bin_sample_count = histogram[i]
            histogram_bins.append(Bin(low_value=low_value, high_value=high_value, sample_count=bin_sample_count))

        return Histogram(HistogramType.STANDARD, histogram_bins)
