# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import glob
import os
from typing import Dict, List, Optional

import numpy as np
from monai.data import ITKReader, load_decathlon_datalist
from monai.transforms import LoadImage

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.statistics_spec import Bin, DataType, Feature, Histogram, HistogramType, Statistics
from nvflare.security.logging import secure_log_traceback


class ImageStatistics(Statistics):
    def __init__(self, data_root: str = "/tmp/nvflare/image_stats/data", data_list_key: str = "data"):
        """local image statistics generator .

        Args:
            data_root: directory with local image data.
            data_list_key: data list key to use.
        Returns:
            a Shareable with the computed local statistics`
        """
        super().__init__()
        self.data_list_key = data_list_key
        self.data_root = data_root
        self.data_list = None
        self.client_name = None

        self.loader = None
        self.failure_images = 0
        self.fl_ctx = None

    def initialize(self, fl_ctx: FLContext):
        self.fl_ctx = fl_ctx
        self.client_name = fl_ctx.get_identity_name()
        self.loader = LoadImage(image_only=True)
        self.loader.register(ITKReader())
        self._load_data_list(self.client_name, fl_ctx)

        if self.data_list is None:
            raise ValueError("data is not loaded. make sure the data is loaded")

    def _load_data_list(self, client_name, fl_ctx: FLContext) -> bool:
        dataset_json = glob.glob(os.path.join(self.data_root, client_name + "*.json"))
        if len(dataset_json) != 1:
            self.log_error(
                fl_ctx, f"No unique matching dataset list found in {self.data_root} for client {client_name}"
            )
            return False
        dataset_json = dataset_json[0]
        self.log_info(fl_ctx, f"Reading data from {dataset_json}")

        data_list = load_decathlon_datalist(
            data_list_file_path=dataset_json, data_list_key=self.data_list_key, base_dir=self.data_root
        )
        self.data_list = {"train": data_list}

        self.log_info(fl_ctx, f"Client {client_name} has {len(self.data_list)} images")
        return True

    def pre_run(
        self,
        statistics: List[str],
        num_of_bins: Optional[Dict[str, Optional[int]]],
        bin_ranges: Optional[Dict[str, Optional[List[float]]]],
    ):
        return {}

    def features(self) -> Dict[str, List[Feature]]:
        return {"train": [Feature("intensity", DataType.FLOAT)]}

    def count(self, dataset_name: str, feature_name: str) -> int:
        image_paths = self.data_list[dataset_name]
        return len(image_paths)

    def failure_count(self, dataset_name: str, feature_name: str) -> int:

        return self.failure_images

    def histogram(
        self, dataset_name: str, feature_name: str, num_of_bins: int, global_min_value: float, global_max_value: float
    ) -> Histogram:
        histogram_bins: List[Bin] = []
        histogram = np.zeros((num_of_bins,), dtype=np.int64)
        bin_edges = []
        for i, entry in enumerate(self.data_list[dataset_name]):
            file = entry.get("image")
            try:
                img = self.loader(file)
                curr_histogram, bin_edges = np.histogram(
                    img, bins=num_of_bins, range=(global_min_value, global_max_value)
                )
                histogram += curr_histogram
                bin_edges = bin_edges.tolist()

                if i % 100 == 0:
                    self.logger.info(
                        f"{self.client_name}, adding {i + 1} of {len(self.data_list[dataset_name])}: {file}"
                    )
            except Exception as e:
                self.failure_images += 1
                self.logger.critical(
                    f"Failed to load file {file} with exception: {e.__str__()}. " f"Skipping this image..."
                )

        if num_of_bins + 1 != len(bin_edges):
            secure_log_traceback()
            raise ValueError(
                f"bin_edges size: {len(bin_edges)} is not matching with number of bins + 1: {num_of_bins + 1}"
            )

        for j in range(num_of_bins):
            low_value = bin_edges[j]
            high_value = bin_edges[j + 1]
            bin_sample_count = histogram[j]
            histogram_bins.append(Bin(low_value=low_value, high_value=high_value, sample_count=bin_sample_count))

        return Histogram(HistogramType.STANDARD, histogram_bins)
