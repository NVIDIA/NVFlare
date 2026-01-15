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

import argparse
import os

from monai.bundle import ConfigParser
#from monai.data import ITKReader, load_decathlon_datalist
from monai.transforms import LoadImage

from nvflare.app_common.abstract.statistics_spec import Bin, DataType, Feature, Histogram, HistogramType, Statistics


class MonaiBundleStatistics(Statistics):
    """Statistics generator that computes statistics from MONAI bundle datasets."""

    def __init__(self, bundle_root: str, data_list_key: str="train"):
        super().__init__()
        self.bundle_root = bundle_root
        self.train_parser = None
        self.data_list = None
        self.image_loader = None
        self.feature_names = None
        self.req_num_of_bins = None
        self.req_bin_ranges = None
        self.data_list_key = data_list_key

    def initialize(self, fl_ctx):
        # Parse MONAI bundle configuration
        self.train_parser = ConfigParser()
        self.train_parser.read_config(os.path.join(self.bundle_root, "configs/train.json"))
        
        # Get dataset configuration
        self.data_list = self.train_parser.get_parsed_content(f"{self.data_list_key}#dataset#data")
        
        if len(self.data_list) == 0:
            raise ValueError(f"Data list is empty for key: {self.data_list_key}")
        
        # Setup image loader
        self.image_loader = LoadImage(image_only=True)

    def pre_run(self, statistics, num_of_bins, bin_ranges):
        """Pre-run to determine feature names and bin configuration."""
        if num_of_bins:
            self.req_num_of_bins = list(num_of_bins.values())
            self.feature_names = list(num_of_bins.keys())
        else:
            self.req_num_of_bins = []
            self.feature_names = ["Intensity"]

        if bin_ranges:
            self.req_bin_ranges = list(bin_ranges.values())
        else:
            self.req_bin_ranges = []

        return {}

    def features(self):
        return {"training": [Feature("Intensity", DataType.FLOAT)]}

    def count(self, dataset_name: str, feature_name: str) -> int:
        return len(self.data_list)

    def failure_count(self, dataset_name: str, feature_name: str) -> int:
        return 0

    def histogram(self, dataset_name: str, feature_name: str, num_of_bins: int, 
                  global_min_value: float, global_max_value: float) -> Histogram:
        """Compute histogram over all images in the dataset."""
        import numpy as np
        
        # Initialize histogram bins
        bins = np.linspace(global_min_value, global_max_value, num_of_bins + 1)
        counts = np.zeros(num_of_bins, dtype=int)
        
        # Accumulate histogram over all images
        for item in self.data_list:
            image_path = item["image"]
            image = self.image_loader(image_path)
            
            # Compute histogram for this image
            hist, _ = np.histogram(image.flatten(), bins=bins)
            counts += hist
        
        # Create Histogram object
        histogram_bins = []
        for i in range(num_of_bins):
            histogram_bins.append(Bin(
                low_value=bins[i],
                high_value=bins[i + 1],
                sample_count=int(counts[i])
            ))
        
        return Histogram(HistogramType.STANDARD, histogram_bins)
