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
from typing import Dict, List, Tuple

from nvflare.apis.fl_context import FLContext


class FeatureContext:
    def __init__(self, feature_id, sample_bin_assignment, num_bins: int):
        self.feature_id = feature_id
        self.num_bins = num_bins  # how many bins this feature has
        self.sample_bin_assignment = sample_bin_assignment  # sample/bin assignment; normalized to [0 .. num_bins-1]


class AggregationContext:
    def __init__(self, features: List[FeatureContext], sample_groups: Dict[int, List[int]]):  # group_id => sample Ids
        self.features = features
        self.sample_groups = sample_groups


class FeatureAggregationResult:
    def __init__(self, feature_id: int, aggregated_hist: List[Tuple[int, int]]):
        self.feature_id = feature_id
        self.aggregated_hist = aggregated_hist  # list of (G, H) values, one for each bin of the feature


class DataConverter:
    def decode_gh_pairs(self, buffer: bytes, fl_ctx: FLContext) -> List[Tuple[int, int]]:
        """Decode the buffer to extract (g, h) pairs.

        Args:
            buffer: the buffer to be decoded
            fl_ctx: FLContext info

        Returns: if the buffer contains (g, h) pairs, return a tuple of (g_numbers, h_numbers);
            otherwise, return None

        """
        pass

    def decode_aggregation_context(self, buffer: bytes, fl_ctx: FLContext) -> AggregationContext:
        """Decode the buffer to extract aggregation context info

        Args:
            buffer: buffer to be decoded
            fl_ctx: FLContext info

        Returns: if the buffer contains aggregation context, return an AggregationContext object;
            otherwise, return None

        """
        pass

    def encode_aggregation_result(
        self, aggr_results: Dict[int, List[FeatureAggregationResult]], fl_ctx: FLContext
    ) -> bytes:
        """Encode an individual rank's aggr result to a buffer based on XGB data structure

        Args:
            aggr_results: aggregation result for all features and all groups from all clients
                group_id => list of feature aggr results
            fl_ctx: FLContext info

        Returns: a buffer of bytes

        """
        pass

    def decode_histograms(self, buffer: bytes, fl_ctx: FLContext) -> List[float]:
        """Decode the buffer to extract flattened histograms

        Args:
            buffer: buffer to be decoded
            fl_ctx: FLContext info

        Returns: if the buffer contains histograms, return the flattened histograms
            otherwise, return None

        """
        pass

    def encode_histograms_result(self, histograms: List[float], fl_ctx: FLContext) -> bytes:
        """Encode flattened histograms to be sent back to XGBoost

        Args:
            histograms: The flattened histograms for all features
            fl_ctx: FLContext info

        Returns: a buffer of bytes

        """
        pass
