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
import random
from typing import Dict, List, Tuple

from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.xgboost.histogram_based_v2.aggr import Aggregator
from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant
from nvflare.app_opt.xgboost.histogram_based_v2.sec.data_converter import (
    AggregationContext,
    DataConverter,
    FeatureAggregationResult,
    FeatureContext,
)

SAMPLE_SIZE = 1000
NUM_FEATURES = 30
WORLD_SIZE = 3
RANK_FEATURES = [(0, 10), (10, 20), (20, 30)]


def decode_msg(msg: bytes) -> dict:
    return json.loads(str(msg, "utf-8"))


class TupleAggregator(Aggregator):
    def __init__(self):
        Aggregator.__init__(self, initial_value=(0, 0))

    def add(self, a, b):
        return a[0] + b[0], a[1] + b[1]


class MockDataConverter(DataConverter):
    def _gen_feature(self, num_bins, fid):
        mask = [0] * SAMPLE_SIZE
        for i in range(SAMPLE_SIZE):
            mask[i] = (i + fid) % num_bins
        return FeatureContext(fid, mask, num_bins)

    def _setup(self):
        self.features = {}
        for fid in range(NUM_FEATURES):
            self.features[fid] = self._gen_feature(256, fid)

        for rank, fid_range in enumerate(RANK_FEATURES):
            if fid_range is not None:
                f, t = fid_range
                self.rank_features[rank] = [self.features[fid] for fid in range(f, t)]

    def __init__(self):
        self._features_done = False
        self.gh_pairs = None

        # feature_id => feature
        self.features = {}
        self.rank_features = {}
        self._setup()
        # self.features = {
        #     0: self._gen_feature(256, 0),
        #     1: self._gen_feature(2, 1),
        #     2: self._gen_feature(256, 2),
        #     3: self._gen_feature(16, 3),
        #     4: self._gen_feature(256, 4),
        #     5: self._gen_feature(128, 5),
        # }
        #
        # # rank => features
        # self.rank_features = {
        #     # 0: [self.features[0], self.features[2]],
        #     1: [self.features[0], self.features[1], self.features[3]],
        #     2: [self.features[2], self.features[4], self.features[5]],
        # }

        self.groups = {}

    def decode_gh_pairs(self, buffer: bytes, fl_ctx: FLContext) -> List[Tuple[int, int]]:
        """Decode the buffer to extract (g, h) pairs.

        Args:
            buffer: the buffer to be decoded
            fl_ctx: FLContext info

        Returns: if the buffer contains (g, h) pairs, return a tuple of (g_numbers, h_numbers);
            otherwise, return None

        """
        rank = fl_ctx.get_prop(Constant.PARAM_KEY_RANK)
        if rank != 0:
            # non-label client
            return None

        msg = decode_msg(buffer)
        op = msg["op"]
        if op != "gh":
            return None

        min_value = -999999
        max_value = 999999
        result = []
        for i in range(SAMPLE_SIZE):
            result.append((random.randint(min_value, max_value), random.randint(min_value, max_value)))
        self.gh_pairs = result
        return result

    def decode_aggregation_context(self, buffer: bytes, fl_ctx: FLContext) -> AggregationContext:
        """Decode the buffer to extract aggregation context info

        Args:
            buffer: buffer to be decoded
            fl_ctx: FLContext info

        Returns: if the buffer contains aggregation context, return an AggregationContext object;
            otherwise, return None

        """
        rank = fl_ctx.get_prop(Constant.PARAM_KEY_RANK)
        features = None
        if not self._features_done:
            self._features_done = True
            features = self.rank_features.get(rank)
        else:
            self.groups = {1: [1, 3, 4, 101], 4: [2, 7, 9, 23, 50]}
        return AggregationContext(features, self.groups)

    def _aggregate_feature(self, ctx: FeatureContext, sample_ids):
        aggr = TupleAggregator()
        return aggr.aggregate(self.gh_pairs, ctx.sample_bin_assignment, ctx.num_bins, sample_ids)

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
        # verify result
        for gid, fars in aggr_results.items():
            for far in fars:
                ctx = self.features[far.feature_id]
                sample_ids = self.groups.get(gid)
                expected = self._aggregate_feature(ctx, sample_ids)
                if expected != far.aggregated_hist:
                    print(f"group {gid}: feature {far.feature_id}: expected aggr != received")
                    print(f"{expected=}")
                    print(f"{far.aggregated_hist=}")
                else:
                    print(f"group {gid}: feature {far.feature_id}: Result OK!")

        return os.urandom(4)
