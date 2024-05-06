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
from nvflare.app_opt.xgboost.histogram_based_v2.sec.dam import DamDecoder, DamEncoder
from nvflare.app_opt.xgboost.histogram_based_v2.sec.data_converter import (
    AggregationContext,
    DataConverter,
    FeatureAggregationResult,
    FeatureContext,
)

DATA_SET_GH_PAIRS = 1
DATA_SET_AGGREGATION = 2
DATA_SET_AGGREGATION_WITH_FEATURES = 3
DATA_SET_AGGREGATION_RESULT = 4
DATA_SET_HISTOGRAMS = 5
DATA_SET_HISTOGRAMS_RESULT = 6

SCALE_FACTOR = 1000000.0  # Preserve 6 decimal places


class ProcessorDataConverter(DataConverter):
    def __init__(self):
        super().__init__()
        self.features = []
        self.feature_list = None
        self.num_samples = 0

    def decode_gh_pairs(self, buffer: bytes, fl_ctx: FLContext) -> List[Tuple[int, int]]:
        decoder = DamDecoder(buffer)
        if not decoder.is_valid():
            return None

        if decoder.get_data_set_id() != DATA_SET_GH_PAIRS:
            raise RuntimeError(f"Data is not for GH Pairs: {decoder.get_data_set_id()}")

        float_array = decoder.decode_float_array()
        result = []
        self.num_samples = int(len(float_array) / 2)

        for i in range(self.num_samples):
            result.append((self.float_to_int(float_array[2 * i]), self.float_to_int(float_array[2 * i + 1])))

        return result

    def decode_aggregation_context(self, buffer: bytes, fl_ctx: FLContext) -> AggregationContext:
        decoder = DamDecoder(buffer)
        if not decoder.is_valid():
            return None
        data_set_id = decoder.get_data_set_id()
        cuts = decoder.decode_int_array()

        if data_set_id == DATA_SET_AGGREGATION_WITH_FEATURES:
            self.feature_list = decoder.decode_int_array()
            num = len(self.feature_list)
            slots = decoder.decode_int_array()
            num_samples = int(len(slots) / num)
            for i in range(num):
                bin_assignment = []
                for row_id in range(num_samples):
                    _, bin_num = self.slot_to_bin(cuts, slots[row_id * num + i])
                    bin_assignment.append(bin_num)

                bin_size = self.get_bin_size(cuts, self.feature_list[i])
                feature_ctx = FeatureContext(self.feature_list[i], bin_assignment, bin_size)
                self.features.append(feature_ctx)
        elif data_set_id != DATA_SET_AGGREGATION:
            raise RuntimeError(f"Invalid DataSet: {data_set_id}")

        node_list = decoder.decode_int_array()
        sample_groups = {}
        for node in node_list:
            row_ids = decoder.decode_int_array()
            sample_groups[node] = row_ids

        return AggregationContext(self.features, sample_groups)

    def encode_aggregation_result(
        self, aggr_results: Dict[int, List[FeatureAggregationResult]], fl_ctx: FLContext
    ) -> bytes:
        encoder = DamEncoder(DATA_SET_AGGREGATION_RESULT)
        node_list = sorted(aggr_results.keys())
        encoder.add_int_array(node_list)
        for node in node_list:
            result_list = aggr_results.get(node)
            feature_list = [result.feature_id for result in result_list]
            encoder.add_int_array(feature_list)
            for result in result_list:
                encoder.add_float_array(self.to_float_array(result))

        return encoder.finish()

    def decode_histograms(self, buffer: bytes, fl_ctx: FLContext) -> List[float]:
        decoder = DamDecoder(buffer)
        if not decoder.is_valid():
            return None
        data_set_id = decoder.get_data_set_id()
        if data_set_id != DATA_SET_HISTOGRAMS:
            raise RuntimeError(f"Invalid DataSet: {data_set_id}")

        return decoder.decode_float_array()

    def encode_histograms_result(self, histograms: List[float], fl_ctx: FLContext) -> bytes:
        encoder = DamEncoder(DATA_SET_HISTOGRAMS_RESULT)
        encoder.add_float_array(histograms)
        return encoder.finish()

    @staticmethod
    def get_bin_size(cuts: [int], feature_id: int) -> int:
        return cuts[feature_id + 1] - cuts[feature_id]

    @staticmethod
    def slot_to_bin(cuts: [int], slot: int) -> Tuple[int, int]:
        if slot < 0 or slot >= cuts[-1]:
            raise RuntimeError(f"Invalid slot {slot}, out of range [0-{cuts[-1]-1}]")

        for i in range(len(cuts) - 1):
            if cuts[i] <= slot < cuts[i + 1]:
                bin_num = slot - cuts[i]
                return i, bin_num

        raise RuntimeError(f"Logic error. Slot {slot}, out of range [0-{cuts[-1] - 1}]")

    @staticmethod
    def float_to_int(value: float) -> int:
        return int(value * SCALE_FACTOR)

    @staticmethod
    def int_to_float(value: int) -> float:
        return value / SCALE_FACTOR

    @staticmethod
    def to_float_array(result: FeatureAggregationResult) -> List[float]:
        float_array = []
        for (g, h) in result.aggregated_hist:
            float_array.append(ProcessorDataConverter.int_to_float(g))
            float_array.append(ProcessorDataConverter.int_to_float(h))

        return float_array
