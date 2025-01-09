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
from typing import Dict, List

import pytest

from nvflare.app_opt.xgboost.histogram_based_v2.sec.dam import DamDecoder, DamEncoder
from nvflare.app_opt.xgboost.histogram_based_v2.sec.data_converter import FeatureAggregationResult
from nvflare.app_opt.xgboost.histogram_based_v2.sec.processor_data_converter import (
    DATA_SET_AGGREGATION_WITH_FEATURES,
    DATA_SET_GH_PAIRS,
    ProcessorDataConverter,
)


class TestDataConverter:
    @pytest.fixture()
    def data_converter(self):
        yield ProcessorDataConverter()

    @pytest.fixture()
    def gh_buffer(self):

        gh = [0.1, 0.2, 1.2, 1.2, 2.1, 2.2, 3.1, 3.2, 4.1, 4.2, 5.1, 5.2, 6.1, 6.2, 7.1, 7.2, 8.1, 8.2, 9.1, 9.2]

        encoder = DamEncoder(DATA_SET_GH_PAIRS)
        encoder.add_float_array(gh)
        return encoder.finish()

    @pytest.fixture()
    def aggr_buffer(self):

        encoder = DamEncoder(DATA_SET_AGGREGATION_WITH_FEATURES)

        cuts = [0, 2, 5, 10]
        encoder.add_int_array(cuts)

        features = [0, 2]
        encoder.add_int_array(features)

        slots = [
            0,
            5,
            1,
            9,
            1,
            6,
            0,
            7,
            0,
            9,
            0,
            8,
            1,
            5,
            0,
            6,
            0,
            8,
            1,
            5,
        ]
        encoder.add_int_array(slots)

        nodes_to_build = [0, 1]
        encoder.add_int_array(nodes_to_build)

        row_id_1 = [0, 3, 6, 8]
        row_id_2 = [1, 2, 4, 5, 7, 9]
        encoder.add_int_array(row_id_1)
        encoder.add_int_array(row_id_2)

        return encoder.finish()

    @pytest.fixture()
    def aggr_results(self) -> Dict[int, List[FeatureAggregationResult]]:
        feature0 = [(1100000, 1200000), (1200000, 1300000)]
        feature2 = [(1100000, 1200000), (2100000, 2200000), (3100000, 3200000), (4100000, 4200000), (5100000, 5200000)]

        aggr_result0 = FeatureAggregationResult(0, feature0)
        aggr_result2 = FeatureAggregationResult(2, feature2)
        result_list = [aggr_result0, aggr_result2]
        return {0: result_list, 1: result_list}

    def test_decode(self, data_converter, gh_buffer, aggr_buffer):
        gh_pair = data_converter.decode_gh_pairs(gh_buffer, None)
        assert len(gh_pair) == data_converter.num_samples

        context = data_converter.decode_aggregation_context(aggr_buffer, None)
        assert len(context.features) == 2
        f1 = context.features[0]
        assert f1.feature_id == 0
        assert f1.num_bins == 2
        assert f1.sample_bin_assignment == [0, 1, 1, 0, 0, 0, 1, 0, 0, 1]

        f2 = context.features[1]
        assert f2.feature_id == 2
        assert f2.num_bins == 5
        assert f2.sample_bin_assignment == [0, 4, 1, 2, 4, 3, 0, 1, 3, 0]

    def test_encode(self, data_converter, aggr_results):

        # Simulate the state of converter after decode call
        data_converter.feature_list = [0, 2]
        buffer = data_converter.encode_aggregation_result(aggr_results, None)

        decoder = DamDecoder(buffer)
        node_list = decoder.decode_int_array()
        assert node_list == [0, 1]

        feature0 = decoder.decode_int_array()
        histo0 = decoder.decode_float_array()
        assert histo0 == [1.1, 1.2, 1.2, 1.3]

        histo2 = decoder.decode_float_array()
        assert histo2 == [1.1, 1.2, 2.1, 2.2, 3.1, 3.2, 4.1, 4.2, 5.1, 5.2]
