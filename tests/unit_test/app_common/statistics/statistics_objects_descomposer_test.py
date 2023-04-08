# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import random
from typing import List

from nvflare.app_common.abstract.statistics_spec import (
    Bin,
    BinRange,
    DataType,
    Feature,
    Histogram,
    HistogramType,
    StatisticConfig,
)
from nvflare.app_common.app_constant import StatisticsConstants
from nvflare.app_common.statistics.statisitcs_objects_decomposer import (
    BinDecomposer,
    BinRangeDecomposer,
    FeatureDecomposer,
    HistogramDecomposer,
    StatisticConfigDecomposer,
)
from nvflare.fuel.utils import fobs


class TestStatisticConfigDecomposer:
    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test_statistic_configs_serde(self):
        fobs.register(StatisticConfigDecomposer)

        data = fobs.dumps(StatisticConfig("foo", {}))
        obj: StatisticConfig = fobs.loads(data)
        assert isinstance(obj, StatisticConfig)
        assert obj.config == {}
        assert obj.name == "foo"

    def test_statistic_configs_serde2(self):

        config = """
            {
                "count": {},
                "mean": {},
                "sum": {},
                "stddev": {},
                "histogram": {
                    "*": {"bins": 20},
                    "Age": { "bins": 10, "range": [0,120] }
                 }
            }
        """
        config_dict = json.loads(config)
        statistic_configs = []
        for k in config_dict:
            statistic_configs.append([k, config_dict[k]])

        data = fobs.dumps(statistic_configs)
        obj = fobs.loads(data)
        assert isinstance(obj, List)
        for o in obj:
            assert isinstance(o, list)
            print(o)
            assert o[0] in config_dict.keys()

    def test_statistic_configs_serde3(self):
        fobs.register(StatisticConfigDecomposer)
        config = """
            {
                "count": {},
                "mean": {},
                "sum": {},
                "stddev": {},
                "histogram": {
                    "*": {"bins": 20},
                    "Age": { "bins": 10, "range": [0,120] }
                 }
            }
        """
        config_dict = json.loads(config)
        from nvflare.app_common.workflows.statistics_controller import StatisticsController

        ordered_statistics = StatisticsConstants.ordered_statistics[StatisticsConstants.STATS_1st_STATISTICS]
        target_configs: List[StatisticConfig] = StatisticsController._get_target_statistics(
            config_dict, ordered_statistics
        )
        o = fobs.dumps(target_configs)
        target_configs1 = fobs.loads(o)
        assert target_configs == target_configs1

    def test_datatype_serde(self):
        dt = DataType.FLOAT
        o = fobs.dumps(dt)
        dt1 = fobs.loads(o)
        assert dt == dt1

    def test_histogram_type_serde(self):
        f = HistogramType.STANDARD
        o = fobs.dumps(f)
        f1 = fobs.loads(o)
        assert f1 == f

    def test_feature_serde(self):
        fobs.register(FeatureDecomposer)

        f = Feature(feature_name="feature1", data_type=DataType.INT)
        o = fobs.dumps(f)
        f1 = fobs.loads(o)
        assert f1 == f

    def test_bin_serde(self):
        fobs.register(BinDecomposer)

        f = Bin(low_value=0, high_value=255, sample_count=100)
        o = fobs.dumps(f)
        f1 = fobs.loads(o)
        assert f1 == f

    def test_bin_range_serde(self):
        fobs.register(BinRangeDecomposer)
        f = BinRange(min_value=0, max_value=255)
        o = fobs.dumps(f)
        f1 = fobs.loads(o)
        assert f1 == f

    def test_histogram_serde(self):
        fobs.register(HistogramDecomposer)
        fobs.register(BinDecomposer)
        fobs.register(BinRangeDecomposer)
        bins = []
        for i in range(0, 10):
            bins = Bin(i, i + 1, random.randint(0, 100))
        f = Histogram(HistogramType.STANDARD, bins)
        o = fobs.dumps(f)
        f1 = fobs.loads(o)
        assert f1 == f
