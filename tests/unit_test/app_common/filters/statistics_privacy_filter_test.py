# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.dxo import DXO, DataKind
from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.filters.statistics_privacy_filter import StatisticsPrivacyFilter
from nvflare.app_common.statistics.min_count_cleanser import MinCountCleanser
from nvflare.app_common.statistics.min_max_cleanser import AddNoiseToMinMax
from nvflare.fuel.utils import fobs


def _dxo(counts):
    statistics = {
        StC.STATS_COUNT: {"train": dict(counts)},
        StC.STATS_FAILURE_COUNT: {"train": {name: 0 for name in counts}},
        StC.STATS_SUM: {"train": {name: float(count) for name, count in counts.items()}},
    }
    data = {StC.STATISTICS_TASK_KEY: StC.STATS_1st_STATISTICS, StC.STATS_1st_STATISTICS: fobs.dumps(statistics)}
    return DXO(data_kind=DataKind.STATISTICS, data=data)


def _filter(dxo, cleansers):
    result = StatisticsPrivacyFilter(result_cleanser_ids=[]).filter_stats_statistics(dxo, "site-1", cleansers)
    statistics = fobs.loads(result.data[StC.STATS_1st_STATISTICS])
    marker = result.data.get(StC.STATS_WITHHELD_FEATURES)
    return statistics, None if marker is None else fobs.loads(marker)


class TestStatisticsPrivacyFilterWithheldFeatures:
    def test_reports_the_features_the_cleanser_withheld(self):
        statistics, withheld = _filter(_dxo({"age": 3, "edu": 50}), [MinCountCleanser(min_count=10)])

        assert withheld == {"train": ["age"]}
        assert statistics[StC.STATS_COUNT] == {"train": {"age": 3, "edu": 50}}
        assert statistics[StC.STATS_SUM] == {"train": {"edu": 50.0}}

    def test_reports_nothing_when_no_feature_is_withheld(self):
        statistics, withheld = _filter(_dxo({"age": 30}), [MinCountCleanser(min_count=10)])

        assert withheld is None
        assert statistics[StC.STATS_SUM] == {"train": {"age": 30.0}}

    def test_cleanser_that_does_not_withhold_features_reports_nothing(self):
        _, withheld = _filter(_dxo({"age": 3}), [AddNoiseToMinMax(min_noise_level=0.1, max_noise_level=0.3)])

        assert withheld is None
