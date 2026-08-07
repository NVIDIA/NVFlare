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

import pytest

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.filter import ContentBlockedException
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.filters.convert_weights import ConvertWeights
from nvflare.app_common.filters.dxo_blocker import DXOBlocker
from nvflare.app_common.filters.exclude_vars import ExcludeVars
from nvflare.app_common.filters.percentile_privacy import PercentilePrivacy
from nvflare.app_common.filters.statistics_privacy_filter import StatisticsPrivacyFilter
from nvflare.app_common.filters.svt_privacy import SVTPrivacy


@pytest.mark.parametrize(
    "filter_,data_kind",
    [
        (ConvertWeights(direction=ConvertWeights.WEIGHTS_TO_DIFF), DataKind.WEIGHTS),
        (ExcludeVars(exclude_vars=["layer"]), DataKind.WEIGHTS),
        (PercentilePrivacy(), DataKind.WEIGHTS),
        (StatisticsPrivacyFilter(result_cleanser_ids=[]), DataKind.STATISTICS),
        (SVTPrivacy(), DataKind.WEIGHTS),
    ],
)
def test_data_dependent_filters_skip_empty_dxo(filter_, data_kind):
    dxo = DXO(data_kind=data_kind, data={})

    assert filter_.process_dxo(dxo, Shareable(), FLContext()) is None


def test_dxo_blocker_checks_empty_dxo():
    filter_ = DXOBlocker(data_kinds=[DataKind.WEIGHTS])
    shareable = DXO(data_kind=DataKind.WEIGHTS, data={}).to_shareable()

    with pytest.raises(ContentBlockedException, match="WEIGHTS"):
        filter_.process(shareable, FLContext())
