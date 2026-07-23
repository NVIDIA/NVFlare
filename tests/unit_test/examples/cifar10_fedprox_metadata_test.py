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

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.app_constant import AlgorithmConstants
from nvflare.app_common.utils.fedprox_utils import get_fedprox_mu


def test_reads_each_round_coefficient():
    first = FLModel(meta={AlgorithmConstants.FEDPROX_MU: 0.01})
    second = FLModel(meta={AlgorithmConstants.FEDPROX_MU: 0.2})

    assert get_fedprox_mu(first) == 0.01
    assert get_fedprox_mu(second) == 0.2


@pytest.mark.parametrize(
    "meta",
    [
        None,
        {},
        {AlgorithmConstants.FEDPROX_MU: None},
        {AlgorithmConstants.FEDPROX_MU: 0.0},
        {AlgorithmConstants.FEDPROX_MU: -0.1},
        {AlgorithmConstants.FEDPROX_MU: float("inf")},
        {AlgorithmConstants.FEDPROX_MU: float("nan")},
        {AlgorithmConstants.FEDPROX_MU: True},
        {AlgorithmConstants.FEDPROX_MU: "0.1"},
    ],
)
def test_rejects_missing_invalid_or_non_positive_metadata(meta):
    with pytest.raises(ValueError, match="FedProx client"):
        get_fedprox_mu(FLModel(meta=meta))
