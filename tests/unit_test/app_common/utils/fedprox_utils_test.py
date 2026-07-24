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
from nvflare.app_common.utils.fedprox_utils import (
    get_fedprox_mu,
    normalize_fedprox_mu,
    set_fedprox_metadata,
    validate_fedprox_mu,
)


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        (0, None),
        (0.0, None),
        (1, 1.0),
        (0.25, 0.25),
    ],
)
def test_normalize_fedprox_mu(value, expected):
    assert normalize_fedprox_mu(value) == expected


@pytest.mark.parametrize("value", [True, "0.1", object()])
def test_normalize_fedprox_mu_rejects_non_numeric_values(value):
    with pytest.raises(TypeError, match="finite non-negative number or None"):
        normalize_fedprox_mu(value)


@pytest.mark.parametrize("value", [-0.1, float("inf"), float("-inf"), float("nan")])
def test_normalize_fedprox_mu_rejects_invalid_numeric_values(value):
    with pytest.raises(ValueError, match="finite non-negative number or None"):
        normalize_fedprox_mu(value)


@pytest.mark.parametrize("value, expected", [(1, 1.0), (0.25, 0.25)])
def test_validate_fedprox_mu(value, expected):
    assert validate_fedprox_mu(value) == expected


@pytest.mark.parametrize("value", [None, True, "0.1", object()])
def test_validate_fedprox_mu_rejects_non_numeric_values(value):
    with pytest.raises(TypeError, match="finite positive number"):
        validate_fedprox_mu(value)


@pytest.mark.parametrize("value", [0.0, -0.1, float("inf"), float("-inf"), float("nan")])
def test_validate_fedprox_mu_rejects_invalid_numeric_values(value):
    with pytest.raises(ValueError, match="finite positive number"):
        validate_fedprox_mu(value)


def test_get_fedprox_mu_reads_each_round_coefficient():
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
def test_get_fedprox_mu_rejects_missing_invalid_or_non_positive_metadata(meta):
    with pytest.raises(ValueError, match="FedProx client"):
        get_fedprox_mu(FLModel(meta=meta))


def test_set_fedprox_metadata_sets_or_updates_value_and_preserves_other_metadata():
    model = FLModel(meta={"other": "value", AlgorithmConstants.FEDPROX_MU: 0.1})

    set_fedprox_metadata(model, 0.2)

    assert model.meta == {"other": "value", AlgorithmConstants.FEDPROX_MU: 0.2}


def test_set_fedprox_metadata_supports_explicit_zero():
    model = FLModel()

    set_fedprox_metadata(model, 0.0)

    assert model.meta == {AlgorithmConstants.FEDPROX_MU: 0.0}


def test_set_fedprox_metadata_removes_stale_value_and_preserves_other_metadata():
    model = FLModel(meta={"other": "value", AlgorithmConstants.FEDPROX_MU: 0.1})

    set_fedprox_metadata(model, None)

    assert model.meta == {"other": "value"}
