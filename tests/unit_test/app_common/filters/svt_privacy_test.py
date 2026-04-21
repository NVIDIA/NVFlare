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

import numpy as np
import pytest

import nvflare.app_common.filters.svt_privacy as svt_privacy_module
from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.filters import SVTPrivacy


def _make_weight_diff(include_scalar=True):
    weight_diff = {
        "layer_a": np.array([0.2, -0.3, 0.4, -0.5], dtype=np.float32),
        "layer_b": np.array([[0.1, -0.2], [0.3, -0.4]], dtype=np.float32),
    }
    if include_scalar:
        weight_diff["scalar"] = np.asarray(7.0, dtype=np.float32)
    return weight_diff


def _run_filter(weight_diff, **kwargs):
    dxo = DXO(
        data_kind=DataKind.WEIGHT_DIFF,
        data=weight_diff,
        meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 1},
    )
    filtered = SVTPrivacy(**kwargs).process(dxo.to_shareable(), FLContext())
    return from_shareable(filtered)


def test_svt_privacy_selects_requested_number_without_replacement():
    np.random.seed(7)

    result = _run_filter(
        _make_weight_diff(include_scalar=False),
        fraction=0.5,
        epsilon=1.0,
        noise_var=1e9,
        gamma=1.0,
        tau=-1.0,
        replace=False,
    )

    nonzero_count = np.count_nonzero(result.data["layer_a"]) + np.count_nonzero(result.data["layer_b"])
    assert nonzero_count == 4
    assert result.data["layer_a"].shape == (4,)
    assert result.data["layer_b"].shape == (2, 2)
    assert result.data["layer_a"].dtype == np.float32


def test_svt_privacy_zero_fraction_zeros_arrays_and_keeps_scalar():
    np.random.seed(13)

    result = _run_filter(
        _make_weight_diff(),
        fraction=0.0,
        epsilon=1.0,
        noise_var=1.0,
        gamma=1.0,
        tau=-1.0,
        replace=False,
    )

    assert np.count_nonzero(result.data["layer_a"]) == 0
    assert np.count_nonzero(result.data["layer_b"]) == 0
    assert result.data["scalar"] == np.float32(7.0)


def test_svt_privacy_replace_keeps_shapes_and_scalar_passthrough():
    np.random.seed(23)

    result = _run_filter(
        _make_weight_diff(include_scalar=False),
        fraction=0.75,
        epsilon=1.0,
        noise_var=1e9,
        gamma=1.0,
        tau=-1.0,
        replace=True,
    )

    nonzero_count = np.count_nonzero(result.data["layer_a"]) + np.count_nonzero(result.data["layer_b"])
    assert 0 < nonzero_count <= 6
    assert result.data["layer_a"].shape == (4,)
    assert result.data["layer_b"].shape == (2, 2)
    assert result.data["layer_a"].dtype == np.float32


def test_svt_privacy_counts_scalar_toward_upload_budget_but_preserves_scalar_value():
    np.random.seed(29)

    result = _run_filter(
        _make_weight_diff(),
        fraction=0.5,
        epsilon=1.0,
        noise_var=1e9,
        gamma=1.0,
        tau=-1.0,
        replace=False,
    )

    nonzero_count = (
        np.count_nonzero(result.data["layer_a"])
        + np.count_nonzero(result.data["layer_b"])
        + np.count_nonzero(result.data["scalar"])
    )
    assert nonzero_count in (5, 6)
    assert result.data["scalar"] == np.float32(7.0)


@pytest.mark.parametrize(
    "group_counts,total_to_sample,replace",
    [
        ([3, 0, 5, 2], 7, False),
        ([3, 0, 5, 2], 7, True),
    ],
)
def test_sample_partition_counts_invariants(group_counts, total_to_sample, replace):
    np.random.seed(31)

    sampled_counts = svt_privacy_module._sample_partition_counts(group_counts, total_to_sample, replace)

    assert len(sampled_counts) == len(group_counts)
    assert sum(sampled_counts) == total_to_sample
    assert all(count >= 0 for count in sampled_counts)
    if not replace:
        assert all(count <= group_count for count, group_count in zip(sampled_counts, group_counts))


def test_svt_privacy_uses_chunked_path_without_mutating_input(monkeypatch):
    monkeypatch.setattr(svt_privacy_module, "_SVT_CHUNK_SIZE", 3)
    np.random.seed(37)

    original = np.linspace(-0.8, 0.8, 8, dtype=np.float32)
    weight_diff = {"layer": original.copy()}

    result = _run_filter(
        weight_diff,
        fraction=0.5,
        epsilon=1.0,
        noise_var=1e9,
        gamma=1.0,
        tau=-1.0,
        replace=False,
    )

    assert np.count_nonzero(result.data["layer"]) == 4
    np.testing.assert_array_equal(weight_diff["layer"], original)
