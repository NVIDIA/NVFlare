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

"""Tests for the FedSCS and clipped FedAvg aggregators."""

import numpy as np
import pytest

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.abstract.fl_model import FLModel
from research.fedscs.src.fedavg_clipped_aggregator import FedAvgClippedAggregator
from research.fedscs.src.fedscs_aggregator import FedSCSAggregator


def test_float32_update_remains_float32_after_norm_clipping():
    """Clipped float32 updates must preserve the authoritative dtype."""
    expected_schema = {"w": (2,)}
    expected_dtypes = {"w": "float32"}

    aggregator = FedSCSAggregator(
        expected_schema=expected_schema,
        expected_dtypes=expected_dtypes,
        max_update_norm=10.0,
    )

    update = {
        "w": np.array([20.0, 20.0], dtype=np.float32),
    }

    bounded_update = aggregator._bound_update_norm(
        update=update,
        client_name="site-1",
    )

    assert bounded_update["w"].dtype == np.dtype("float32")
    assert np.all(np.isfinite(bounded_update["w"]))

    bounded_norm = np.linalg.norm(bounded_update["w"].astype(np.float64))
    assert bounded_norm <= 10.0 + 1e-6


def test_unclipped_float32_update_preserves_dtype():
    """Unclipped float32 updates must preserve the authoritative dtype."""
    expected_schema = {"w": (2,)}
    expected_dtypes = {"w": "float32"}

    aggregator = FedSCSAggregator(
        expected_schema=expected_schema,
        expected_dtypes=expected_dtypes,
        max_update_norm=10.0,
    )

    update = {
        "w": np.array([3.0, 4.0], dtype=np.float32),
    }

    bounded_update = aggregator._bound_update_norm(
        update=update,
        client_name="site-1",
    )

    assert bounded_update["w"].dtype == np.dtype("float32")
    np.testing.assert_array_equal(
        bounded_update["w"],
        update["w"],
    )


def _make_clipped_aggregator():
    """Create a schema-validated clipped FedAvg aggregator for testing."""
    return FedAvgClippedAggregator(
        max_update_norm=10.0,
        expected_schema={"w": (2,)},
        expected_dtypes={"w": "float32"},
    )


def _make_clipped_model(params, num_steps=1):
    """Create a clipped FedAvg test model."""
    return FLModel(
        params=params,
        params_type="DIFF",
        current_round=1,
        meta={
            "client_name": "site-1",
            FLMetaKey.NUM_STEPS_CURRENT_ROUND: num_steps,
        },
    )


def test_clipped_fedavg_bounds_large_num_steps_weight():
    """Extremely large client step weights must remain bounded."""
    aggregator = _make_clipped_aggregator()

    model = _make_clipped_model(
        {"w": np.array([1.0, 2.0], dtype=np.float32)},
        num_steps=1e308,
    )

    assert aggregator.accept_model(model)

    result = aggregator.aggregate_model()

    assert result.params is not None
    assert np.all(np.isfinite(result.params["w"]))
    np.testing.assert_allclose(
        result.params["w"],
        np.array([1.0, 2.0], dtype=np.float32),
    )


def test_clipped_fedavg_rejects_parameter_schema_mismatch():
    """Clipped FedAvg must reject missing or extra parameters."""
    aggregator = _make_clipped_aggregator()

    model = _make_clipped_model(
        {
            "w": np.array([1.0, 2.0], dtype=np.float32),
            "unexpected": np.array([1.0], dtype=np.float32),
        }
    )

    with pytest.raises(ValueError, match="Parameter schema mismatch"):
        aggregator.accept_model(model)


def test_clipped_fedavg_rejects_parameter_shape_mismatch():
    """Clipped FedAvg must reject incompatible parameter shapes."""
    aggregator = _make_clipped_aggregator()

    model = _make_clipped_model({"w": np.array([1.0, 2.0, 3.0], dtype=np.float32)})

    with pytest.raises(ValueError, match="Parameter shape mismatch"):
        aggregator.accept_model(model)


def test_clipped_fedavg_rejects_parameter_dtype_mismatch():
    """Clipped FedAvg must reject incompatible parameter dtypes."""
    aggregator = _make_clipped_aggregator()

    model = _make_clipped_model({"w": np.array([1.0, 2.0], dtype=np.float64)})

    with pytest.raises(ValueError, match="Parameter dtype mismatch"):
        aggregator.accept_model(model)
