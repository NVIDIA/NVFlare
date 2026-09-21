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

"""Tests for the FedSCS aggregator."""

import numpy as np

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
