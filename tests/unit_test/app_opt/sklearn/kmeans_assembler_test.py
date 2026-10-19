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

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_opt.sklearn.kmeans_assembler import KMeansAssembler


def assemble_round(assembler, current_round, contributions):
    assembler.reset()
    assembler.collection.update(contributions)
    fl_ctx = FLContext()
    fl_ctx.set_prop(AppConstants.CURRENT_ROUND, current_round)
    with np.errstate(divide="raise", invalid="raise"):
        return assembler.assemble(assembler.collection, fl_ctx)


@pytest.fixture()
def initialized_assembler():
    assembler = KMeansAssembler()
    seeds = np.array([[0.0, 0.0], [10.0, 10.0]])
    assemble_round(
        assembler,
        0,
        {
            "site-1": {"center": seeds, "count": None},
            "site-2": {"center": seeds, "count": None},
        },
    )
    return assembler


def test_empty_cluster_preserves_initial_center_and_recovers(initialized_assembler):
    assembler = initialized_assembler
    initial_centers = assembler.center.copy()
    result = assemble_round(
        assembler,
        1,
        {
            "site-1": {"center": np.array([[100.0, 100.0], [2.0, 4.0]]), "count": np.array([0.0, 2.0])},
            "site-2": {"center": np.array([[200.0, 200.0], [8.0, 10.0]]), "count": np.array([0.0, 3.0])},
        },
    )
    np.testing.assert_allclose(result.data["center"][0], initial_centers[0])
    np.testing.assert_allclose(result.data["center"][1], [5.6, 7.6])
    np.testing.assert_allclose(assembler.count, [0.0, 5.0])
    assert np.isfinite(result.data["center"]).all()

    # The empty cluster can receive contributions in the next round, while a
    # previously populated cluster retains its center if it has no new assignments.
    result = assemble_round(
        assembler,
        2,
        {
            "site-1": {"center": np.array([[4.0, 6.0], [100.0, 100.0]]), "count": np.array([2.0, 0.0])},
            "site-2": {"center": np.array([[8.0, 10.0], [200.0, 200.0]]), "count": np.array([3.0, 0.0])},
        },
    )
    np.testing.assert_allclose(result.data["center"], [[6.4, 8.4], [5.6, 7.6]])
    np.testing.assert_allclose(assembler.count, [5.0, 5.0])
    assert np.isfinite(result.data["center"]).all()


def test_weighted_aggregation_includes_historical_counts(initialized_assembler):
    assembler = initialized_assembler
    result = assemble_round(
        assembler,
        1,
        {
            "site-1": {"center": np.array([[2.0, 4.0], [6.0, 8.0]]), "count": np.array([2.0, 3.0])},
            "site-2": {"center": np.array([[12.0, 14.0], [10.0, 12.0]]), "count": np.array([3.0, 1.0])},
        },
    )
    np.testing.assert_allclose(result.data["center"], [[8.0, 10.0], [7.0, 9.0]])
    np.testing.assert_allclose(assembler.count, [5.0, 4.0])

    result = assemble_round(
        assembler,
        2,
        {
            "site-1": {"center": np.array([[20.0, 22.0], [4.0, 6.0]]), "count": np.array([1.0, 2.0])},
            "site-2": {"center": np.array([[10.0, 12.0], [16.0, 18.0]]), "count": np.array([4.0, 1.0])},
        },
    )
    np.testing.assert_allclose(result.data["center"], [[10.0, 12.0], [52.0 / 7.0, 66.0 / 7.0]])
    np.testing.assert_allclose(assembler.count, [10.0, 7.0])
