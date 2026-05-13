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

import importlib.util
import os
import sys

import numpy as np
import pytest

from nvflare.fuel.utils import fobs

HAS_CMA_DEPS = importlib.util.find_spec("cma") is not None
pytestmark = pytest.mark.skipif(not HAS_CMA_DEPS, reason="FedBPT CMA dependencies are not installed")


def test_cma_evolution_strategy_roundtrip_after_fobs_reset():
    import cma
    import cma.evolution_strategy

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    fedbpt_src = os.path.join(repo_root, "research", "fed-bpt", "src")

    sys.path.insert(0, fedbpt_src)
    try:
        from cma_decomposer import register_decomposers

        fobs.reset()
        register_decomposers()
        strategy = cma.CMAEvolutionStrategy(4 * [5], 10, {"ftarget": 1e-9, "seed": 5, "verbose": -9})
        data = fobs.dumps(strategy)

        fobs.reset()
        register_decomposers()
        restored = fobs.loads(data)
    finally:
        try:
            fobs.reset()
        finally:
            if fedbpt_src in sys.path:
                sys.path.remove(fedbpt_src)
            sys.modules.pop("cma_decomposer", None)

    assert isinstance(restored, cma.CMAEvolutionStrategy)
    assert isinstance(restored._stoptolxstagnation, cma.evolution_strategy._StopTolXStagnation)
    np.testing.assert_allclose(restored.mean, strategy.mean)
    assert restored.sigma == strategy.sigma
