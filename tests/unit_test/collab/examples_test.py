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

import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLES_ROOT = _REPO_ROOT / "examples"


def test_hello_numpy_collab_trains_and_averages_models(monkeypatch):
    monkeypatch.syspath_prepend(str(_EXAMPLES_ROOT))
    module = importlib.import_module("collab.hello_numpy_collab.hello_numpy_collab")
    initial_model = module.INITIAL_MODEL.copy()

    updated_model, weight_mean = module.train(initial_model, "full")

    np.testing.assert_array_equal(updated_model, initial_model + 1)
    assert weight_mean == 6.0

    model_diff, weight_mean = module.train(initial_model, "diff")
    np.testing.assert_array_equal(model_diff, np.ones_like(initial_model))
    assert weight_mean == 6.0


def test_hello_numpy_collab_recipe_finalizes_with_module_functions(monkeypatch):
    monkeypatch.syspath_prepend(str(_EXAMPLES_ROOT))
    module = importlib.import_module("collab.hello_numpy_collab.hello_numpy_collab")
    recipe = module.make_recipe(SimpleNamespace(n_clients=2, num_rounds=3, update_type="full"))

    job = recipe.finalize()

    assert recipe.finalize() is job


def test_async_numpy_example_imports_without_torch():
    script = f"""
import importlib.abc
import sys

sys.path[:0] = [{str(_EXAMPLES_ROOT)!r}, {str(_REPO_ROOT)!r}]

class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError("torch is blocked for this test")
        return None

sys.meta_path.insert(0, BlockTorch())
import collab.async_filters_metrics.async_filters_metrics
"""

    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
