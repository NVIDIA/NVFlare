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

"""Smoke tests for the Collab example catalog.

Examples are documentation, so nothing here asserts training behavior. Each
example must import and its recipe must finalize into a job — the cheapest
alarm for drift between the catalog and the nvflare.collab API.
"""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLES_ROOT = _REPO_ROOT / "examples"
_ADVANCED_EXAMPLES_ROOT = _EXAMPLES_ROOT / "advanced"
_HELLO_COLLAB_ROOT = _EXAMPLES_ROOT / "hello-world" / "hello-collab"

_EXAMPLES = [
    pytest.param(
        _HELLO_COLLAB_ROOT,
        "hello_numpy_collab",
        SimpleNamespace(n_clients=2, num_rounds=3, update_type="full"),
        id="hello_numpy_collab",
    ),
    pytest.param(
        _ADVANCED_EXAMPLES_ROOT,
        "collab.async_aggregation.async_aggregation",
        SimpleNamespace(num_clients=2, num_rounds=2),
        id="async_aggregation",
    ),
]


@pytest.mark.parametrize("example_root,module_name,args", _EXAMPLES)
def test_example_recipe_finalizes(monkeypatch, example_root, module_name, args):
    monkeypatch.syspath_prepend(str(example_root))
    module = importlib.import_module(module_name)
    recipe = module.make_recipe(args)

    job = recipe.finalize()

    assert recipe.finalize() is job
