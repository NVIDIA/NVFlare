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

_PT_SYNC_CIFAR10_ROOT = _ADVANCED_EXAMPLES_ROOT / "collab" / "pt_sync_cifar10"


def _pt_sync_cifar10_args(output_root):
    return SimpleNamespace(
        data_root="/tmp/nvflare/datasets/cifar10_sync",
        output_root=output_root,
        workspace_root="/tmp/nvflare/collab",
        num_clients=2,
        num_rounds=1,
        local_epochs=1,
        batch_size=64,
        learning_rate=0.01,
        momentum=0.9,
        num_workers=0,
        eval_batch_size=256,
        device="cpu",
        seed=42,
        call_timeout=3600.0,
        mu=0.01,
    )


_EXAMPLES = [
    pytest.param(
        _HELLO_COLLAB_ROOT,
        "job",
        SimpleNamespace(n_clients=2, num_rounds=3, update_type="full"),
        (),
        id="hello_numpy_collab",
    ),
    pytest.param(
        _ADVANCED_EXAMPLES_ROOT,
        "collab.hello_fedavg.hello_fedavg",
        SimpleNamespace(num_clients=2, num_rounds=3),
        ("torch",),
        id="hello_fedavg",
    ),
    pytest.param(
        _ADVANCED_EXAMPLES_ROOT,
        "collab.simple_split_learning.simple_split_learning",
        None,
        ("torch", "torchvision"),
        id="simple_split_learning",
    ),
    pytest.param(
        _ADVANCED_EXAMPLES_ROOT,
        "collab.async_aggregation.async_aggregation",
        SimpleNamespace(num_clients=2, num_rounds=2),
        (),
        id="async_aggregation",
    ),
    pytest.param(
        _ADVANCED_EXAMPLES_ROOT,
        "collab.swarm.swarm",
        SimpleNamespace(num_clients=3, num_rounds=5),
        (),
        id="swarm",
    ),
    pytest.param(
        _PT_SYNC_CIFAR10_ROOT,
        "fedavg",
        _pt_sync_cifar10_args("/tmp/nvflare/collab/pt_sync_cifar10/fedavg"),
        ("torch", "torchvision"),
        id="pt_sync_cifar10_fedavg",
    ),
    pytest.param(
        _PT_SYNC_CIFAR10_ROOT,
        "fedprox",
        _pt_sync_cifar10_args("/tmp/nvflare/collab/pt_sync_cifar10/fedprox"),
        ("torch", "torchvision"),
        id="pt_sync_cifar10_fedprox",
    ),
    pytest.param(
        _PT_SYNC_CIFAR10_ROOT,
        "scaffold",
        _pt_sync_cifar10_args("/tmp/nvflare/collab/pt_sync_cifar10/scaffold"),
        ("torch", "torchvision"),
        id="pt_sync_cifar10_scaffold",
    ),
]


@pytest.mark.parametrize("example_root,module_name,args,required_modules", _EXAMPLES)
def test_example_recipe_finalizes(monkeypatch, example_root, module_name, args, required_modules):
    for required_module in required_modules:
        pytest.importorskip(required_module)

    monkeypatch.syspath_prepend(str(example_root))
    module = importlib.import_module(module_name)
    recipe = module.make_recipe(args) if args is not None else module.make_recipe()

    job = recipe.finalize()

    assert recipe.finalize() is job


def test_hello_fedavg_sim_env_uses_configured_clients(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(_ADVANCED_EXAMPLES_ROOT))
    module = importlib.import_module("collab.hello_fedavg.hello_fedavg")
    recipe = module.make_recipe(SimpleNamespace(num_clients=2, num_rounds=3))

    env = module.make_env(recipe)

    assert env.clients == recipe.configured_sites() == ["site-1", "site-2"]


@pytest.mark.parametrize(
    "args,error",
    [
        (SimpleNamespace(num_clients=1, num_rounds=5), "at least 2 clients"),
        (SimpleNamespace(num_clients=3, num_rounds=0), "at least 1 round"),
    ],
)
def test_swarm_rejects_invalid_topology(monkeypatch, args, error):
    monkeypatch.syspath_prepend(str(_ADVANCED_EXAMPLES_ROOT))
    module = importlib.import_module("collab.swarm.swarm")

    with pytest.raises(ValueError, match=error):
        module.make_recipe(args)
