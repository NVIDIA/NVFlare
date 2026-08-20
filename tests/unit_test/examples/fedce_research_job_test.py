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
import json
import os
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from nvflare.app_common.abstract.fl_model import ParamsType
from nvflare.app_opt.pt.recipes import FedCERecipe

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_DIR = ROOT / "research" / "fed-ce" / "jobs" / "fedce_prostate"


@pytest.fixture
def job_module():
    spec = importlib.util.spec_from_file_location("fedce_research_job", EXAMPLE_DIR / "job.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    original_model = sys.modules.pop("model", None)
    sys.path.insert(0, str(EXAMPLE_DIR))
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.pop(0)
        if original_model is not None:
            sys.modules["model"] = original_model
        else:
            sys.modules.pop("model", None)


def _args(data_root="/tmp/fedce data"):
    return SimpleNamespace(
        batch_size=8,
        cache_rate=1.0,
        data_root=data_root,
        fedce_mode="plus",
        learning_rate=1e-3,
        local_epochs=1,
        num_rounds=2,
        num_workers=0,
        seed=0,
    )


def test_fedce_research_job_builds_recipe_and_tracking(monkeypatch, job_module):
    tracking_calls = []
    monkeypatch.setattr(
        job_module,
        "add_experiment_tracking",
        lambda recipe, tracking_type, tracking_config: tracking_calls.append((recipe, tracking_type, tracking_config)),
    )

    recipe = job_module.build_recipe(_args())

    assert isinstance(recipe, FedCERecipe)
    assert recipe.name == "fedce_prostate"
    assert recipe.min_clients == len(job_module.CLIENTS) == 6
    assert recipe.num_rounds == 2
    assert recipe.params_transfer_type.value == ParamsType.DIFF.value
    assert recipe.trainable_param_names == [
        name for name, parameter in recipe.model.named_parameters() if parameter.requires_grad
    ]
    assert tracking_calls == [(recipe, "tensorboard", {"tb_folder": "tb_events"})]


def test_fedce_research_job_quotes_client_arguments(job_module):
    train_args = shlex.split(job_module._build_train_args(_args()))

    assert train_args == [
        "--data-root",
        str(Path("/tmp/fedce data").resolve()),
        "--batch-size",
        "8",
        "--cache-rate",
        "1.0",
        "--learning-rate",
        "0.001",
        "--local-epochs",
        "1",
        "--num-workers",
        "0",
        "--seed",
        "0",
    ]


def test_fedce_research_job_preserves_documented_defaults(job_module):
    args = job_module._parse_args(["--data-root", "/tmp/data"])

    assert args.num_rounds == 100
    assert args.num_threads == len(job_module.CLIENTS)
    assert args.gpu_config == "0,1,0,1,0,1"
    assert args.local_epochs == 1


def test_fedce_research_unet_preserves_spatial_shape(job_module):
    model = job_module.UNet(in_channels=1, out_channels=1, init_features=4)

    # Codex's ARM sandbox advertises CPU features that its virtualized MKLDNN
    # ConvTranspose kernel cannot execute. This assertion validates the model's
    # shape contract, so use PyTorch's portable fallback in that sandbox only.
    if os.environ.get("SANDBOX_VM_ID"):
        with torch.backends.mkldnn.flags(enabled=False):
            output = model(torch.randn(2, 1, 32, 32))
    else:
        output = model(torch.randn(2, 1, 32, 32))

    assert output.shape == (2, 1, 32, 32)


def test_fedce_research_job_exports_model_configuration(job_module, tmp_path):
    recipe = job_module.build_recipe(_args())

    recipe.export(str(tmp_path))

    job_root = tmp_path / "fedce_prostate"
    server_config = json.loads((job_root / "app" / "config" / "config_fed_server.json").read_text())
    controller = server_config["workflows"][0]
    assert controller["args"]["aggregator"]["path"].endswith("FedCEModelAggregator")
    persistor = next(component for component in server_config["components"] if component["id"] == "persistor")
    assert persistor["args"]["model"] == {
        "path": "model.UNet",
        "args": {"in_channels": 1},
    }
    receiver = next(component for component in server_config["components"] if component["id"] == "receiver")
    assert receiver["path"].endswith("TBAnalyticsReceiver")
    assert (job_root / "app" / "custom" / "client.py").is_file()
    assert (job_root / "app" / "custom" / "model.py").is_file()
