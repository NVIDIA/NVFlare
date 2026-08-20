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
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

HAS_PT_DEPS = all(importlib.util.find_spec(dep) is not None for dep in ("torch", "torchvision"))
pytestmark = pytest.mark.skipif(not HAS_PT_DEPS, reason="PyTorch example dependencies are not installed")


def _load_client_module():
    repo_root = Path(__file__).parents[3]
    source_dir = repo_root / "examples" / "advanced" / "cifar10" / "pt" / "src"
    module_path = (
        repo_root / "examples" / "advanced" / "cifar10" / "pt" / "cifar10-sim" / "cifar10_fedprox" / "client.py"
    )
    spec = importlib.util.spec_from_file_location("cifar10_fedprox_client", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    original_modules = {name: sys.modules.pop(name, None) for name in ("data", "model", "train_utils")}
    sys.path.insert(0, str(source_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
        for name, original_module in original_modules.items():
            if original_module is not None:
                sys.modules[name] = original_module
            else:
                sys.modules.pop(name, None)
    return module


def _args():
    return SimpleNamespace(
        aggregation_epochs=1,
        batch_size=2,
        cosine_lr_eta_min_factor=0.01,
        evaluate_local=False,
        lr=0.01,
        no_lr_scheduler=True,
        num_workers=0,
        train_idx_root="/unused",
    )


def test_evaluate_task_does_not_require_fedprox_metadata(monkeypatch):
    module = _load_client_module()
    model = module.ModerateCNN()
    input_model = module.flare.FLModel(params=model.state_dict(), meta={})
    send = Mock()
    get_fedprox_mu = Mock(side_effect=AssertionError("FedProx metadata must not be read for evaluation"))
    fedprox_loss = Mock(side_effect=AssertionError("FedProx loss must not be created for evaluation"))

    monkeypatch.setattr(module, "ModerateCNN", lambda: model)
    monkeypatch.setattr(module, "create_datasets", lambda *args, **kwargs: (object(), object()))
    monkeypatch.setattr(module, "create_data_loaders", lambda *args, **kwargs: (object(), object()))
    monkeypatch.setattr(module, "evaluate", lambda *args, **kwargs: 0.75)
    monkeypatch.setattr(module, "get_fedprox_mu", get_fedprox_mu)
    monkeypatch.setattr(module, "PTFedProxLoss", fedprox_loss)
    monkeypatch.setattr(module, "SummaryWriter", Mock)
    monkeypatch.setattr(module.flare, "init", Mock())
    monkeypatch.setattr(module.flare, "get_site_name", Mock(return_value="site-1"))
    monkeypatch.setattr(module.flare, "is_running", Mock(side_effect=[True, False]))
    monkeypatch.setattr(module.flare, "receive", Mock(return_value=input_model))
    monkeypatch.setattr(module.flare, "is_evaluate", Mock(return_value=True))
    monkeypatch.setattr(module.flare, "is_submit_model", Mock(return_value=False))
    monkeypatch.setattr(module.flare, "is_train", Mock(return_value=False))
    monkeypatch.setattr(module.flare, "send", send)

    module.main(_args())

    get_fedprox_mu.assert_not_called()
    fedprox_loss.assert_not_called()
    output_model = send.call_args.args[0]
    assert output_model.metrics == {"accuracy": 0.75}
    assert output_model.params is None


def test_submit_task_does_not_require_fedprox_metadata(monkeypatch):
    module = _load_client_module()
    get_fedprox_mu = Mock(side_effect=AssertionError("FedProx metadata must not be read for model submission"))
    fedprox_loss = Mock(side_effect=AssertionError("FedProx loss must not be created for model submission"))

    monkeypatch.setattr(module, "create_datasets", lambda *args, **kwargs: (object(), object()))
    monkeypatch.setattr(module, "create_data_loaders", lambda *args, **kwargs: (object(), object()))
    monkeypatch.setattr(module, "get_fedprox_mu", get_fedprox_mu)
    monkeypatch.setattr(module, "PTFedProxLoss", fedprox_loss)
    monkeypatch.setattr(module, "SummaryWriter", Mock)
    monkeypatch.setattr(module.flare, "init", Mock())
    monkeypatch.setattr(module.flare, "get_site_name", Mock(return_value="site-1"))
    monkeypatch.setattr(module.flare, "is_running", Mock(return_value=True))
    monkeypatch.setattr(module.flare, "receive", Mock(return_value=module.flare.FLModel(meta={})))
    monkeypatch.setattr(module.flare, "is_evaluate", Mock(return_value=False))
    monkeypatch.setattr(module.flare, "is_submit_model", Mock(return_value=True))

    with pytest.raises(RuntimeError, match="before completing a training round"):
        module.main(_args())

    get_fedprox_mu.assert_not_called()
    fedprox_loss.assert_not_called()
