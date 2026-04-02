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

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_opt.lightning.api import FLCallback


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)


def _make_callback(load_state_dict_strict: bool = False) -> FLCallback:
    with (
        patch("nvflare.app_opt.lightning.api.init"),
        patch("nvflare.app_opt.lightning.api.get_config", return_value={}),
    ):
        return FLCallback(rank=0, load_state_dict_strict=load_state_dict_strict)


def _clone_state_dict(model: nn.Module) -> dict:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def test_receive_and_update_model_raises_when_no_keys_match_and_suggests_prefix():
    callback = _make_callback()
    module = SimpleNet()
    params = {
        "model.fc.weight": torch.ones_like(module.state_dict()["fc.weight"]),
        "model.fc.bias": torch.zeros_like(module.state_dict()["fc.bias"]),
    }
    callback._receive_model = lambda trainer: FLModel(params=params)

    with pytest.raises(RuntimeError, match="stripping common prefix 'model\\.'"):
        callback._receive_and_update_model(SimpleNamespace(), module)


def test_receive_and_update_model_raises_on_shape_mismatch():
    callback = _make_callback()
    module = SimpleNet()
    params = _clone_state_dict(module)
    params["fc.weight"] = torch.ones(3, 4)
    callback._receive_model = lambda trainer: FLModel(params=params)

    with pytest.raises(RuntimeError, match=r"fc.weight: expected \(2, 4\), got \(3, 4\)"):
        callback._receive_and_update_model(SimpleNamespace(), module)


def test_receive_and_update_model_warns_on_unexpected_keys_when_some_match(caplog):
    callback = _make_callback()
    module = SimpleNet()
    params = _clone_state_dict(module)
    params["fc.weight"] = torch.full_like(params["fc.weight"], 3.0)
    params["fc.bias"] = torch.full_like(params["fc.bias"], -2.0)
    params["model.fc.weight"] = torch.ones_like(module.state_dict()["fc.weight"])
    callback._receive_model = lambda trainer: FLModel(params=params)

    with caplog.at_level(logging.WARNING):
        callback._receive_and_update_model(SimpleNamespace(), module)

    assert "Ignoring 1 unexpected model parameter" in caplog.text
    assert torch.equal(module.state_dict()["fc.weight"], params["fc.weight"])
    assert torch.equal(module.state_dict()["fc.bias"], params["fc.bias"])


def test_receive_and_update_model_rejects_unexpected_keys_in_strict_mode(caplog):
    callback = _make_callback(load_state_dict_strict=True)
    module = SimpleNet()
    params = _clone_state_dict(module)
    params["model.fc.weight"] = torch.ones_like(module.state_dict()["fc.weight"])
    callback._receive_model = lambda trainer: FLModel(params=params)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError, match="Rejecting 1 unexpected model parameter"):
            callback._receive_and_update_model(SimpleNamespace(), module)

    assert "Ignoring 1 unexpected model parameter" not in caplog.text


def test_receive_and_update_model_logs_missing_keys_for_partial_loads(caplog):
    callback = _make_callback(load_state_dict_strict=False)
    module = SimpleNet()
    params = {"fc.weight": torch.full_like(module.state_dict()["fc.weight"], 4.0)}
    callback._receive_model = lambda trainer: FLModel(params=params)

    with caplog.at_level(logging.WARNING):
        callback._receive_and_update_model(SimpleNamespace(), module)

    assert "There were missing keys when loading the global state_dict: ['fc.bias']" in caplog.text
    assert torch.equal(module.state_dict()["fc.weight"], params["fc.weight"])
