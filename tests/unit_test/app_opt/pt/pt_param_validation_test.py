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

"""Unit tests for PyTorch model param validation during exchange and persistence."""

import logging

import pytest
import torch
import torch.nn as nn

from nvflare.app_common.abstract.model import make_model_learnable
from nvflare.app_opt.pt.model_persistence_format_manager import PTModelPersistenceFormatManager
from nvflare.app_opt.pt.utils import feed_vars


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)


def _clone_state_dict(model: nn.Module) -> dict:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def test_feed_vars_assigns_matching_params():
    model = SimpleNet()
    expected = _clone_state_dict(model)
    expected["fc.weight"] = torch.full_like(expected["fc.weight"], 2.0)
    expected["fc.bias"] = torch.full_like(expected["fc.bias"], -1.0)

    assign_ops, updated_local_model = feed_vars(model, expected)

    assert len(assign_ops) == 2
    assert torch.equal(updated_local_model["fc.weight"], expected["fc.weight"])
    assert torch.equal(updated_local_model["fc.bias"], expected["fc.bias"])


def test_feed_vars_raises_when_no_keys_match_and_suggests_prefix():
    model = SimpleNet()
    params = {
        "model.fc.weight": torch.ones_like(model.state_dict()["fc.weight"]),
        "model.fc.bias": torch.zeros_like(model.state_dict()["fc.bias"]),
    }

    with pytest.raises(
        RuntimeError,
        match=r"None of the 2 incoming model parameter\(s\) matched the local model's 2 parameter\(s\)",
    ):
        feed_vars(model, params)

    with pytest.raises(RuntimeError, match="stripping common prefix 'model\\.'"):
        feed_vars(model, params)

    with pytest.raises(RuntimeError, match=r"Incoming keys: 2 sample=\['model.fc.bias', 'model.fc.weight'\]"):
        feed_vars(model, params)

    with pytest.raises(RuntimeError, match=r"Local keys: 2 sample=\['fc.bias', 'fc.weight'\]"):
        feed_vars(model, params)


def test_feed_vars_raises_on_shape_mismatch():
    model = SimpleNet()
    params = _clone_state_dict(model)
    params["fc.weight"] = torch.ones(3, 4)

    with pytest.raises(RuntimeError, match=r"fc.weight: expected \(2, 4\), got \(3, 4\)"):
        feed_vars(model, params)

    with pytest.raises(RuntimeError, match=r"Incoming keys: 2 sample=\['fc.bias', 'fc.weight'\]"):
        feed_vars(model, params)


def test_feed_vars_warns_on_unexpected_keys_when_some_match(caplog):
    model = SimpleNet()
    params = _clone_state_dict(model)
    params["model.fc.weight"] = torch.ones_like(model.state_dict()["fc.weight"])

    with caplog.at_level(logging.WARNING):
        assign_ops, _ = feed_vars(model, params)

    assert len(assign_ops) == 2
    assert "Ignoring 1 unexpected model parameter" in caplog.text
    assert "Incoming keys: 3 sample=" in caplog.text


def test_persistence_manager_accepts_partial_known_updates():
    model = SimpleNet()
    manager = PTModelPersistenceFormatManager(_clone_state_dict(model))
    new_weight = torch.full_like(model.state_dict()["fc.weight"], 5.0)

    manager.update(make_model_learnable(weights={"fc.weight": new_weight}, meta_props={}))

    assert torch.equal(manager.var_dict["fc.weight"], new_weight)
    assert torch.equal(manager.var_dict["fc.bias"], model.state_dict()["fc.bias"])


def test_persistence_manager_rejects_unexpected_keys():
    model = SimpleNet()
    manager = PTModelPersistenceFormatManager(_clone_state_dict(model))
    weights = {
        "fc.weight": torch.full_like(model.state_dict()["fc.weight"], 5.0),
        "model.fc.bias": torch.zeros_like(model.state_dict()["fc.bias"]),
    }

    with pytest.raises(ValueError, match="Rejecting 1 unexpected model parameter"):
        manager.update(make_model_learnable(weights=weights, meta_props={}))
