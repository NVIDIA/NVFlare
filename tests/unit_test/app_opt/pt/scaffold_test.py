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

"""Tests for the PyTorch SCAFFOLD helper."""

import copy
from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nvflare.app_opt.pt.scaffold import PTScaffoldHelper


class _StateDictSnapshot:
    def __init__(self, model):
        self._state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    def state_dict(self):
        return self._state_dict


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_terms_update_returns_float32_numpy_control_delta(dtype):
    model = torch.nn.Linear(2, 1, bias=False).to(dtype=dtype)
    with torch.no_grad():
        model.weight.fill_(1)
    model_global = copy.deepcopy(model)

    helper = PTScaffoldHelper()
    helper.init(model)
    c_global_para, c_local_para = helper.get_params()
    with torch.no_grad():
        model.weight.add_(1)
    helper.model_update(model, curr_lr=1.0, c_global_para=c_global_para, c_local_para=c_local_para)
    helper.terms_update(
        model=model,
        curr_lr=1.0,
        c_global_para=c_global_para,
        c_local_para=c_local_para,
        model_global=model_global,
    )

    actual = helper.get_delta_controls()["weight"]
    expected = (model_global.state_dict()["weight"] - model.state_dict()["weight"]).cpu().to(torch.float32).numpy()

    assert isinstance(actual, np.ndarray)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected)
    assert helper.c_local.state_dict()["weight"].dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_terms_update_accepts_cpu_state_snapshot(dtype):
    model = torch.nn.Linear(2, 1, bias=False).to(dtype=dtype)
    with torch.no_grad():
        model.weight.fill_(1)
    model_global = _StateDictSnapshot(model)

    helper = PTScaffoldHelper()
    helper.init(model)
    c_global_para, c_local_para = helper.get_params()
    with torch.no_grad():
        model.weight.add_(1)
    helper.model_update(model, curr_lr=1.0, c_global_para=c_global_para, c_local_para=c_local_para)
    helper.terms_update(
        model=model,
        curr_lr=1.0,
        c_global_para=c_global_para,
        c_local_para=c_local_para,
        model_global=model_global,
    )

    actual = helper.get_delta_controls()["weight"]
    expected = (model_global.state_dict()["weight"] - model.state_dict()["weight"]).to(torch.float32).numpy()
    np.testing.assert_allclose(actual, expected)


def test_scaffold_updates_only_trainable_parameters_and_leaves_batch_norm_buffers_to_model_aggregation():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
    model_global = copy.deepcopy(model)
    helper = PTScaffoldHelper()
    helper.init(model)
    c_global_para, c_local_para = helper.get_params()

    model.train()
    model(torch.randn(8, 4))
    buffers_after_forward = {key: value.detach().clone() for key, value in model.named_buffers()}
    with patch.object(model, "load_state_dict", wraps=model.load_state_dict) as load_state_dict:
        helper.model_update(
            model=model,
            curr_lr=0.1,
            c_global_para=c_global_para,
            c_local_para=c_local_para,
        )

    load_state_dict.assert_not_called()
    for key, value in model.named_buffers():
        assert torch.equal(value, buffers_after_forward[key])

    helper.terms_update(
        model=model,
        curr_lr=0.1,
        c_global_para=c_global_para,
        c_local_para=c_local_para,
        model_global=model_global,
    )

    delta_controls = helper.get_delta_controls()
    trainable_keys = {key for key, parameter in model.named_parameters() if parameter.requires_grad}
    assert set(delta_controls) == trainable_keys

    server_controls = {
        key: np.zeros_like(value.detach().cpu().numpy()) for key, value in model_global.state_dict().items()
    }
    for key, delta in delta_controls.items():
        server_controls[key] += delta
    assert server_controls["1.num_batches_tracked"].dtype == np.int64
    assert server_controls["1.num_batches_tracked"] == 0
