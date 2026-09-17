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

torch = pytest.importorskip("torch")

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnableKey, make_model_learnable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_opt.pt.fedopt import PTFedOptModelShareableGenerator


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 1)
        self.register_buffer("offset", torch.zeros(1))

    def forward(self, x):
        return self.lin(x)


@pytest.mark.filterwarnings("ignore:To copy construct from a tensor:UserWarning")
def test_fedopt_shareable_generator_preserves_torch_weights_when_base_model_uses_tensors():
    model = SimpleModel()
    generator = PTFedOptModelShareableGenerator(
        optimizer_args={"path": "torch.optim.SGD", "args": {"lr": 1.0}},
        device="cpu",
    )
    generator.model = model
    generator.device = torch.device("cpu")
    generator.optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    generator.optimizer_name = "torch.optim.SGD"

    fl_ctx = FLContext()
    base_weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
    fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, make_model_learnable(base_weights, {}), private=True, sticky=True)
    fl_ctx.set_prop(AppConstants.CURRENT_ROUND, 0, private=True, sticky=False)

    model_diff = {name: torch.ones_like(value) for name, value in base_weights.items()}
    shareable = DXO(data_kind=DataKind.WEIGHT_DIFF, data=model_diff).to_shareable()

    learnable = generator.shareable_to_learnable(shareable, fl_ctx)
    weights = learnable[ModelLearnableKey.WEIGHTS]

    assert weights
    model_state = generator.model.state_dict()
    for name, value in weights.items():
        assert isinstance(value, torch.Tensor)
        assert value.device.type == "cpu"
        assert value.data_ptr() != model_state[name].data_ptr()
        assert torch.allclose(value, base_weights[name] + model_diff[name])


def _make_generator(model):
    generator = PTFedOptModelShareableGenerator(
        optimizer_args={"path": "torch.optim.SGD", "args": {"lr": 1.0}},
        device="cpu",
    )
    generator.model = model
    generator.device = torch.device("cpu")
    generator.optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    generator.optimizer_name = "torch.optim.SGD"
    return generator


def _empty_global_model_fl_ctx():
    fl_ctx = FLContext()
    fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, make_model_learnable({}, {}), private=True, sticky=True)
    fl_ctx.set_prop(AppConstants.CURRENT_ROUND, 0, private=True, sticky=False)
    return fl_ctx


@pytest.mark.filterwarnings("ignore:To copy construct from a tensor:UserWarning")
def test_fedopt_buffer_update_falls_back_to_model_state_when_base_weights_empty():
    model = SimpleModel()
    generator = _make_generator(model)
    fl_ctx = _empty_global_model_fl_ctx()

    model_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model_diff = {name: torch.ones_like(value) for name, value in model_state.items()}
    shareable = DXO(data_kind=DataKind.WEIGHT_DIFF, data=model_diff).to_shareable()

    learnable = generator.shareable_to_learnable(shareable, fl_ctx)
    weights = learnable[ModelLearnableKey.WEIGHTS]

    assert isinstance(weights["offset"], torch.Tensor)
    assert torch.allclose(weights["offset"], model_state["offset"] + model_diff["offset"])


def test_fedopt_numpy_buffer_update_falls_back_to_model_state_when_base_weights_empty():
    model = SimpleModel()
    generator = _make_generator(model)
    fl_ctx = _empty_global_model_fl_ctx()

    model_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model_diff = {name: np.ones_like(value.numpy()) for name, value in model_state.items()}
    shareable = DXO(data_kind=DataKind.WEIGHT_DIFF, data=model_diff).to_shareable()

    learnable = generator.shareable_to_learnable(shareable, fl_ctx)
    weights = learnable[ModelLearnableKey.WEIGHTS]

    assert isinstance(weights["offset"], np.ndarray)
    assert np.allclose(weights["offset"], model_state["offset"].numpy() + model_diff["offset"])
