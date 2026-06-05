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
from enum import Enum

import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_SAFETENSORS = importlib.util.find_spec("safetensors") is not None


class _ExamplePeftType(Enum):
    LORA = "LORA"


def _example_dir():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "integration", "nemo", "examples", "peft")
    )


def _load_example_module(module_name: str):
    example_dir = _example_dir()
    sys.path.insert(0, example_dir)
    try:
        spec = importlib.util.spec_from_file_location(module_name, os.path.join(example_dir, f"{module_name}.py"))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(example_dir)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for PEFT adapter checkpoint tests")
def test_adapter_checkpoint_round_trip(tmp_path):
    adapter_checkpoint = _load_example_module("adapter_checkpoint")
    import torch

    state = {
        "model.layers.0.self_attn.q_proj.lora_A.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones((3, 2), dtype=torch.bfloat16),
    }
    adapter_config = {"base_model_name_or_path": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "r": 8}
    ckpt_path = tmp_path / "adapter.pt"

    adapter_checkpoint.save_nvflare_adapter_checkpoint(state, str(ckpt_path), adapter_config=adapter_config)
    loaded = adapter_checkpoint.load_adapter_state(str(ckpt_path))

    assert loaded.keys() == state.keys()
    for key, value in state.items():
        assert torch.equal(loaded[key], value.cpu())
    assert adapter_checkpoint.load_adapter_config(str(ckpt_path)) == adapter_config

    stripped = adapter_checkpoint.strip_model_prefix(loaded)
    assert "layers.0.self_attn.q_proj.lora_A.weight" in stripped
    prefixed = adapter_checkpoint.add_model_prefix(stripped)
    assert prefixed.keys() == loaded.keys()

    assert (
        adapter_checkpoint.canonical_adapter_key("model.base_model.model.layers.0.self_attn.q_proj.lora_A.weight")
        == "layers.0.self_attn.q_proj.lora_A.weight"
    )
    matched = adapter_checkpoint.match_adapter_state_to_reference(
        {"layers.0.self_attn.q_proj.lora_A.weight": torch.ones((2, 3))},
        {"base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros((2, 3))},
    )
    assert list(matched) == ["base_model.model.layers.0.self_attn.q_proj.lora_A.weight"]


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for PEFT adapter checkpoint tests")
def test_adapter_checkpoint_metadata_is_weights_only_safe(tmp_path):
    adapter_checkpoint = _load_example_module("adapter_checkpoint")
    import torch

    ckpt_path = tmp_path / "adapter_with_enum_config.pt"
    adapter_checkpoint.save_nvflare_adapter_checkpoint(
        {"model.layer.lora_A.weight": torch.zeros((2, 2))},
        str(ckpt_path),
        adapter_config={"peft_type": _ExamplePeftType.LORA, "target_modules": {"linear"}},
    )

    loaded = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)

    assert loaded["adapter_config"] == {"peft_type": "LORA", "target_modules": ["linear"]}
    assert adapter_checkpoint.load_adapter_config(str(ckpt_path)) == loaded["adapter_config"]


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for PEFT adapter checkpoint tests")
def test_client_builds_full_adapter_update():
    import torch

    automodel_peft_client = _load_example_module("automodel_peft_client")

    params_type, params = automodel_peft_client._build_param_update(
        {"model.layer.lora_A.weight": torch.full((2, 2), 0.2)},
        {"model.layer.lora_A.weight": torch.full((2, 2), 0.5)},
        torch.device("cpu"),
    )

    assert params_type == automodel_peft_client.flare.ParamsType.FULL
    assert params["model.layer.lora_A.weight"].device.type == "cpu"
    assert torch.equal(params["model.layer.lora_A.weight"], torch.full((2, 2), 0.5))


@pytest.mark.skipif(not HAS_TORCH or not HAS_SAFETENSORS, reason="PyTorch and safetensors are required")
def test_fl_model_serializes_lora_tensors_with_nvflare_fobs():
    import torch

    from nvflare.apis.fl_constant import FLMetaKey
    from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
    from nvflare.app_common.decomposers import common_decomposers
    from nvflare.app_opt.pt.decomposers import TensorDecomposer
    from nvflare.fuel.utils import fobs

    common_decomposers.register()
    fobs.register(TensorDecomposer)

    model = FLModel(
        params_type=ParamsType.FULL,
        params={"model.layer.lora_A.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
        metrics={"loss": 0.25},
        meta={FLMetaKey.NUM_STEPS_CURRENT_ROUND: 3},
    )

    restored = fobs.loads(fobs.dumps(model))
    assert restored.params_type == ParamsType.FULL
    assert torch.equal(restored.params["model.layer.lora_A.weight"], model.params["model.layer.lora_A.weight"])
    assert restored.metrics == {"loss": 0.25}
    assert restored.meta[FLMetaKey.NUM_STEPS_CURRENT_ROUND] == 3


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for adapter loader tests")
def test_automodel_adapter_loader_keeps_matching_tensors_on_target_dtype_and_device():
    import torch

    automodel_adapter_loader = _load_example_module("automodel_adapter_loader")

    model_state = {
        "layer.lora_A.weight": torch.zeros((2, 2), dtype=torch.bfloat16),
        "layer.lora_B.weight": torch.zeros((2, 2), dtype=torch.float32),
    }
    adapter_state = {
        "base_model.model.layer.lora_A.weight": torch.ones((2, 2), dtype=torch.float32),
        "extra.lora_A.weight": torch.ones((2, 2), dtype=torch.float32),
    }

    compatible = automodel_adapter_loader._compatible_adapter_state(model_state, adapter_state)

    assert list(compatible) == ["layer.lora_A.weight"]
    assert compatible["layer.lora_A.weight"].dtype == torch.bfloat16
    assert compatible["layer.lora_A.weight"].device == model_state["layer.lora_A.weight"].device
