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
import logging
import os
from types import SimpleNamespace

import pytest

from nvflare.app_common.abstract.fl_model import FLModel

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def _load_hf_utils():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    utils_path = os.path.join(repo_root, "nvflare", "app_opt", "hf", "utils.py")
    spec = importlib.util.spec_from_file_location("nvflare_app_opt_hf_utils_under_test", utils_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


hf_utils = _load_hf_utils()


class _Trainer:
    def __init__(self, model, accelerator=None):
        self.model = model
        self.accelerator = accelerator


class _Accelerator:
    def __init__(self, unwrapped=None, state_dict=None):
        self.unwrapped = unwrapped
        self.state_dict = state_dict
        self.unwrap_called = False
        self.get_state_dict_called = False

    def unwrap_model(self, model):
        self.unwrap_called = True
        return self.unwrapped or model

    def get_state_dict(self, model):
        self.get_state_dict_called = True
        return self.state_dict or model.state_dict()


def test_import_does_not_require_hf_optional_dependencies():
    assert hf_utils.PARAMS_SCOPE_AUTO == "auto"
    assert hf_utils.resolve_params_scope(_Trainer(SimpleNamespace(state_dict=lambda: {})), "auto") == "model"


def test_unwrap_model_uses_accelerator_when_available():
    raw = object()
    wrapped = object()
    accelerator = _Accelerator(unwrapped=raw)

    assert hf_utils.unwrap_model(_Trainer(wrapped, accelerator)) is raw
    assert accelerator.unwrap_called


def test_server_key_prefix_strip_apply_and_collision_detection():
    params = {"model.weight": 1, "bias": 2}

    assert hf_utils.strip_server_key_prefix(params, "model.") == {"weight": 1, "bias": 2}
    assert hf_utils.apply_server_key_prefix({"weight": 1}, "model.") == {"model.weight": 1}
    assert params == {"model.weight": 1, "bias": 2}

    with pytest.raises(ValueError, match="duplicate"):
        hf_utils.strip_server_key_prefix({"model.weight": 1, "weight": 2}, "model.")


def test_resolve_params_scope_rejects_invalid_or_adapter_without_peft():
    trainer = _Trainer(SimpleNamespace(state_dict=lambda: {}))

    with pytest.raises(ValueError, match="params_scope"):
        hf_utils.resolve_params_scope(trainer, "bad")
    with pytest.raises(ValueError, match="requires"):
        hf_utils.resolve_params_scope(trainer, "adapter")


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for parameter helper tests")
def test_load_params_validates_prefixes_and_loads_full_model():
    import torch

    model = torch.nn.Linear(2, 1)
    trainer = _Trainer(model)
    params = {
        "server.weight": torch.full_like(model.weight, 3.0),
        "server.bias": torch.full_like(model.bias, 4.0),
    }

    report = hf_utils.load_params(
        trainer,
        FLModel(params=params),
        params_scope="model",
        strict=True,
        server_key_prefix="server.",
    )

    assert report.matched_keys == ("bias", "weight")
    assert torch.equal(model.weight, torch.full_like(model.weight, 3.0))
    assert torch.equal(model.bias, torch.full_like(model.bias, 4.0))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for parameter helper tests")
def test_load_params_filters_unexpected_keys_in_non_strict_mode(caplog):
    import torch

    model = torch.nn.Linear(2, 1)
    trainer = _Trainer(model)
    params = {"weight": torch.ones_like(model.weight), "unexpected": torch.zeros(1)}

    with caplog.at_level(logging.WARNING, logger=hf_utils.__name__):
        report = hf_utils.load_params(trainer, FLModel(params=params), params_scope="model", strict=False)

    assert report.unexpected_keys == ("unexpected",)
    assert "Ignoring 1 unexpected model parameter" in caplog.text
    assert torch.equal(model.weight, torch.ones_like(model.weight))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for parameter helper tests")
def test_load_params_fails_on_shape_mismatch_and_zero_match():
    import torch

    model = torch.nn.Linear(2, 1)
    trainer = _Trainer(model)

    with pytest.raises(RuntimeError, match="shape mismatch"):
        hf_utils.load_params(trainer, FLModel(params={"weight": torch.ones(1)}), params_scope="model")

    with pytest.raises(RuntimeError, match="None of the"):
        hf_utils.load_params(trainer, FLModel(params={"model.weight": torch.ones_like(model.weight)}), "model")


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for parameter helper tests")
def test_extract_params_uses_accelerator_state_dict_and_keeps_cpu_tensors():
    import torch

    model = torch.nn.Linear(2, 1)
    state_dict = {"weight": torch.full_like(model.weight, 5.0)}
    accelerator = _Accelerator(state_dict=state_dict)

    params = hf_utils.extract_params(_Trainer(model, accelerator), "model")

    assert accelerator.get_state_dict_called
    assert params["weight"].device.type == "cpu"
    assert torch.equal(params["weight"], torch.full_like(model.weight, 5.0))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for PEFT parameter helper tests")
def test_adapter_scope_uses_peft_adapter_keyspace(monkeypatch):
    import torch

    class FakePeftModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2, 2))
            self.adapter_state = {"lora_A.weight": torch.zeros(2, 2)}
            self.loaded_adapter_state = None

    FakePeftModel.__name__ = "PeftModel"
    FakePeftModel.__module__ = "peft.fake"

    class FakePeftModule:
        @staticmethod
        def get_peft_model_state_dict(model):
            return model.adapter_state

        @staticmethod
        def set_peft_model_state_dict(model, params):
            model.loaded_adapter_state = params

    monkeypatch.setattr(hf_utils, "_import_peft", lambda reason="": FakePeftModule)

    model = FakePeftModel()
    trainer = _Trainer(model)

    assert hf_utils.is_peft_model(model)
    assert hf_utils.resolve_params_scope(trainer, "auto") == "adapter"
    reference_state = hf_utils.get_reference_state_dict(trainer, "adapter")
    assert list(reference_state) == ["lora_A.weight"]
    assert torch.equal(reference_state["lora_A.weight"], torch.zeros(2, 2))

    hf_utils.load_params(
        trainer,
        FLModel(params={"server.lora_A.weight": torch.ones(2, 2)}),
        params_scope="adapter",
        server_key_prefix="server.",
    )

    assert torch.equal(model.loaded_adapter_state["lora_A.weight"], torch.ones(2, 2))
    extracted = hf_utils.extract_params(trainer, "adapter")
    assert list(extracted) == ["lora_A.weight"]
    assert torch.equal(extracted["lora_A.weight"], torch.zeros(2, 2))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for PEFT parameter helper tests")
def test_strict_adapter_load_rejects_missing_reference_keys(monkeypatch):
    import torch

    class FakePeftModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter_state = {
                "lora_A.weight": torch.zeros(2, 2),
                "lora_B.weight": torch.zeros(2, 2),
            }

    FakePeftModel.__name__ = "PeftModel"
    FakePeftModel.__module__ = "peft.fake"

    class FakePeftModule:
        @staticmethod
        def get_peft_model_state_dict(model):
            return model.adapter_state

        @staticmethod
        def set_peft_model_state_dict(model, params):
            raise AssertionError("incomplete adapter params must be rejected before PEFT loading")

    monkeypatch.setattr(hf_utils, "_import_peft", lambda reason="": FakePeftModule)
    trainer = _Trainer(FakePeftModel())

    with pytest.raises(RuntimeError, match=r"incomplete PEFT adapter parameters.*lora_B\.weight"):
        hf_utils.load_params(
            trainer,
            FLModel(params={"lora_A.weight": torch.ones(2, 2)}),
            params_scope="adapter",
            strict=True,
        )


@pytest.mark.skipif(not HAS_TORCH or not HAS_NUMPY, reason="PyTorch and NumPy are required for dtype tests")
def test_prepare_out_params_casts_halves_for_numpy_and_preserves_tensor_dtype():
    import numpy as np
    import torch

    params = {
        "fp16": torch.ones(2, dtype=torch.float16),
        "bf16": torch.ones(2, dtype=torch.bfloat16),
        "fp32": torch.ones(2, dtype=torch.float32),
    }

    numpy_params = hf_utils.prepare_out_params(params, "numpy")
    assert numpy_params["fp16"].dtype == np.float32
    assert numpy_params["bf16"].dtype == np.float32
    assert numpy_params["fp32"].dtype == np.float32

    torch_params = hf_utils.prepare_out_params(params, "pytorch")
    assert torch_params["fp16"].dtype == torch.float16
    assert torch_params["bf16"].dtype == torch.bfloat16
    assert torch_params["fp16"].device.type == "cpu"


@pytest.mark.skipif(not HAS_TORCH or not HAS_NUMPY, reason="PyTorch and NumPy are required for dtype tests")
def test_prepare_out_params_casts_halves_when_exchange_format_is_numpy_even_for_pytorch_server():
    import numpy as np
    import torch

    params = {
        "bf16": torch.ones(2, dtype=torch.bfloat16),
        "fp16": torch.ones(2, dtype=torch.float16),
    }

    numpy_params = hf_utils.prepare_out_params(params, "numpy", server_expected_format="pytorch")

    assert numpy_params["bf16"].dtype == np.float32
    assert numpy_params["fp16"].dtype == np.float32


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for dtype tests")
def test_prepare_out_params_casts_halves_for_numpy_server_while_preserving_pytorch_exchange():
    import torch

    params = {
        "bf16": torch.ones(2, dtype=torch.bfloat16),
        "fp16": torch.ones(2, dtype=torch.float16),
        "fp32": torch.ones(2, dtype=torch.float32),
    }

    torch_params = hf_utils.prepare_out_params(params, "pytorch", server_expected_format="numpy")

    assert torch.is_tensor(torch_params["bf16"])
    assert torch.is_tensor(torch_params["fp16"])
    assert torch_params["bf16"].dtype == torch.float32
    assert torch_params["fp16"].dtype == torch.float32
    assert torch_params["fp32"].dtype == torch.float32


def test_total_train_steps_prefers_max_steps_and_estimates_epoch_budget(monkeypatch):
    assert hf_utils.total_train_steps(999, SimpleNamespace(max_steps=7), total_rounds=3) == 21

    args = SimpleNamespace(
        max_steps=-1,
        num_train_epochs=1.5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        world_size=2,
    )
    assert hf_utils.total_train_steps(65, args, total_rounds=3) == 24

    monkeypatch.setenv("WORLD_SIZE", "4")
    args = SimpleNamespace(
        max_steps=0,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
    )
    assert hf_utils.total_train_steps(33, args, total_rounds=2) == 6


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for checkpoint parameter tests")
def test_extract_params_from_checkpoint_reads_known_hf_weight_files(tmp_path):
    import torch

    checkpoint_dir = tmp_path / "checkpoint-3"
    checkpoint_dir.mkdir()
    expected = {"weight": torch.ones(2, 2), "bias": torch.zeros(2)}
    torch.save(expected, checkpoint_dir / "pytorch_model.bin")

    params = hf_utils.extract_params_from_checkpoint(checkpoint_dir)

    assert torch.equal(params["weight"], expected["weight"])
    assert torch.equal(params["bias"], expected["bias"])


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for checkpoint parameter tests")
def test_extract_params_from_checkpoint_reads_sharded_safetensors_index(tmp_path):
    import torch

    safetensors = pytest.importorskip("safetensors.torch")
    checkpoint_dir = tmp_path / "checkpoint-3"
    checkpoint_dir.mkdir()
    first = {"layer1.weight": torch.ones(2, 2)}
    second = {"layer2.weight": torch.full((2, 2), 2.0)}
    safetensors.save_file(first, checkpoint_dir / "model-00001-of-00002.safetensors")
    safetensors.save_file(second, checkpoint_dir / "model-00002-of-00002.safetensors")
    with open(checkpoint_dir / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {"total_size": 32},
                "weight_map": {
                    "layer1.weight": "model-00001-of-00002.safetensors",
                    "layer2.weight": "model-00002-of-00002.safetensors",
                },
            },
            f,
        )

    params = hf_utils.extract_params_from_checkpoint(checkpoint_dir)

    assert torch.equal(params["layer1.weight"], first["layer1.weight"])
    assert torch.equal(params["layer2.weight"], second["layer2.weight"])


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for params exchange tests")
def test_params_exchange_file_round_trips_tensors_and_cleans_up(tmp_path):
    import torch

    params = {"weight": torch.ones(2, 2)}
    descriptor = hf_utils.write_params_exchange_file(tmp_path, params)

    assert os.path.exists(descriptor["path"])
    loaded = hf_utils.read_params_exchange_file(descriptor)
    assert torch.equal(loaded["weight"], params["weight"])

    hf_utils.cleanup_params_exchange_file(descriptor)
    assert not os.path.exists(descriptor["path"])


@pytest.mark.skipif(not HAS_TORCH or not HAS_NUMPY, reason="PyTorch and NumPy are required for params exchange tests")
def test_params_exchange_file_round_trips_numpy_arrays_as_tensors(tmp_path):
    import numpy as np
    import torch

    params = {"weight": np.ones((2, 2), dtype=np.float32), "bias": np.zeros(2, dtype=np.float32)}
    descriptor = hf_utils.write_params_exchange_file(tmp_path, params)

    loaded = hf_utils.read_params_exchange_file(descriptor)

    assert torch.equal(loaded["weight"], torch.ones(2, 2))
    assert torch.equal(loaded["bias"], torch.zeros(2))

    hf_utils.cleanup_params_exchange_file(descriptor)
    assert not os.path.exists(descriptor["path"])
