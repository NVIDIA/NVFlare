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
        {"model.layer.lora_A.weight": torch.full((2, 2), 0.5)},
        torch.device("cpu"),
    )

    assert params_type == automodel_peft_client.flare.ParamsType.FULL
    assert params["model.layer.lora_A.weight"].device.type == "cpu"
    assert torch.equal(params["model.layer.lora_A.weight"], torch.full((2, 2), 0.5))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for adapter exchange dtype tests")
def test_client_fp32_exchange_is_opt_in_for_nano_and_required_for_lightning():
    import torch

    automodel_peft_client = _load_example_module("automodel_peft_client")
    state = {"model.layer.lora_A.weight": torch.ones((2, 2), dtype=torch.bfloat16)}

    def exchange_state(profile, fp32_adapter_exchange):
        args = type("Args", (), {"model_profile": profile, "fp32_adapter_exchange": fp32_adapter_exchange})()
        return automodel_peft_client._prepare_exchange_state(args, state)

    assert exchange_state("nano", False)["model.layer.lora_A.weight"].dtype == torch.bfloat16
    assert exchange_state("nano", True)["model.layer.lora_A.weight"].dtype == torch.float32
    assert exchange_state("lightning35", False)["model.layer.lora_A.weight"].dtype == torch.float32


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for adapter namespace tests")
def test_client_preserves_nano_mapping_but_requires_exact_lightning_state():
    import torch

    automodel_peft_client = _load_example_module("automodel_peft_client")
    incoming = {"base_model.model.layer.lora_A.weight": torch.zeros((2, 2))}
    updated = {
        "base_model.model.layer.lora_A.weight": torch.ones((2, 2)),
        "lm_head.lora_A.weight": torch.ones((2, 2)),
    }

    nano = automodel_peft_client._align_updated_state_for_exchange(
        type("Args", (), {"model_profile": "nano"})(), updated, incoming
    )
    assert list(nano) == ["base_model.model.layer.lora_A.weight"]

    with pytest.raises(ValueError, match="unexpected=1"):
        automodel_peft_client._align_updated_state_for_exchange(
            type("Args", (), {"model_profile": "lightning35"})(), updated, incoming
        )


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for strict adapter validation tests")
def test_strict_adapter_validation_rejects_partial_unexpected_shape_duplicate_and_nonfinite():
    adapter_checkpoint = _load_example_module("adapter_checkpoint")
    import torch

    reference = {
        "base_model.model.layer.lora_A.weight": torch.zeros((2, 2)),
        "base_model.model.layer.lora_B.weight": torch.zeros((2, 2)),
    }
    with pytest.raises(ValueError, match="missing=1"):
        adapter_checkpoint.align_adapter_state_strict(
            {"layer.lora_A.weight": torch.zeros((2, 2))}, reference, normalize_peft_prefixes=True
        )
    with pytest.raises(ValueError, match="unexpected=1"):
        adapter_checkpoint.align_adapter_state_strict(
            {**reference, "extra.lora_A.weight": torch.zeros((2, 2))}, reference
        )
    with pytest.raises(ValueError, match="shape mismatch"):
        adapter_checkpoint.align_adapter_state_strict(
            {**reference, "base_model.model.layer.lora_A.weight": torch.zeros((3, 2))}, reference
        )
    with pytest.raises(ValueError, match="Duplicate adapter key"):
        adapter_checkpoint.align_adapter_state_strict(
            {
                "layer.lora_A.weight": torch.zeros((2, 2)),
                "base_model.model.layer.lora_A.weight": torch.zeros((2, 2)),
                "layer.lora_B.weight": torch.zeros((2, 2)),
            },
            reference,
            normalize_peft_prefixes=True,
        )
    with pytest.raises(ValueError, match="non-finite"):
        adapter_checkpoint.align_adapter_state_strict(
            {**reference, "base_model.model.layer.lora_A.weight": torch.full((2, 2), float("nan"))},
            reference,
        )


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for adapter manifest tests")
def test_adapter_manifest_rejects_hash_and_profile_conflicts():
    adapter_checkpoint = _load_example_module("adapter_checkpoint")
    import torch

    state = {"layer.lora_A.weight": torch.ones((2, 2))}
    manifest = adapter_checkpoint.build_adapter_manifest(
        state,
        model_profile="lightning35",
        model_name_or_path="model",
        tokenizer_name_or_path="model",
        model_revision="abc",
        tokenizer_revision="abc",
        profile_settings={"lora_rank": 8},
    )
    adapter_checkpoint.validate_adapter_manifest(manifest, state, {"model_profile": "lightning35"})
    with pytest.raises(ValueError, match="hash"):
        adapter_checkpoint.validate_adapter_manifest(manifest, {"layer.lora_A.weight": torch.zeros((2, 2))})
    with pytest.raises(ValueError, match="conflict"):
        adapter_checkpoint.validate_adapter_manifest(manifest, state, {"model_profile": "nano"})


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for adapter contract tests")
def test_adapter_contract_rejects_partial_and_stale_inputs():
    adapter_checkpoint = _load_example_module("adapter_checkpoint")
    import torch

    state = {
        "layer.lora_A.weight": torch.ones((2, 2)),
        "layer.lora_B.weight": torch.zeros((2, 2)),
    }
    contract = adapter_checkpoint.build_adapter_manifest(
        state,
        model_profile="lightning35",
        model_name_or_path="model",
        tokenizer_name_or_path="model",
        model_revision="abc",
        tokenizer_revision="abc",
        profile_settings={"lora_rank": 8},
    )
    adapter_checkpoint.validate_adapter_contract(
        contract,
        state,
        {"model_profile": "lightning35", "base_model_revision": "abc"},
    )
    with pytest.raises(ValueError, match="tensor count"):
        adapter_checkpoint.validate_adapter_contract(contract, {"layer.lora_A.weight": state["layer.lora_A.weight"]})
    with pytest.raises(ValueError, match="conflict"):
        adapter_checkpoint.validate_adapter_contract(
            contract,
            state,
            {"model_profile": "lightning35", "base_model_revision": "stale-revision"},
        )


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for independent FedAvg tests")
def test_independent_fp32_fedavg_matches_three_unequal_clients_for_three_rounds():
    import torch

    verify = _load_example_module("verify_federated_run")
    from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper

    global_state = {"layer.lora_A.weight": torch.zeros((2,), dtype=torch.float32)}
    weights = [1.0, 2.0, 4.0]
    for round_idx in range(3):
        client_states = [
            {"layer.lora_A.weight": global_state["layer.lora_A.weight"] + delta} for delta in (0.25, 0.5, 1.0)
        ]
        helper = WeightedAggregationHelper()
        for site_idx, (state, weight) in enumerate(zip(client_states, weights), start=1):
            helper.add(state, weight, f"site-{site_idx}", round_idx)
        global_state = helper.get_result()
        report = verify.verify_aggregate(client_states, weights, global_state)
        assert report["max_abs_error"] == 0.0
    assert torch.equal(global_state["layer.lora_A.weight"], torch.full((2,), 2.25))


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
