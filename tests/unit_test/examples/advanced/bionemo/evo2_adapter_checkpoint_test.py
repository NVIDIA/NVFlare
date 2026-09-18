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
from collections import OrderedDict
from pathlib import Path

import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for Evo2 adapter checkpoint tests")


def _load_adapter_checkpoint():
    module_path = (
        Path(__file__).parents[5] / "examples" / "advanced" / "bionemo" / "evo2" / "evo2_adapter_checkpoint.py"
    )
    spec = importlib.util.spec_from_file_location("evo2_adapter_checkpoint", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_model(torch):
    class TinyEvo2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.Module()
            self.decoder.linear_in = torch.nn.Linear(3, 2, bias=False)
            self.decoder.linear_out = torch.nn.Linear(2, 3, bias=False)
            self.decoder.adapter = torch.nn.Linear(3, 3, bias=False)
            self.classification_head = torch.nn.Linear(3, 2)
            self.backbone = torch.nn.Linear(3, 3, bias=False)
            self.frozen = torch.nn.Module()
            self.frozen.adapter = torch.nn.Linear(3, 3, bias=False)
            self.frozen.adapter.requires_grad_(False)

    return TinyEvo2()


def test_parameter_predicate_matches_only_lora_and_classification_head_segments():
    adapter_checkpoint = _load_adapter_checkpoint()

    matches = [
        "decoder.layers.0.mixer.dense_projection.adapter.linear_in.weight",
        "module.decoder.layers.3.self_attention.linear_qkv.adapter.linear_out.weight",
        "decoder.layers.0.adapter.weight",
        "classification_head.weight",
        "wrapper.classification_head.out_proj.bias",
    ]
    misses = [
        "decoder.layers.0.weight",
        "decoder.layers.0.linear_in.weight",
        "module.decoder.linear_out.weight",
        "decoder.layers.0.linear_input.weight",
        "decoder.layers.0.adapter_norm.weight",
        "classification_header.weight",
    ]

    assert all(adapter_checkpoint.is_trainable_parameter(name) for name in matches)
    assert not any(adapter_checkpoint.is_trainable_parameter(name) for name in misses)


def test_extract_trainable_state_uses_requires_grad_and_returns_independent_cpu_clones():
    import torch

    adapter_checkpoint = _load_adapter_checkpoint()
    model = _make_model(torch)
    parameters = dict(model.named_parameters())

    state = adapter_checkpoint.extract_trainable_state([model])

    assert list(state) == [
        "decoder.adapter.weight",
        "classification_head.weight",
        "classification_head.bias",
    ]
    assert "backbone.weight" not in state
    assert "frozen.adapter.weight" not in state
    assert all(tensor.device.type == "cpu" and tensor.grad_fn is None for tensor in state.values())
    assert all(tensor.dtype == torch.float32 for tensor in state.values())
    assert state["decoder.adapter.weight"].data_ptr() != parameters["decoder.adapter.weight"].data_ptr()

    original = parameters["decoder.adapter.weight"].detach().clone()
    state["decoder.adapter.weight"].add_(10)
    assert torch.equal(parameters["decoder.adapter.weight"], original)

    with pytest.raises(ValueError, match="exactly one model chunk"):
        adapter_checkpoint.extract_trainable_state([model, model])


def test_bfloat16_model_boundary_extracts_float32_and_loads_with_explicit_rounding():
    import torch

    adapter_checkpoint = _load_adapter_checkpoint()
    model = _make_model(torch).to(dtype=torch.bfloat16)

    extracted = adapter_checkpoint.extract_trainable_state(model)

    assert all(tensor.dtype == torch.float32 and tensor.device.type == "cpu" for tensor in extracted.values())
    incoming = OrderedDict((name, tensor.clone()) for name, tensor in extracted.items())
    incoming["decoder.adapter.weight"].fill_(1.003)

    adapter_checkpoint.load_trainable_state(model, incoming)

    parameters = dict(model.named_parameters())
    assert parameters["decoder.adapter.weight"].dtype == torch.bfloat16
    assert torch.equal(
        parameters["decoder.adapter.weight"].float(),
        incoming["decoder.adapter.weight"].to(torch.bfloat16).float(),
    )
    rounded = adapter_checkpoint.extract_trainable_state(model)
    assert torch.equal(rounded["decoder.adapter.weight"], parameters["decoder.adapter.weight"].float())


def test_copy_trainable_state_rejects_backbone_keys_in_exchange_payload():
    import torch

    adapter_checkpoint = _load_adapter_checkpoint()
    source = OrderedDict(
        [
            ("decoder.adapter.weight", torch.ones(2, 2)),
            ("classification_head.bias", torch.zeros(2)),
            ("decoder.linear_in.weight", torch.full((2, 2), 2.0)),
            ("decoder.backbone.weight", torch.full((2, 2), 3.0)),
        ]
    )

    with pytest.raises(
        ValueError, match="unsupported parameter names.*decoder.linear_in.weight.*decoder.backbone.weight"
    ):
        adapter_checkpoint.copy_trainable_state(source, context="NVFlare global trainable state")

    del source["decoder.linear_in.weight"]
    del source["decoder.backbone.weight"]
    copied = adapter_checkpoint.copy_trainable_state(source)
    assert list(copied) == list(source)
    assert all(copied[name].data_ptr() != source[name].data_ptr() for name in source)


def test_strict_load_validates_before_modifying_model():
    import torch

    adapter_checkpoint = _load_adapter_checkpoint()
    model = _make_model(torch)
    reference = adapter_checkpoint.extract_trainable_state(model)
    incoming = OrderedDict((name, tensor + 1) for name, tensor in reference.items())
    backbone_before = model.backbone.weight.detach().clone()

    assert adapter_checkpoint.load_trainable_state((model,), incoming) == len(incoming)
    assert torch.equal(model.decoder.adapter.weight, incoming["decoder.adapter.weight"])
    assert torch.equal(model.backbone.weight, backbone_before)

    state_before_bad_load = adapter_checkpoint.extract_trainable_state(model)
    bad_shape = OrderedDict((name, tensor + 2) for name, tensor in state_before_bad_load.items())
    bad_shape["classification_head.bias"] = torch.zeros(3, dtype=bad_shape["classification_head.bias"].dtype)
    with pytest.raises(ValueError, match="classification_head.bias.*shape"):
        adapter_checkpoint.load_trainable_state(model, bad_shape)
    state_after_bad_load = adapter_checkpoint.extract_trainable_state(model)
    assert all(torch.equal(state_after_bad_load[name], tensor) for name, tensor in state_before_bad_load.items())


def test_validation_rejects_missing_unexpected_shape_and_dtype():
    import torch

    adapter_checkpoint = _load_adapter_checkpoint()
    reference = OrderedDict(
        [
            ("decoder.adapter.weight", torch.zeros(2, 2)),
            ("classification_head.bias", torch.zeros(2)),
        ]
    )

    with pytest.raises(KeyError, match="missing=.*classification_head.bias"):
        adapter_checkpoint.validate_trainable_state({"decoder.adapter.weight": torch.zeros(2, 2)}, reference)
    with pytest.raises(KeyError, match="unexpected=.*adapter.linear_in.weight"):
        adapter_checkpoint.validate_trainable_state(
            {
                **reference,
                "decoder.layers.0.adapter.linear_in.weight": torch.zeros(2, 2),
            },
            reference,
        )
    with pytest.raises(ValueError, match="decoder.adapter.weight.*shape"):
        adapter_checkpoint.validate_trainable_state(
            {
                "decoder.adapter.weight": torch.zeros(3, 2),
                "classification_head.bias": torch.zeros(2),
            },
            reference,
        )
    with pytest.raises(ValueError, match="classification_head.bias.*dtype"):
        adapter_checkpoint.validate_trainable_state(
            {
                "decoder.adapter.weight": torch.zeros(2, 2),
                "classification_head.bias": torch.zeros(2, dtype=torch.float64),
            },
            reference,
        )
    with pytest.raises(ValueError, match="classification_head.bias.*shape"):
        adapter_checkpoint.validate_trainable_state(
            {
                "decoder.adapter.weight": torch.zeros(2, 2),
                "classification_head.bias": torch.full((3,), float("nan")),
            },
            reference,
        )
    with pytest.raises(ValueError, match="classification_head.bias.*dtype"):
        adapter_checkpoint.validate_trainable_state(
            {
                "decoder.adapter.weight": torch.zeros(2, 2),
                "classification_head.bias": torch.full((2,), float("nan"), dtype=torch.float64),
            },
            reference,
        )


def test_reusable_schema_validates_reference_once_and_each_checkpoint_boundary(tmp_path, monkeypatch):
    import torch

    adapter_checkpoint = _load_adapter_checkpoint()
    reference = OrderedDict(
        [
            ("decoder.adapter.weight", torch.zeros(2, 2)),
            ("classification_head.bias", torch.zeros(2)),
        ]
    )
    validation_labels = []
    original_validate_finite = adapter_checkpoint._validate_finite_tensors

    def track_finite_validation(tensors, label):
        validation_labels.append(label)
        original_validate_finite(tensors, label)

    monkeypatch.setattr(adapter_checkpoint, "_validate_finite_tensors", track_finite_validation)
    schema = adapter_checkpoint.ValidatedTrainableStateSchema(reference)

    for index in range(2):
        checkpoint_path = tmp_path / f"valid_{index}.pt"
        torch.save(
            {"model": OrderedDict((name, tensor + index) for name, tensor in reference.items())}, checkpoint_path
        )
        loaded = adapter_checkpoint.load_nvflare_checkpoint(checkpoint_path, schema=schema)
        assert list(loaded) == list(reference)

    invalid_checkpoint = tmp_path / "non_finite.pt"
    invalid_state = OrderedDict((name, tensor.clone()) for name, tensor in reference.items())
    invalid_state["classification_head.bias"][0] = float("nan")
    torch.save({"model": invalid_state}, invalid_checkpoint)
    with pytest.raises(ValueError, match="NVFlare checkpoint model state contains non-finite values"):
        adapter_checkpoint.load_nvflare_checkpoint(invalid_checkpoint, schema=schema)

    assert validation_labels == [
        "Reference trainable state",
        "NVFlare checkpoint model state",
        "NVFlare checkpoint model state",
        "NVFlare checkpoint model state",
    ]


def test_exchange_boundaries_reject_bfloat16_payloads(tmp_path):
    import torch

    adapter_checkpoint = _load_adapter_checkpoint()
    reference = OrderedDict(
        [
            ("decoder.adapter.weight", torch.zeros(2, 2)),
            ("classification_head.bias", torch.zeros(2)),
        ]
    )
    bfloat16_state = OrderedDict((name, tensor.to(torch.bfloat16)) for name, tensor in reference.items())

    with pytest.raises(ValueError, match="only float32 tensors.*torch.bfloat16"):
        adapter_checkpoint.copy_trainable_state(bfloat16_state)
    with pytest.raises(ValueError, match="dtype.*torch.bfloat16"):
        adapter_checkpoint.validate_trainable_state(bfloat16_state, reference)
    with pytest.raises(ValueError, match="only float32 tensors.*torch.bfloat16"):
        adapter_checkpoint.save_nvflare_checkpoint(bfloat16_state, tmp_path / "bfloat16.pt")
    legacy_checkpoint = tmp_path / "legacy_bfloat16.pt"
    torch.save({"model": bfloat16_state}, legacy_checkpoint)
    with pytest.raises(ValueError, match="only float32 tensors.*torch.bfloat16"):
        adapter_checkpoint.load_nvflare_checkpoint(legacy_checkpoint)


@pytest.mark.parametrize(
    "non_finite_value",
    (float("nan"), float("inf"), float("-inf")),
    ids=("nan", "positive_inf", "negative_inf"),
)
def test_exchange_and_checkpoint_save_reject_non_finite_tensors(tmp_path, non_finite_value):
    import torch

    adapter_checkpoint = _load_adapter_checkpoint()
    reference = OrderedDict(
        [
            ("decoder.adapter.weight", torch.zeros(2, 2)),
            ("classification_head.bias", torch.zeros(2)),
        ]
    )
    invalid = OrderedDict((name, tensor.clone()) for name, tensor in reference.items())
    invalid["classification_head.bias"][0] = non_finite_value

    with pytest.raises(
        ValueError, match="Incoming trainable state contains non-finite values.*classification_head.bias"
    ):
        adapter_checkpoint.validate_trainable_state(invalid, reference)
    with pytest.raises(
        ValueError, match="Received trainable state contains non-finite values.*classification_head.bias"
    ):
        adapter_checkpoint.copy_trainable_state(invalid, context="Received trainable state")

    checkpoint_path = tmp_path / "invalid.pt"
    with pytest.raises(
        ValueError, match="Trainable checkpoint state contains non-finite values.*classification_head.bias"
    ):
        adapter_checkpoint.save_nvflare_checkpoint(invalid, checkpoint_path)
    assert not checkpoint_path.exists()


def test_diff_round_trip_preserves_float32_ulp_and_returns_cpu_clones():
    import torch

    adapter_checkpoint = _load_adapter_checkpoint()
    reference = OrderedDict(
        [
            ("decoder.layers.0.adapter.linear_in.weight", torch.ones(2, 2, dtype=torch.float32)),
            ("classification_head.bias", torch.tensor([1.0, 2.0], dtype=torch.float32)),
        ]
    )
    current = OrderedDict(
        (name, torch.nextafter(tensor, torch.full_like(tensor, float("inf")))) for name, tensor in reference.items()
    )

    diff = adapter_checkpoint.compute_trainable_diff(current, reference)
    restored = adapter_checkpoint.apply_trainable_diff(reference, diff)

    assert all(tensor.device.type == "cpu" and tensor.dtype == torch.float32 for tensor in diff.values())
    assert all(torch.count_nonzero(tensor) == tensor.numel() for tensor in diff.values())
    assert all(torch.equal(restored[name], current[name]) for name in reference)
    diff["decoder.layers.0.adapter.linear_in.weight"].zero_()
    assert not torch.equal(
        current["decoder.layers.0.adapter.linear_in.weight"], diff["decoder.layers.0.adapter.linear_in.weight"]
    )


def test_nvflare_checkpoint_round_trip_and_payload_size(tmp_path):
    import torch

    adapter_checkpoint = _load_adapter_checkpoint()
    state = OrderedDict(
        [
            ("decoder.adapter.weight", torch.arange(4, dtype=torch.float32).reshape(2, 2)),
            ("classification_head.bias", torch.ones(2, dtype=torch.float32)),
        ]
    )
    checkpoint_path = tmp_path / "nested" / "initial_adapter.pt"

    adapter_checkpoint.save_nvflare_checkpoint(state, checkpoint_path, metadata={"rank": 16, "targets": {"qkv"}})
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    loaded = adapter_checkpoint.load_nvflare_checkpoint(checkpoint_path)

    assert list(raw) == ["model", "train_conf", "meta_props"]
    assert raw["train_conf"] == {"train": {"model": "Evo2LoRAClassifier"}}
    assert raw["meta_props"] == {"rank": 16, "targets": ["qkv"]}
    assert list(loaded) == list(state)
    assert all(torch.equal(loaded[name], tensor) for name, tensor in state.items())
    assert all(tensor.dtype == torch.float32 for tensor in raw["model"].values())
    assert all(tensor.dtype == torch.float32 for tensor in loaded.values())
    expected_bytes = 6 * torch.tensor([], dtype=torch.float32).element_size()
    assert adapter_checkpoint.state_dict_size_mb(state) == expected_bytes / (1024 * 1024)


def test_checkpoint_load_never_retries_with_unrestricted_pickle(tmp_path, monkeypatch):
    adapter_checkpoint = _load_adapter_checkpoint()
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.touch()
    load_kwargs = []

    def reject_weights_only(*_args, **kwargs):
        load_kwargs.append(kwargs)
        if kwargs.get("weights_only") is not True:
            raise AssertionError("Checkpoint load retried without weights_only=True")
        raise TypeError("load() got an unexpected keyword argument 'weights_only'")

    monkeypatch.setattr(adapter_checkpoint.torch, "load", reject_weights_only)

    with pytest.raises(RuntimeError, match="Safe checkpoint loading requires a PyTorch version"):
        adapter_checkpoint.load_nvflare_checkpoint(checkpoint_path)

    assert load_kwargs == [{"map_location": "cpu", "weights_only": True}]
