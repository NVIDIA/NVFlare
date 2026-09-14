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
import sys
import types
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_NVFLARE_RUNTIME_DEPS = importlib.util.find_spec("msgpack") is not None
pytestmark = pytest.mark.skipif(
    not (HAS_TORCH and HAS_NVFLARE_RUNTIME_DEPS),
    reason="PyTorch and NVFlare runtime dependencies are required for Evo2 mock client tests",
)


def _example_dir() -> Path:
    return Path(__file__).parents[5] / "examples" / "advanced" / "bionemo" / "evo2"


def _load_module(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, _example_dir() / file_name)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_example_modules():
    example_dir = _example_dir()
    module_names = ("evo2_adapter_checkpoint", "evo2_runtime")
    previous_modules = {name: sys.modules.pop(name, None) for name in module_names}
    sys.path.insert(0, str(example_dir))
    try:
        prepare_initial_model = _load_module("evo2_prepare_initial_model_under_test", "prepare_initial_model.py")
        client = _load_module("evo2_client_under_test", "client.py")
        return prepare_initial_model, client, prepare_initial_model.adapter_checkpoint
    finally:
        sys.path.remove(str(example_dir))
        for name in module_names:
            sys.modules.pop(name, None)
        for name, previous in previous_modules.items():
            if previous is not None:
                sys.modules[name] = previous


def _mock_args(tmp_path: Path, *, delta: float, sample_count: int):
    return SimpleNamespace(
        backend="mock",
        work_dir=str(tmp_path / "client work"),
        mock_delta=delta,
        local_steps=20,
        sample_count=sample_count,
    )


def test_mock_initialization_is_common_deterministic_and_peft_mode_specific():
    import torch

    prepare_initial_model, _client, _adapter_checkpoint = _load_example_modules()

    first = prepare_initial_model.create_mock_state("lora", seed=1234)
    repeated = prepare_initial_model.create_mock_state("lora", seed=1234)
    different_seed = prepare_initial_model.create_mock_state("lora", seed=1235)
    head_only = prepare_initial_model.create_mock_state("head-only", seed=1234)

    assert list(first) == list(repeated)
    assert all(torch.equal(first[name], repeated[name]) for name in first)
    assert all(first[name].data_ptr() != repeated[name].data_ptr() for name in first)
    assert not torch.equal(
        first["decoder.classification_head.weight"],
        different_seed["decoder.classification_head.weight"],
    )
    assert list(head_only) == ["decoder.classification_head.weight", "decoder.classification_head.bias"]


def test_mock_initialization_checkpoint_records_canonical_exchange_dtype(tmp_path):
    import torch

    prepare_initial_model, _client, adapter_checkpoint = _load_example_modules()
    output = tmp_path / "initial.pt"
    args = SimpleNamespace(
        backend="mock",
        base_checkpoint=str(tmp_path / "unused-base"),
        data_file=str(tmp_path / "unused-data.jsonl"),
        output=str(output),
        work_dir=str(tmp_path / "unused-work"),
        classifier_file=None,
        seq_length=600,
        seed=1234,
        peft_mode="lora",
        lora_dim=16,
        lora_alpha=32,
        lora_dropout=0.1,
        lora_target_modules="linear_qkv,linear_proj",
    )

    state = prepare_initial_model.prepare_initial_model(args)
    metadata = adapter_checkpoint.load_nvflare_checkpoint_metadata(output)

    assert metadata["exchange_dtype"] == "float32"
    assert all(tensor.dtype == torch.float32 for tensor in state.values())
    assert all(tensor.dtype == torch.float32 for tensor in adapter_checkpoint.load_nvflare_checkpoint(output).values())


def test_mock_round_returns_diff_without_mutating_common_initialization(tmp_path):
    import torch

    prepare_initial_model, client, adapter_checkpoint = _load_example_modules()
    initial = prepare_initial_model.create_mock_state("lora", seed=7)
    pristine = OrderedDict((name, tensor.clone()) for name, tensor in initial.items())
    args = _mock_args(tmp_path, delta=0.125, sample_count=17)

    diff, metrics, attempt_dir = client.train_one_round(
        args,
        initial,
        site_name="site-1",
        current_round=2,
    )

    assert all(torch.equal(initial[name], pristine[name]) for name in initial)
    assert all(torch.allclose(value, torch.full_like(value, 0.125)) for value in diff.values())
    assert metrics["validation_accuracy"] == pytest.approx(0.625)
    assert metrics["local_steps"] == 20.0
    assert metrics["samples_available"] == 17.0
    assert metrics["received_mebibytes"] == adapter_checkpoint.state_dict_size_mb(initial)
    assert metrics["sent_mebibytes"] == adapter_checkpoint.state_dict_size_mb(diff)
    assert Path(attempt_dir).name.startswith("site-1_round_002_")
    persisted_metrics = json.loads((Path(attempt_dir) / "round_metrics.json").read_text(encoding="utf-8"))
    assert persisted_metrics["samples_available"] == 17.0
    assert persisted_metrics["site_name"] == "site-1"
    assert persisted_metrics["round"] == 2
    local_checkpoint = Path(persisted_metrics["local_checkpoint"])
    assert local_checkpoint == Path(attempt_dir).resolve() / "local_trainable_model.pt"
    saved_state = adapter_checkpoint.load_nvflare_checkpoint(local_checkpoint)
    assert all(torch.equal(saved_state[name], initial[name] + 0.125) for name in initial)

    reconstructed = adapter_checkpoint.apply_trainable_diff(initial, diff)
    assert all(torch.equal(reconstructed[name], initial[name] + 0.125) for name in initial)


def test_client_validates_received_exchange_dtype_and_labels_sent_diff(tmp_path, monkeypatch):
    prepare_initial_model, client, _adapter_checkpoint = _load_example_modules()
    initial = prepare_initial_model.create_mock_state("lora", seed=7)
    received = client.flare.FLModel(
        params_type=client.flare.ParamsType.FULL,
        params=initial,
        current_round=0,
        meta={"exchange_dtype": "float32"},
    )
    sent = []
    monkeypatch.setattr(client.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(client.flare, "init", lambda: None)
    monkeypatch.setattr(client.flare, "receive", lambda: received)
    monkeypatch.setattr(client.flare, "system_info", lambda: {"site_name": "site-1"})
    monkeypatch.setattr(client.flare, "send", sent.append)

    client.main(
        [
            "--backend",
            "mock",
            "--train-file",
            str(tmp_path / "train.jsonl"),
            "--validation-file",
            str(tmp_path / "validation.jsonl"),
            "--base-checkpoint",
            str(tmp_path / "base"),
            "--work-dir",
            str(tmp_path / "work"),
            "--sample-count",
            "9",
        ]
    )

    assert len(sent) == 1
    assert sent[0].params_type == client.flare.ParamsType.DIFF
    assert sent[0].meta == {
        client.FLMetaKey.NUM_STEPS_CURRENT_ROUND: 1,
        "exchange_dtype": "float32",
    }

    for metadata in ({}, {"exchange_dtype": "bfloat16"}):
        received.meta = metadata
        with pytest.raises(ValueError, match="global model metadata.*exchange_dtype='float32'"):
            client.main(
                [
                    "--backend",
                    "mock",
                    "--train-file",
                    str(tmp_path / "train.jsonl"),
                    "--validation-file",
                    str(tmp_path / "validation.jsonl"),
                    "--base-checkpoint",
                    str(tmp_path / "base"),
                    "--work-dir",
                    str(tmp_path / "invalid-work"),
                    "--sample-count",
                    "9",
                ]
            )


def test_sample_weighted_mock_diffs_match_manifest_weighted_fedavg(tmp_path):
    import torch

    prepare_initial_model, client, adapter_checkpoint = _load_example_modules()
    initial = prepare_initial_model.create_mock_state("lora", seed=11)
    site_specs = (("site-1", 10, 0.01), ("site-2", 30, 0.02), ("site-3", 60, 0.03))
    site_diffs = []

    for site_name, sample_count, delta in site_specs:
        diff, _metrics, _attempt_dir = client.train_one_round(
            _mock_args(tmp_path / site_name, delta=delta, sample_count=sample_count),
            initial,
            site_name=site_name,
            current_round=0,
        )
        site_diffs.append((sample_count, diff))

    total_samples = sum(sample_count for sample_count, _diff in site_diffs)
    aggregated = OrderedDict(
        (
            name,
            sum(diff[name] * sample_count for sample_count, diff in site_diffs) / total_samples,
        )
        for name in initial
    )
    global_state = adapter_checkpoint.apply_trainable_diff(initial, aggregated)
    expected_delta = (10 * 0.01 + 30 * 0.02 + 60 * 0.03) / 100

    assert expected_delta == pytest.approx(0.025)
    assert all(torch.allclose(aggregated[name], torch.full_like(aggregated[name], expected_delta)) for name in initial)
    assert all(torch.allclose(global_state[name], initial[name] + expected_delta) for name in initial)


def test_bionemo_round_sends_the_model_boundary_diff_and_persists_the_rebased_state(tmp_path):
    import torch

    prepare_initial_model, client, adapter_checkpoint = _load_example_modules()
    template = prepare_initial_model.create_mock_state("lora", seed=11)
    incoming = OrderedDict((name, torch.full_like(value, 454.4017639160156)) for name, value in template.items())
    model_delta = OrderedDict((name, torch.full_like(value, 138.0)) for name, value in incoming.items())
    rebased = adapter_checkpoint.apply_trainable_diff(incoming, model_delta)
    assert any(
        not torch.equal(value, model_delta[name])
        for name, value in adapter_checkpoint.compute_trainable_diff(rebased, incoming).items()
    )

    fake_runtime = types.ModuleType("evo2_runtime")
    fake_runtime.TRAINING_STATE_MANIFEST_FILENAME = "nvflare_training_state.json"
    fake_runtime.parse_lora_targets = lambda value: tuple(value.split(","))
    fake_runtime.train_round = lambda received, **_kwargs: (rebased, model_delta, {})
    args = SimpleNamespace(
        backend="bionemo",
        train_file=str(tmp_path / "train.jsonl"),
        validation_file=str(tmp_path / "validation.jsonl"),
        base_checkpoint=str(tmp_path / "base"),
        classifier_file=str(tmp_path / "classifier.py"),
        work_dir=str(tmp_path / "client work"),
        training_state_dir=None,
        local_steps=1,
        sample_count=9,
        seq_length=600,
        micro_batch_size=1,
        global_batch_size=1,
        learning_rate=5e-4,
        min_learning_rate=5e-5,
        warmup_iters=0,
        eval_iters=1,
        seed=1234,
        peft_mode="lora",
        lora_dim=16,
        lora_alpha=32,
        lora_dropout=0.1,
        lora_target_modules="linear_qkv",
        mock_delta=0.01,
    )
    previous_runtime = sys.modules.get("evo2_runtime")
    sys.modules["evo2_runtime"] = fake_runtime
    try:
        returned_diff, _metrics, attempt_dir = client.train_one_round(
            args,
            incoming,
            site_name="site-1",
            current_round=0,
        )
    finally:
        if previous_runtime is None:
            sys.modules.pop("evo2_runtime", None)
        else:
            sys.modules["evo2_runtime"] = previous_runtime

    assert all(torch.equal(returned_diff[name], model_delta[name]) for name in returned_diff)
    saved = adapter_checkpoint.load_nvflare_checkpoint(Path(attempt_dir) / "local_trainable_model.pt")
    assert all(torch.equal(saved[name], rebased[name]) for name in saved)


def test_bionemo_rounds_use_constant_seed_and_keep_private_state_path_out_of_fl_metrics(tmp_path):
    import torch

    prepare_initial_model, client, adapter_checkpoint = _load_example_modules()
    incoming = prepare_initial_model.create_mock_state("lora", seed=11)
    training_state_dir = tmp_path / "private state" / "site-1"
    captured = {}
    fake_runtime = types.ModuleType("evo2_runtime")
    fake_runtime.TRAINING_STATE_MANIFEST_FILENAME = "nvflare_training_state.json"
    fake_runtime.parse_lora_targets = lambda value: tuple(value.split(","))

    def fake_train_round(received, **kwargs):
        captured.update(kwargs)
        updated = OrderedDict((name, value + 0.25) for name, value in received.items())
        diff = adapter_checkpoint.compute_trainable_diff(updated, received)
        manifest = training_state_dir / "round_003" / fake_runtime.TRAINING_STATE_MANIFEST_FILENAME
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{}\n", encoding="utf-8")
        return updated, diff, {"persistent_training_state": 1.0}

    fake_runtime.train_round = fake_train_round
    args = SimpleNamespace(
        backend="bionemo",
        train_file=str(tmp_path / "train.jsonl"),
        validation_file=str(tmp_path / "validation.jsonl"),
        base_checkpoint=str(tmp_path / "base"),
        classifier_file=str(tmp_path / "classifier.py"),
        work_dir=str(tmp_path / "client work"),
        training_state_dir=str(training_state_dir),
        local_steps=111,
        sample_count=9000,
        seq_length=600,
        micro_batch_size=4,
        global_batch_size=64,
        learning_rate=5e-4,
        min_learning_rate=5e-5,
        warmup_iters=30,
        eval_iters=10,
        seed=1234,
        peft_mode="lora",
        lora_dim=16,
        lora_alpha=32,
        lora_dropout=0.1,
        lora_target_modules="linear_qkv,linear_proj",
        mock_delta=0.01,
    )
    previous_runtime = sys.modules.get("evo2_runtime")
    sys.modules["evo2_runtime"] = fake_runtime
    try:
        diff, metrics, attempt_dir = client.train_one_round(
            args,
            incoming,
            site_name="site-1",
            current_round=3,
        )

        assert captured["seed"] == 1234
        assert captured["training_state_dir"] == str(training_state_dir)
        assert captured["site_name"] == "site-1"
        assert all(torch.allclose(value, torch.full_like(value, 0.25)) for value in diff.values())
        assert "training_state_manifest" not in metrics
        assert all(isinstance(value, float) for value in metrics.values())
        persisted = json.loads((Path(attempt_dir) / "round_metrics.json").read_text(encoding="utf-8"))
        assert persisted["training_state_manifest"] == str(
            (training_state_dir / "round_003" / fake_runtime.TRAINING_STATE_MANIFEST_FILENAME).resolve()
        )

        args.training_state_dir = None
        _diff, reset_metrics, reset_attempt_dir = client.train_one_round(
            args,
            incoming,
            site_name="site-1",
            current_round=4,
        )
    finally:
        if previous_runtime is None:
            sys.modules.pop("evo2_runtime", None)
        else:
            sys.modules["evo2_runtime"] = previous_runtime

    assert captured["seed"] == 1234
    assert captured["round_index"] == 4
    assert captured["training_state_dir"] is None
    assert "training_state_manifest" not in reset_metrics
    reset_persisted = json.loads((Path(reset_attempt_dir) / "round_metrics.json").read_text(encoding="utf-8"))
    assert "training_state_manifest" not in reset_persisted


def test_client_rejects_persistent_training_state_with_mock_backend(tmp_path):
    _prepare_initial_model, client, _adapter_checkpoint = _load_example_modules()
    args = client.define_parser().parse_args(
        [
            "--backend",
            "mock",
            "--train-file",
            str(tmp_path / "train.jsonl"),
            "--validation-file",
            str(tmp_path / "validation.jsonl"),
            "--base-checkpoint",
            str(tmp_path / "base"),
            "--work-dir",
            str(tmp_path / "work"),
            "--sample-count",
            "10",
            "--training-state-dir",
            str(tmp_path / "state"),
        ]
    )

    with pytest.raises(ValueError, match="only by the BioNeMo backend"):
        client._validate_args(args)

    args.training_state_dir = "  "
    with pytest.raises(ValueError, match="non-empty path"):
        client._validate_args(args)
