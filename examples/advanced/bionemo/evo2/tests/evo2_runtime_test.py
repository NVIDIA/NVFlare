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
import types
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for Evo2 runtime tests")


def _example_dir() -> Path:
    return Path(__file__).parents[1]


def _load_runtime_module():
    example_dir = _example_dir()
    previous_modules = {name: sys.modules.pop(name, None) for name in ("evo2_adapter_checkpoint", "evo2_runtime")}
    sys.path.insert(0, str(example_dir))
    try:
        spec = importlib.util.spec_from_file_location("evo2_runtime_under_test", example_dir / "evo2_runtime.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(example_dir))
        for name in ("evo2_adapter_checkpoint", "evo2_runtime"):
            sys.modules.pop(name, None)
        for name, previous in previous_modules.items():
            if previous is not None:
                sys.modules[name] = previous


def _install_fake_megatron(monkeypatch):
    megatron = types.ModuleType("megatron")
    megatron.__path__ = []
    bridge = types.ModuleType("megatron.bridge")
    bridge.__path__ = []
    training = types.ModuleType("megatron.bridge.training")
    training.__path__ = []
    callbacks = types.ModuleType("megatron.bridge.training.callbacks")
    core = types.ModuleType("megatron.core")
    core.__path__ = []
    utils = types.ModuleType("megatron.core.utils")

    class Callback:
        pass

    callbacks.Callback = Callback
    utils.unwrap_model = lambda model: model[0] if isinstance(model, (list, tuple)) else model
    for name, module in {
        "megatron": megatron,
        "megatron.bridge": bridge,
        "megatron.bridge.training": training,
        "megatron.bridge.training.callbacks": callbacks,
        "megatron.core": core,
        "megatron.core.utils": utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def _tiny_model(torch):
    class TinyEvo2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.backbone.requires_grad_(False)
            self.adapter = torch.nn.Linear(2, 2)
            self.classification_head = torch.nn.Linear(2, 3)

    return TinyEvo2()


def test_runtime_import_and_lora_target_parsing_do_not_require_megatron():
    runtime = _load_runtime_module()

    assert "megatron" not in runtime.__dict__
    assert runtime.parse_lora_targets(" linear_qkv,linear_proj, linear_qkv ") == (
        "linear_qkv",
        "linear_proj",
        "linear_qkv",
    )
    with pytest.raises(ValueError, match="At least one"):
        runtime.parse_lora_targets(" , ")


def test_round_train_sample_offset_continues_the_cyclic_sampler_across_fresh_processes():
    runtime = _load_runtime_module()

    assert [runtime.round_train_sample_offset(round_index, 20, 32) for round_index in range(3)] == [0, 640, 1280]
    assert runtime.round_train_sample_offset(round_index=4, local_steps=333, global_batch_size=64) == 85248
    with pytest.raises(ValueError, match="round_index"):
        runtime.round_train_sample_offset(-1, 20, 32)
    with pytest.raises(ValueError, match="positive"):
        runtime.round_train_sample_offset(1, 0, 32)


@pytest.mark.parametrize(
    "train_iters, expected",
    [
        (0, 1),
        (1, 1),
        (20, 10),
        (111, 3),
        (148, 4),
        (333, 9),
    ],
)
def test_complete_log_interval_covers_the_final_optimizer_step(train_iters, expected):
    runtime = _load_runtime_module()

    interval = runtime._complete_log_interval(train_iters)

    assert interval == expected
    assert train_iters == 0 or train_iters % interval == 0


def test_classifier_config_forces_single_gpu_and_disables_stale_checkpoint_state(tmp_path):
    runtime = _load_runtime_module()
    captured = {}

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(checkpoint=SimpleNamespace())

    upstream = SimpleNamespace(evo2_1b_classifier_config=fake_builder)
    config = runtime.build_classifier_config(
        upstream,
        base_checkpoint=str(tmp_path / "base"),
        train_file=str(tmp_path / "train.jsonl"),
        validation_file=str(tmp_path / "validation.jsonl"),
        test_file=None,
        result_dir=str(tmp_path / "round"),
        experiment_name="local_train",
        train_iters=20,
        seq_length=600,
        micro_batch_size=4,
        global_batch_size=32,
        learning_rate=5e-4,
        min_learning_rate=5e-5,
        warmup_iters=2,
        eval_interval=20,
        eval_iters=10,
        seed=1234,
        lora_dim=16,
        lora_alpha=32,
        lora_dropout=0.1,
        lora_target_modules=("linear_qkv",),
    )

    assert captured["tensor_model_parallel_size"] == 1
    assert captured["model_size"] == "evo2_1b_base"
    assert captured["use_lora"] is True
    assert captured["train_iters"] == 20
    assert captured["global_batch_size"] == 32
    assert captured["micro_batch_size"] == 4
    assert captured["log_interval"] == 10
    assert config.checkpoint.load is None
    assert config.checkpoint.save is None
    assert config.checkpoint.save_interval is None
    assert config.checkpoint.load_optim is False
    assert config.checkpoint.save_optim is False
    assert config.checkpoint.load_rng is False
    assert config.checkpoint.save_rng is False

    with pytest.raises(ValueError, match="divisible"):
        runtime.build_classifier_config(
            upstream,
            base_checkpoint="base",
            train_file="train",
            validation_file=None,
            test_file=None,
            result_dir="result",
            experiment_name="bad_batch",
            train_iters=1,
            seq_length=600,
            micro_batch_size=3,
            global_batch_size=8,
            learning_rate=1e-4,
            min_learning_rate=1e-5,
            warmup_iters=0,
            eval_interval=1,
            eval_iters=1,
            seed=1,
            lora_dim=4,
            lora_alpha=8,
            lora_dropout=0.0,
            lora_target_modules=("linear_qkv",),
        )


def test_trainable_boundary_requires_lora_and_classification_head_tensors():
    import torch

    runtime = _load_runtime_module()
    model = _tiny_model(torch)
    model.adapter.requires_grad_(False)

    with pytest.raises(ValueError, match="no trainable LoRA adapter"):
        runtime.validate_model_trainable_boundary(model)

    model.adapter.requires_grad_(True)
    boundary = runtime.validate_model_trainable_boundary(model)
    assert boundary["lora_tensors"] == 2
    assert boundary["classification_head_tensors"] == 2


def test_training_callback_loads_global_state_then_reloads_optimizer_master_parameters(monkeypatch):
    import torch

    runtime = _load_runtime_module()
    _install_fake_megatron(monkeypatch)
    model = _tiny_model(torch)
    reference = runtime.adapter_checkpoint.extract_trainable_state(model)
    incoming = OrderedDict((name, value + 3) for name, value in reference.items())

    class Optimizer:
        reload_count = 0

        def reload_model_params(self):
            self.reload_count += 1

    optimizer = Optimizer()
    context = SimpleNamespace(model=[model], optimizer=optimizer)
    callback = runtime.make_exchange_callback(incoming, extract_after_training=True)

    callback.on_data_init_start(context)
    assert all(torch.equal(callback.initial_state[name], reference[name]) for name in reference)

    callback.on_train_start(context)
    assert optimizer.reload_count == 1
    assert all(
        torch.equal(parameter, incoming[name]) for name, parameter in model.named_parameters() if name in incoming
    )
    assert all(torch.equal(callback.initial_state[name], incoming[name]) for name in incoming)

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in incoming:
                parameter.add_(2)
    callback.on_train_step_end(SimpleNamespace(loss_dict={"classification loss": torch.tensor(0.25)}, grad_norm=1.5))
    callback.on_eval_end(SimpleNamespace(total_loss_dict={"ce loss": torch.tensor(0.5)}))
    callback.on_train_end(context)

    assert callback.metrics["train_classification_loss"] == pytest.approx(0.25)
    assert callback.metrics["train_grad_norm"] == pytest.approx(1.5)
    assert callback.metrics["validation_ce_loss"] == pytest.approx(0.5)
    assert callback.metrics["trainable_tensors"] == float(len(incoming))
    assert callback.metrics["trainable_parameters"] == float(sum(tensor.numel() for tensor in incoming.values()))
    assert callback.metrics["frozen_parameters_unchanged"] == 1.0
    assert all(torch.equal(callback.updated_state[name], incoming[name] + 2) for name in incoming)


def test_training_callback_captures_rounded_bfloat16_baseline_as_float32(monkeypatch):
    import torch

    runtime = _load_runtime_module()
    _install_fake_megatron(monkeypatch)
    model = _tiny_model(torch).to(torch.bfloat16)
    extracted = runtime.adapter_checkpoint.extract_trainable_state(model)
    incoming = OrderedDict((name, torch.full_like(value, 1.003)) for name, value in extracted.items())
    optimizer = SimpleNamespace(reload_model_params=lambda: None)
    context = SimpleNamespace(model=model, optimizer=optimizer)
    callback = runtime.make_exchange_callback(incoming, extract_after_training=True)

    callback.on_data_init_start(context)
    callback.on_train_start(context)
    callback.on_train_end(context)

    expected = OrderedDict((name, tensor.to(torch.bfloat16).float()) for name, tensor in incoming.items())
    assert all(tensor.dtype == torch.float32 for tensor in callback.initial_state.values())
    assert all(torch.equal(callback.initial_state[name], expected[name]) for name in incoming)
    assert all(torch.equal(callback.updated_state[name], expected[name]) for name in incoming)
    assert any(not torch.equal(callback.initial_state[name], incoming[name]) for name in incoming)


def test_training_callback_exports_a_nonzero_bfloat16_step_relative_to_the_rounded_baseline(monkeypatch):
    import torch

    runtime = _load_runtime_module()
    _install_fake_megatron(monkeypatch)
    model = _tiny_model(torch).to(torch.bfloat16)
    extracted = runtime.adapter_checkpoint.extract_trainable_state(model)
    incoming = OrderedDict((name, torch.full_like(value, 1.003)) for name, value in extracted.items())
    context = SimpleNamespace(model=model, optimizer=SimpleNamespace(reload_model_params=lambda: None))
    callback = runtime.make_exchange_callback(incoming, extract_after_training=True)

    callback.on_data_init_start(context)
    callback.on_train_start(context)
    rounded_baseline = OrderedDict((name, value.clone()) for name, value in callback.initial_state.items())
    bfloat16_ulp_at_one = torch.finfo(torch.bfloat16).eps
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in incoming:
                parameter.add_(bfloat16_ulp_at_one)
    callback.on_train_end(context)

    model_delta = runtime.adapter_checkpoint.compute_trainable_diff(callback.updated_state, rounded_baseline)
    rebased = runtime.adapter_checkpoint.apply_trainable_diff(incoming, model_delta)
    assert all(torch.equal(value, torch.full_like(value, bfloat16_ulp_at_one)) for value in model_delta.values())
    assert all(torch.equal(rebased[name], incoming[name] + bfloat16_ulp_at_one) for name in rebased)
    assert any(not torch.equal(rebased[name], callback.updated_state[name]) for name in rebased)


def test_training_callback_sets_the_round_sample_offset_before_data_loading(monkeypatch):
    import torch

    runtime = _load_runtime_module()
    _install_fake_megatron(monkeypatch)
    model = _tiny_model(torch)
    incoming = runtime.adapter_checkpoint.extract_trainable_state(model)
    train_state = SimpleNamespace(consumed_train_samples=0)
    context = SimpleNamespace(model=model, state=SimpleNamespace(train_state=train_state))
    callback = runtime.make_exchange_callback(
        incoming,
        extract_after_training=True,
        train_sample_offset=1280,
    )

    callback.on_data_init_start(context)

    assert train_state.consumed_train_samples == 1280
    assert callback.metrics["train_sample_offset"] == 1280.0


def test_train_round_wires_the_federated_round_into_the_sampler_cursor(monkeypatch, tmp_path):
    import torch

    runtime = _load_runtime_module()
    incoming = OrderedDict(
        (
            ("decoder.adapter.lora_a.weight", torch.zeros(2, 2)),
            ("decoder.classification_head.weight", torch.zeros(3, 2)),
        )
    )
    captured = {}
    callback = SimpleNamespace(
        initial_state=OrderedDict((name, value.clone()) for name, value in incoming.items()),
        updated_state=OrderedDict((name, value.clone()) for name, value in incoming.items()),
        metrics={},
    )
    upstream = SimpleNamespace(classifier_forward_step=object())
    upstream.pretrain = lambda _config, _forward_step, callbacks: captured.update(callbacks=callbacks)

    monkeypatch.setattr(runtime, "load_classifier_module", lambda _path: upstream)
    monkeypatch.setattr(runtime, "build_classifier_config", lambda *_args, **_kwargs: object())

    def fake_make_exchange_callback(_incoming, **kwargs):
        captured.update(kwargs)
        return callback

    monkeypatch.setattr(runtime, "make_exchange_callback", fake_make_exchange_callback)

    updated, diff, metrics = runtime.train_round(
        incoming,
        classifier_path=None,
        base_checkpoint=str(tmp_path / "base"),
        train_file=str(tmp_path / "train.jsonl"),
        validation_file=str(tmp_path / "validation.jsonl"),
        result_dir=str(tmp_path / "result"),
        local_steps=5,
        seq_length=600,
        micro_batch_size=4,
        global_batch_size=8,
        learning_rate=5e-4,
        min_learning_rate=5e-5,
        warmup_iters=1,
        eval_iters=1,
        seed=1236,
        round_index=3,
        lora_dim=16,
        lora_alpha=32,
        lora_dropout=0.1,
        lora_target_modules=("linear_qkv",),
    )

    assert captured["train_sample_offset"] == 120
    assert captured["callbacks"] == [callback]
    assert list(updated) == list(incoming)
    assert all(torch.count_nonzero(tensor) == 0 for tensor in diff.values())
    assert metrics == {}


def test_train_round_preserves_fp32_server_residual_when_bfloat16_model_makes_no_update(monkeypatch, tmp_path):
    import torch

    runtime = _load_runtime_module()
    incoming = OrderedDict(
        (
            ("decoder.adapter.lora_a.weight", torch.full((2, 2), 1.003, dtype=torch.float32)),
            ("decoder.classification_head.weight", torch.full((3, 2), -1.003, dtype=torch.float32)),
        )
    )
    rounded_baseline = OrderedDict((name, tensor.to(torch.bfloat16).float()) for name, tensor in incoming.items())
    callback = SimpleNamespace(
        initial_state=rounded_baseline,
        updated_state=OrderedDict((name, tensor.clone()) for name, tensor in rounded_baseline.items()),
        metrics={},
    )
    upstream = SimpleNamespace(classifier_forward_step=object(), pretrain=lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runtime, "load_classifier_module", lambda _path: upstream)
    monkeypatch.setattr(runtime, "build_classifier_config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime, "make_exchange_callback", lambda *_args, **_kwargs: callback)

    updated, diff, _metrics = runtime.train_round(
        incoming,
        classifier_path=None,
        base_checkpoint=str(tmp_path / "base"),
        train_file=str(tmp_path / "train.jsonl"),
        validation_file=str(tmp_path / "validation.jsonl"),
        result_dir=str(tmp_path / "result"),
        local_steps=1,
        seq_length=600,
        micro_batch_size=1,
        global_batch_size=1,
        learning_rate=5e-4,
        min_learning_rate=5e-5,
        warmup_iters=0,
        eval_iters=1,
        seed=1234,
        round_index=0,
        lora_dim=16,
        lora_alpha=32,
        lora_dropout=0.1,
        lora_target_modules=("linear_qkv",),
    )

    assert all(torch.equal(updated[name], tensor) for name, tensor in incoming.items())
    assert all(torch.count_nonzero(tensor) == 0 for tensor in diff.values())
    assert any(not torch.equal(rounded_baseline[name], tensor) for name, tensor in incoming.items())


def test_train_round_returns_exact_model_delta_without_redundant_fp32_subtraction(monkeypatch, tmp_path):
    import torch

    runtime = _load_runtime_module()
    incoming = OrderedDict(
        (
            ("decoder.adapter.lora_a.weight", torch.full((2, 2), 454.4017639160156)),
            ("decoder.classification_head.weight", torch.full((3, 2), 454.4017639160156)),
        )
    )
    rounded_baseline = OrderedDict((name, tensor.to(torch.bfloat16).float()) for name, tensor in incoming.items())
    endpoint = OrderedDict((name, tensor + 138.0) for name, tensor in rounded_baseline.items())
    callback = SimpleNamespace(initial_state=rounded_baseline, updated_state=endpoint, metrics={})
    upstream = SimpleNamespace(classifier_forward_step=object(), pretrain=lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runtime, "load_classifier_module", lambda _path: upstream)
    monkeypatch.setattr(runtime, "build_classifier_config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime, "make_exchange_callback", lambda *_args, **_kwargs: callback)

    updated, diff, _metrics = runtime.train_round(
        incoming,
        classifier_path=None,
        base_checkpoint=str(tmp_path / "base"),
        train_file=str(tmp_path / "train.jsonl"),
        validation_file=str(tmp_path / "validation.jsonl"),
        result_dir=str(tmp_path / "result"),
        local_steps=1,
        seq_length=600,
        micro_batch_size=1,
        global_batch_size=1,
        learning_rate=5e-4,
        min_learning_rate=5e-5,
        warmup_iters=0,
        eval_iters=1,
        seed=1234,
        round_index=0,
        lora_dim=16,
        lora_alpha=32,
        lora_dropout=0.1,
        lora_target_modules=("linear_qkv",),
    )

    intended = OrderedDict((name, torch.full_like(tensor, 138.0)) for name, tensor in incoming.items())
    assert all(torch.equal(diff[name], intended[name]) for name in diff)
    assert all(torch.equal(updated[name], incoming[name] + intended[name]) for name in updated)
    redundantly_subtracted = runtime.adapter_checkpoint.compute_trainable_diff(updated, incoming)
    assert any(not torch.equal(redundantly_subtracted[name], diff[name]) for name in diff)


def test_training_callback_rejects_a_changed_frozen_backbone(monkeypatch):
    import torch

    runtime = _load_runtime_module()
    _install_fake_megatron(monkeypatch)
    model = _tiny_model(torch)
    incoming = runtime.adapter_checkpoint.extract_trainable_state(model)
    optimizer = SimpleNamespace(reload_model_params=lambda: None)
    context = SimpleNamespace(model=model, optimizer=optimizer)
    callback = runtime.make_exchange_callback(incoming, extract_after_training=True)

    callback.on_data_init_start(context)
    callback.on_train_start(context)
    with torch.no_grad():
        model.backbone.weight.add_(1)

    with pytest.raises(RuntimeError, match="frozen Evo2 backbone parameter changed"):
        callback.on_train_end(context)


def test_evaluation_callback_loads_before_data_iteration_without_optimizer_reload(monkeypatch):
    import torch

    runtime = _load_runtime_module()
    _install_fake_megatron(monkeypatch)
    model = _tiny_model(torch)
    reference = runtime.adapter_checkpoint.extract_trainable_state(model)
    incoming = OrderedDict((name, value - 4) for name, value in reference.items())

    class Optimizer:
        reload_count = 0

        def reload_model_params(self):
            self.reload_count += 1

    optimizer = Optimizer()
    context = SimpleNamespace(model=model, optimizer=optimizer)
    callback = runtime.make_exchange_callback(
        incoming,
        extract_after_training=False,
        load_on_data_init=True,
    )

    callback.on_data_init_start(context)
    callback.on_train_start(context)

    assert optimizer.reload_count == 0
    assert all(
        torch.equal(parameter, incoming[name]) for name, parameter in model.named_parameters() if name in incoming
    )
    assert all(torch.equal(callback.initial_state[name], incoming[name]) for name in incoming)
