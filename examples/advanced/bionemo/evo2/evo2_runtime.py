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
"""Lazy BioNeMo/Megatron Bridge integration for federated Evo2 training."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import evo2_adapter_checkpoint as adapter_checkpoint
import torch

BIONEMO_CLASSIFIER_ENV = "BIONEMO_EVO2_CLASSIFIER"
DEFAULT_CLASSIFIER_PATH = "/opt/bionemo-recipes/recipes/evo2_megatron/examples/evo2_classifier.py"
DEFAULT_LORA_TARGET_MODULES = (
    "linear_qkv",
    "linear_proj",
    "linear_fc1",
    "linear_fc2",
    "dense_projection",
    "dense",
)


def resolve_classifier_path(classifier_path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve and validate the pinned upstream BioNeMo classifier module."""

    requested = classifier_path or os.environ.get(BIONEMO_CLASSIFIER_ENV) or DEFAULT_CLASSIFIER_PATH
    resolved = Path(requested).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"BioNeMo Evo2 classifier was not found at {resolved}. Set {BIONEMO_CLASSIFIER_ENV} "
            "or pass --classifier-file. The example Docker image sets this automatically."
        )
    return resolved


def load_classifier_module(classifier_path: str | os.PathLike[str] | None = None) -> ModuleType:
    """Load the pinned tutorial classifier without requiring it to be an installed package."""

    resolved = resolve_classifier_path(classifier_path)
    existing = sys.modules.get("evo2_classifier")
    if existing is not None:
        existing_path = Path(getattr(existing, "__file__", "")).resolve()
        if existing_path != resolved:
            raise RuntimeError(f"evo2_classifier is already loaded from {existing_path}, cannot also load {resolved}.")
        return existing

    spec = importlib.util.spec_from_file_location("evo2_classifier", resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import specification for {resolved}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def parse_lora_targets(value: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Normalize the comma-separated LoRA module CLI value."""

    if isinstance(value, str):
        targets = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        targets = tuple(str(item).strip() for item in value if str(item).strip())
    if not targets:
        raise ValueError("At least one LoRA target module is required.")
    return targets


def round_train_sample_offset(round_index: int, local_steps: int, global_batch_size: int) -> int:
    """Return the cyclic-sampler cursor at the start of a federated round."""

    if round_index < 0:
        raise ValueError("round_index must be non-negative.")
    if local_steps <= 0 or global_batch_size <= 0:
        raise ValueError("local_steps and global_batch_size must be positive.")
    return round_index * local_steps * global_batch_size


def _complete_log_interval(train_iters: int, maximum: int = 10) -> int:
    """Choose a short interval that emits Megatron's counters on the final step."""

    if train_iters <= 0:
        return 1
    for interval in range(min(maximum, train_iters), 0, -1):
        if train_iters % interval == 0:
            return interval
    return 1


def build_classifier_config(
    upstream: ModuleType,
    *,
    base_checkpoint: str,
    train_file: str,
    validation_file: str | None,
    test_file: str | None,
    result_dir: str,
    experiment_name: str,
    train_iters: int,
    seq_length: int,
    micro_batch_size: int,
    global_batch_size: int,
    learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    eval_interval: int,
    eval_iters: int,
    seed: int,
    lora_dim: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: tuple[str, ...],
) -> Any:
    """Build the upstream classifier configuration with round-local checkpointing disabled."""

    if train_iters < 0:
        raise ValueError("train_iters must be non-negative.")
    if micro_batch_size <= 0 or global_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")
    if global_batch_size % micro_batch_size:
        raise ValueError("global_batch_size must be divisible by micro_batch_size on the one-GPU path.")

    config = upstream.evo2_1b_classifier_config(
        base_ckpt_dir=Path(base_checkpoint),
        train_jsonl=Path(train_file),
        val_jsonl=Path(validation_file) if validation_file else None,
        test_jsonl=Path(test_file) if test_file else None,
        num_classes=3,
        result_dir=Path(result_dir),
        experiment_name=experiment_name,
        model_size="evo2_1b_base",
        tensor_model_parallel_size=1,
        seq_length_tokens=seq_length,
        backbone_seq_length=8192,
        train_iters=train_iters,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        lr=learning_rate,
        min_lr=min_learning_rate,
        warmup_iters=warmup_iters,
        decay_steps=100000,
        weight_decay=0.0,
        eval_interval=max(1, eval_interval),
        eval_iters=max(1, eval_iters),
        save_interval=None,
        log_interval=_complete_log_interval(train_iters),
        seed=seed,
        classifier_dropout=0.1,
        pool="mean",
        use_lora=True,
        lora_dim=lora_dim,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        no_activation_checkpointing=True,
        precision_recipe="bf16_mixed",
        log_throughput_to_tensorboard=True,
    )
    # The federated state is persisted by NVFlare. Each task starts in a fresh
    # process from the immutable base checkpoint, so loading a prior local DCP
    # would restore stale adapter, optimizer, or RNG state from another round.
    config.checkpoint.load = None
    config.checkpoint.save = None
    config.checkpoint.save_interval = None
    config.checkpoint.load_optim = False
    config.checkpoint.load_rng = False
    config.checkpoint.save_optim = False
    config.checkpoint.save_rng = False
    config.checkpoint.async_save = False
    return config


def _metric_name(prefix: str, name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return f"{prefix}_{normalized}"


def _metric_value(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().item())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def validate_model_trainable_boundary(model) -> dict[str, int]:
    """Require every gradient-enabled parameter to belong to the federated state."""

    if isinstance(model, (list, tuple)):
        if len(model) != 1:
            raise ValueError(f"The one-GPU Evo2 path requires one model chunk, received {len(model)}.")
        model = model[0]
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    unexpected = [name for name, _parameter in trainable if not adapter_checkpoint.is_trainable_parameter(name)]
    if unexpected:
        raise ValueError(
            "Evo2 exposes trainable parameters outside the LoRA/head federation boundary: "
            f"{unexpected}. Check the pinned PEFT configuration before training."
        )
    head_names = [
        name
        for name, _parameter in trainable
        if adapter_checkpoint.CLASSIFICATION_HEAD_MARKER in f".{name.strip('.')}."
    ]
    lora_names = [name for name, _parameter in trainable if name not in head_names]
    if not head_names:
        raise ValueError("Evo2 exposes no trainable classification-head parameters.")
    if not lora_names:
        raise ValueError(
            "Evo2 exposes no trainable LoRA adapter parameters. Check --lora-target-modules against the pinned model."
        )
    trainable_parameters = [parameter for _name, parameter in trainable]
    frozen_parameters = [parameter for parameter in model.parameters() if not parameter.requires_grad]
    return {
        "trainable_tensors": len(trainable_parameters),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable_parameters),
        "lora_tensors": len(lora_names),
        "classification_head_tensors": len(head_names),
        "frozen_parameters": sum(parameter.numel() for parameter in frozen_parameters),
    }


def fingerprint_frozen_parameters(model) -> str:
    """Return a deterministic SHA-256 digest of every frozen model parameter."""

    if isinstance(model, (list, tuple)):
        if len(model) != 1:
            raise ValueError(f"The one-GPU Evo2 path requires one model chunk, received {len(model)}.")
        model = model[0]
    digest = hashlib.sha256()
    frozen_tensors = 0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                continue
            frozen_tensors += 1
            header = f"{name}\0{parameter.dtype}\0{tuple(parameter.shape)}\0".encode()
            digest.update(header)
            value = parameter.detach().contiguous().view(torch.uint8).cpu().numpy()
            digest.update(memoryview(value))
    if not frozen_tensors:
        raise ValueError("Evo2 exposes no frozen parameters to verify.")
    return digest.hexdigest()


def make_exchange_callback(
    incoming_state: Mapping[str, torch.Tensor] | None,
    *,
    extract_after_training: bool,
    load_on_data_init: bool = False,
    train_sample_offset: int | None = None,
):
    """Create a Megatron Bridge callback that imports and exports trainable tensors."""

    if train_sample_offset is not None and train_sample_offset < 0:
        raise ValueError("train_sample_offset must be non-negative.")
    from megatron.bridge.training.callbacks import Callback
    from megatron.core.utils import unwrap_model

    class NVFlareExchangeCallback(Callback):
        def __init__(self):
            self.initial_state = None
            self.updated_state = None
            self.metrics: dict[str, float] = {}
            self.frozen_fingerprint = None

        def on_data_init_start(self, context) -> None:
            if train_sample_offset is not None:
                # Every FL task is a fresh process, so MBridge otherwise starts its
                # cyclic sampler from consumed_train_samples=0 on every round. The
                # hook fires immediately before the loaders are built, making this
                # the narrowest place to resume only the data cursor while keeping
                # optimizer, scheduler, and RNG checkpoint state local and fresh.
                context.state.train_state.consumed_train_samples = train_sample_offset
                self.metrics["train_sample_offset"] = float(train_sample_offset)
            model = unwrap_model(context.model)
            boundary = validate_model_trainable_boundary(model)
            self.metrics.update({name: float(value) for name, value in boundary.items()})
            self.initial_state = adapter_checkpoint.extract_trainable_state(model)
            if incoming_state is not None and load_on_data_init:
                adapter_checkpoint.validate_trainable_state(
                    incoming_state, self.initial_state, context="NVFlare global trainable state"
                )
                adapter_checkpoint.load_trainable_state(model, incoming_state)
                self.initial_state = adapter_checkpoint.extract_trainable_state(model)

        def on_train_start(self, context) -> None:
            if incoming_state is None or load_on_data_init:
                return
            model = unwrap_model(context.model)
            adapter_checkpoint.validate_trainable_state(
                incoming_state, self.initial_state, context="NVFlare global trainable state"
            )
            adapter_checkpoint.load_trainable_state(model, incoming_state)
            # Megatron's distributed optimizer keeps FP32 master parameters. Copying only
            # the BF16 model values would let those stale master values overwrite the
            # received global state on the first optimizer step.
            context.optimizer.reload_model_params()
            self.initial_state = adapter_checkpoint.extract_trainable_state(model)
            self.frozen_fingerprint = fingerprint_frozen_parameters(model)

        def on_train_step_end(self, context) -> None:
            if context.loss_dict:
                for name, value in context.loss_dict.items():
                    metric = _metric_value(value)
                    if metric is not None:
                        self.metrics[_metric_name("train", name)] = metric
            if context.grad_norm is not None:
                self.metrics["train_grad_norm"] = float(context.grad_norm)

        def on_train_end(self, context) -> None:
            if extract_after_training:
                model = unwrap_model(context.model)
                trained_fingerprint = fingerprint_frozen_parameters(model)
                if trained_fingerprint != self.frozen_fingerprint:
                    raise RuntimeError("A frozen Evo2 backbone parameter changed during local training.")
                self.metrics["frozen_parameters_unchanged"] = 1.0
                self.updated_state = adapter_checkpoint.extract_trainable_state(model)

        def on_eval_end(self, context) -> None:
            if context.total_loss_dict:
                for name, value in context.total_loss_dict.items():
                    metric = _metric_value(value)
                    if metric is not None:
                        self.metrics[_metric_name("validation", name)] = metric

        def on_test_end(self, context) -> None:
            if context.total_loss_dict:
                for name, value in context.total_loss_dict.items():
                    metric = _metric_value(value)
                    if metric is not None:
                        self.metrics[_metric_name("test", name)] = metric

    return NVFlareExchangeCallback()


def initialize_trainable_state(
    *,
    classifier_path: str | None,
    base_checkpoint: str,
    data_file: str,
    result_dir: str,
    seq_length: int,
    seed: int,
    lora_dim: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: tuple[str, ...],
) -> Mapping[str, torch.Tensor]:
    """Instantiate Evo2 once and return its deterministic initial LoRA/head state."""

    upstream = load_classifier_module(classifier_path)
    callback = make_exchange_callback(None, extract_after_training=False)
    config = build_classifier_config(
        upstream,
        base_checkpoint=base_checkpoint,
        train_file=data_file,
        validation_file=None,
        test_file=None,
        result_dir=result_dir,
        experiment_name="initialize",
        train_iters=0,
        seq_length=seq_length,
        micro_batch_size=1,
        global_batch_size=1,
        learning_rate=1e-6,
        min_learning_rate=1e-6,
        warmup_iters=0,
        eval_interval=1,
        eval_iters=1,
        seed=seed,
        lora_dim=lora_dim,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
    )
    upstream.pretrain(config, upstream.classifier_forward_step, callbacks=[callback])
    if callback.initial_state is None:
        raise RuntimeError("Megatron Bridge completed without exposing the initial trainable state.")
    return callback.initial_state


def train_round(
    incoming_state: Mapping[str, torch.Tensor],
    *,
    classifier_path: str | None,
    base_checkpoint: str,
    train_file: str,
    validation_file: str,
    result_dir: str,
    local_steps: int,
    seq_length: int,
    micro_batch_size: int,
    global_batch_size: int,
    learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    eval_iters: int,
    seed: int,
    round_index: int,
    lora_dim: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: tuple[str, ...],
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor], dict[str, float]]:
    """Return the rebased full state, exact model-boundary DIFF, and metrics for one local round."""

    if round_index < 0:
        raise ValueError("round_index must be non-negative.")
    if local_steps <= 0:
        raise ValueError("local_steps must be positive.")
    upstream = load_classifier_module(classifier_path)
    config = build_classifier_config(
        upstream,
        base_checkpoint=base_checkpoint,
        train_file=train_file,
        validation_file=validation_file,
        test_file=None,
        result_dir=result_dir,
        experiment_name="local_train",
        train_iters=local_steps,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        learning_rate=learning_rate,
        min_learning_rate=min_learning_rate,
        warmup_iters=warmup_iters,
        eval_interval=local_steps,
        eval_iters=eval_iters,
        seed=seed,
        lora_dim=lora_dim,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
    )
    train_sample_offset = round_train_sample_offset(round_index, local_steps, global_batch_size)
    callback = make_exchange_callback(
        incoming_state,
        extract_after_training=True,
        train_sample_offset=train_sample_offset,
    )
    upstream.pretrain(config, upstream.classifier_forward_step, callbacks=[callback])
    if callback.updated_state is None:
        raise RuntimeError("Megatron Bridge completed without exporting the trained LoRA/head state.")
    if callback.initial_state is None:
        raise RuntimeError("Megatron Bridge completed without capturing the loaded LoRA/head baseline.")
    adapter_checkpoint.validate_trainable_state(
        callback.updated_state,
        callback.initial_state,
        context="Locally trained model state",
    )
    # The FP32 server state can retain residuals that are not representable by the
    # BF16 model. Measure local training relative to the rounded model baseline,
    # then apply that delta to the original FP32 state. A no-op local round thus
    # returns the exact incoming state instead of erasing its residuals.
    model_delta = adapter_checkpoint.compute_trainable_diff(callback.updated_state, callback.initial_state)
    updated_state = adapter_checkpoint.apply_trainable_diff(incoming_state, model_delta)
    return updated_state, model_delta, callback.metrics
