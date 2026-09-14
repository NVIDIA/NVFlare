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
import json
import os
import re
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import evo2_adapter_checkpoint as adapter_checkpoint
import provenance
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
TRAINING_STATE_FORMAT_VERSION = 1
TRAINING_STATE_MANIFEST_FILENAME = "nvflare_training_state.json"
NATIVE_TRAIN_STATE_FILENAME = "train_state.pt"
NATIVE_LATEST_TRAIN_STATE_FILENAME = f"latest_{NATIVE_TRAIN_STATE_FILENAME}"

_TRAINING_STATE_MANIFEST_KEYS = {
    "format_version",
    "status",
    "checkpoint_format",
    "site_name",
    "round_index",
    "previous_manifest_sha256",
    "configuration",
    "configuration_sha256",
    "trainable_schema",
    "trainable_schema_sha256",
    "incoming_global_state_sha256",
    "updated_local_state_sha256",
    "counters",
    "payload",
}


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


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trainable_schema_from_validated_state(state: Mapping[str, torch.Tensor]) -> list[dict[str, Any]]:
    return [{"name": name, "shape": list(state[name].shape), "dtype": str(state[name].dtype)} for name in sorted(state)]


def _trainable_schema(state: Mapping[str, torch.Tensor]) -> list[dict[str, Any]]:
    adapter_checkpoint.ValidatedTrainableStateSchema(state, context="Persistent training-state schema")
    return _trainable_schema_from_validated_state(state)


def _trainable_state_sha256_from_validated_state(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name in sorted(state):
            tensor = state[name].detach().contiguous().cpu()
            digest.update(f"{name}\0{tensor.dtype}\0{tuple(tensor.shape)}\0".encode())
            digest.update(memoryview(tensor.view(torch.uint8).numpy()))
    return digest.hexdigest()


def _trainable_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    adapter_checkpoint.ValidatedTrainableStateSchema(state, context="Persistent training-state digest")
    return _trainable_state_sha256_from_validated_state(state)


def _counter_value(value: Any, label: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RuntimeError(f"{label} must be a scalar, received tensor shape {tuple(value.shape)}.")
        value = value.detach().cpu().item()
    if type(value) is not int:
        raise RuntimeError(f"{label} must be an integer, received {value!r}.")
    if value < 0:
        raise RuntimeError(f"{label} must be non-negative, received {value}.")
    return value


def _read_native_train_state(path: Path) -> dict[str, int]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Megatron Bridge train-state file is missing or unsafe: {path}")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"Could not safely load Megatron Bridge train state {path}: {exc}") from exc
    if not isinstance(state, Mapping):
        raise RuntimeError(f"Megatron Bridge train state {path} must contain a mapping.")
    required = ("step", "consumed_train_samples", "skipped_train_samples")
    missing = [name for name in required if name not in state]
    if missing:
        raise RuntimeError(f"Megatron Bridge train state {path} is missing counters: {missing}.")
    return {name: _counter_value(state[name], f"{path}:{name}") for name in required}


def _expected_counters(round_index: int, local_steps: int, global_batch_size: int) -> dict[str, int]:
    start_step = round_index * local_steps
    end_step = (round_index + 1) * local_steps
    return {
        "start_step": start_step,
        "end_step": end_step,
        "start_train_samples": start_step * global_batch_size,
        "end_train_samples": end_step * global_batch_size,
        "start_scheduler_steps": start_step * global_batch_size,
        "end_scheduler_steps": end_step * global_batch_size,
        "start_skipped_train_samples": 0,
        "end_skipped_train_samples": 0,
    }


def _checkpoint_format(config: Any) -> str:
    value = getattr(config.checkpoint, "ckpt_format", None)
    if hasattr(value, "value"):
        value = value.value
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Megatron Bridge returned an invalid checkpoint format: {value!r}.")
    return value


def _persistent_configuration(
    *,
    base_checkpoint: str,
    classifier_path: str,
    train_file: str,
    validation_file: str,
    local_steps: int,
    seq_length: int,
    micro_batch_size: int,
    global_batch_size: int,
    learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    eval_iters: int,
    seed: int,
    peft_mode: str,
    lora_dim: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: tuple[str, ...],
    checkpoint_format: str,
) -> dict[str, Any]:
    """Return the round-invariant settings bound to one site's native state chain."""

    return {
        "base_checkpoint": provenance.directory_identity(base_checkpoint),
        "classifier_file": provenance.file_identity(classifier_path),
        "train_file": provenance.jsonl_identity(train_file, label="persistent training input"),
        "validation_file": provenance.jsonl_identity(validation_file, label="persistent validation input"),
        "local_steps": local_steps,
        "seq_length": seq_length,
        "micro_batch_size": micro_batch_size,
        "global_batch_size": global_batch_size,
        "learning_rate": learning_rate,
        "min_learning_rate": min_learning_rate,
        "warmup_iters": warmup_iters,
        "eval_iters": eval_iters,
        "seed": seed,
        "peft_mode": peft_mode,
        "lora_dim": lora_dim,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_target_modules": list(lora_target_modules),
        "checkpoint_format": checkpoint_format,
        "scheduler_decay_steps": 100000,
    }


def _read_state_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Persistent client training-state manifest is missing or unsafe: {path}")
    try:
        with path.open(encoding="utf-8") as file:
            manifest = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read persistent client training-state manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict) or set(manifest) != _TRAINING_STATE_MANIFEST_KEYS:
        received = set(manifest) if isinstance(manifest, dict) else type(manifest).__name__
        raise RuntimeError(f"Persistent client training-state manifest {path} has invalid keys: {received}.")
    return manifest


def _checkpoint_payload_identity(checkpoint_dir: Path) -> list[dict[str, Any]]:
    identities = []
    for path in sorted(checkpoint_dir.rglob("*")):
        relative_path = path.relative_to(checkpoint_dir).as_posix()
        if relative_path == TRAINING_STATE_MANIFEST_FILENAME:
            continue
        if path.is_symlink():
            raise RuntimeError(f"Persistent client training state cannot contain a symbolic link: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError(f"Persistent client training state contains an unsupported entry: {path}")
        identities.append({"path": relative_path, "bytes": path.stat().st_size, "sha256": _file_sha256(path)})
    if not identities:
        raise RuntimeError(f"Megatron Bridge did not write a checkpoint payload under {checkpoint_dir}.")
    return identities


def _validate_native_checkpoint(checkpoint_dir: Path, expected_end_step: int, expected_end_samples: int) -> None:
    expected_state = {
        "step": expected_end_step,
        "consumed_train_samples": expected_end_samples,
        "skipped_train_samples": 0,
    }
    iteration_dir = checkpoint_dir / f"iter_{expected_end_step:07d}"
    latest_state = _read_native_train_state(checkpoint_dir / NATIVE_LATEST_TRAIN_STATE_FILENAME)
    iteration_state = _read_native_train_state(iteration_dir / NATIVE_TRAIN_STATE_FILENAME)
    if latest_state != expected_state:
        raise RuntimeError(
            f"Megatron Bridge latest train-state counters do not match the completed round: "
            f"expected {expected_state}, received {latest_state}."
        )
    if iteration_state != expected_state:
        raise RuntimeError(
            f"Megatron Bridge iteration train-state counters do not match the completed round: "
            f"expected {expected_state}, received {iteration_state}."
        )


def _validate_state_manifest(
    manifest: dict[str, Any],
    *,
    path: Path,
    site_name: str,
    round_index: int,
    previous_manifest_sha256: str | None,
    configuration: dict[str, Any],
    configuration_sha256: str,
    trainable_schema: list[dict[str, Any]],
    trainable_schema_sha256: str,
    counters: dict[str, int],
    checkpoint_format: str,
) -> None:
    expected_values = {
        "format_version": TRAINING_STATE_FORMAT_VERSION,
        "status": "complete",
        "checkpoint_format": checkpoint_format,
        "site_name": site_name,
        "round_index": round_index,
        "previous_manifest_sha256": previous_manifest_sha256,
        "configuration": configuration,
        "configuration_sha256": configuration_sha256,
        "trainable_schema": trainable_schema,
        "trainable_schema_sha256": trainable_schema_sha256,
        "counters": counters,
    }
    mismatches = {
        name: {"expected": expected, "received": manifest.get(name)}
        for name, expected in expected_values.items()
        if manifest.get(name) != expected
    }
    for name in ("incoming_global_state_sha256", "updated_local_state_sha256"):
        value = manifest.get(name)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            mismatches[name] = {"expected": "64 lowercase hexadecimal characters", "received": value}
    payload = manifest.get("payload")
    if not isinstance(payload, list) or not payload:
        mismatches["payload"] = {"expected": "non-empty file identity list", "received": payload}
    if mismatches:
        raise RuntimeError(f"Persistent client training-state manifest {path} is incompatible: {mismatches}.")


def _prepare_persistent_training_state(
    training_state_dir: str,
    *,
    site_name: str,
    round_index: int,
    local_steps: int,
    global_batch_size: int,
    configuration: dict[str, Any],
    incoming_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    if not site_name:
        raise ValueError("site_name is required when persistent client training state is enabled.")
    state_root = Path(training_state_dir).expanduser().resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    stable_dir = state_root / f"round_{round_index:03d}"
    if stable_dir.exists():
        raise RuntimeError(
            f"Persistent client training state for {site_name} round {round_index} already exists at {stable_dir}."
        )
    for path in state_root.glob("round_*"):
        match = re.fullmatch(r"round_([0-9]+)", path.name)
        if match and int(match.group(1)) >= round_index:
            raise RuntimeError(f"Unexpected future persistent client training state exists at {path}.")

    checkpoint_format = configuration["checkpoint_format"]
    configuration_sha256 = _json_sha256(configuration)
    trainable_schema = _trainable_schema(incoming_state)
    trainable_schema_sha256 = _json_sha256(trainable_schema)
    previous_manifest_sha256 = None
    previous_dir = None
    counters = _expected_counters(round_index, local_steps, global_batch_size)
    for completed_round in range(round_index):
        completed_dir = state_root / f"round_{completed_round:03d}"
        manifest_path = completed_dir / TRAINING_STATE_MANIFEST_FILENAME
        manifest = _read_state_manifest(manifest_path)
        _validate_state_manifest(
            manifest,
            path=manifest_path,
            site_name=site_name,
            round_index=completed_round,
            previous_manifest_sha256=previous_manifest_sha256,
            configuration=configuration,
            configuration_sha256=configuration_sha256,
            trainable_schema=trainable_schema,
            trainable_schema_sha256=trainable_schema_sha256,
            counters=_expected_counters(completed_round, local_steps, global_batch_size),
            checkpoint_format=checkpoint_format,
        )
        previous_manifest_sha256 = _file_sha256(manifest_path)
        previous_dir = completed_dir

    if round_index > 0:
        if previous_dir is None:
            raise RuntimeError(f"Persistent client training state for {site_name} round {round_index - 1} is missing.")
        previous_manifest = _read_state_manifest(previous_dir / TRAINING_STATE_MANIFEST_FILENAME)
        observed_payload = _checkpoint_payload_identity(previous_dir)
        if observed_payload != previous_manifest["payload"]:
            raise RuntimeError(f"Persistent client training-state payload was modified after round {round_index - 1}.")
        _validate_native_checkpoint(previous_dir, counters["start_step"], counters["start_train_samples"])

    candidate_dir = Path(tempfile.mkdtemp(prefix=f".round_{round_index:03d}_", dir=state_root))
    return {
        "state_root": state_root,
        "stable_dir": stable_dir,
        "candidate_dir": candidate_dir,
        "load_dir": previous_dir,
        "site_name": site_name,
        "round_index": round_index,
        "configuration": configuration,
        "configuration_sha256": configuration_sha256,
        "trainable_schema": trainable_schema,
        "trainable_schema_sha256": trainable_schema_sha256,
        "previous_manifest_sha256": previous_manifest_sha256,
        "counters": counters,
        "checkpoint_format": checkpoint_format,
    }


def _configure_persistent_checkpoint(config: Any, state_plan: dict[str, Any]) -> None:
    """Enable native MBridge optimizer, scheduler, and RNG save/resume for one site."""

    resume = state_plan["load_dir"] is not None
    config.checkpoint.load = str(state_plan["load_dir"]) if resume else None
    config.checkpoint.save = str(state_plan["candidate_dir"])
    config.checkpoint.save_interval = None
    config.checkpoint.load_optim = resume
    config.checkpoint.load_rng = resume
    config.checkpoint.save_optim = True
    config.checkpoint.save_rng = True
    config.checkpoint.async_save = False
    config.checkpoint.exit_on_missing_checkpoint = resume


def _shutdown_mcore_checkpoint_results_manager() -> None:
    """Stop the process behind MCore's synchronous ``torch_dist`` results queue.

    MCore implements a synchronous ``torch_dist`` save by executing its async
    writer in the training process. The writer still creates a module-global
    ``SyncManager().Queue()``. NVFlare's one-task external client exits with
    ``os._exit`` after its result is downloaded, so Python's normal manager
    finalizer cannot be relied on to stop that child process.
    """

    from megatron.core.dist_checkpointing.strategies import filesystem_async

    results_queue = getattr(filesystem_async, "_results_queue", None)
    if results_queue is None:
        return
    manager = getattr(results_queue, "_manager", None)
    shutdown = getattr(manager, "shutdown", None)
    process = getattr(manager, "_process", None)
    if manager is None or not callable(shutdown) or process is None or not callable(getattr(process, "is_alive", None)):
        raise RuntimeError("MCore checkpoint results queue does not expose a verifiable manager process for shutdown.")

    shutdown_error = None
    try:
        shutdown()
    except Exception as exc:
        shutdown_error = exc

    try:
        process_alive = process.is_alive()
    except Exception as exc:
        raise RuntimeError("Could not verify that the MCore checkpoint results manager stopped.") from exc
    if process_alive:
        raise RuntimeError("MCore checkpoint results manager is still alive after shutdown.") from shutdown_error

    filesystem_async._results_queue = None
    if shutdown_error is not None:
        raise RuntimeError("MCore checkpoint results manager shutdown raised an exception.") from shutdown_error


def _write_state_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("wb") as file:
        file.write(json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))
        file.write(b"\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)


def _commit_persistent_training_state(
    state_plan: dict[str, Any],
    *,
    incoming_state: Mapping[str, torch.Tensor],
    updated_state: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    candidate_dir = state_plan["candidate_dir"]
    counters = state_plan["counters"]
    _validate_native_checkpoint(candidate_dir, counters["end_step"], counters["end_train_samples"])
    payload = _checkpoint_payload_identity(candidate_dir)
    manifest = {
        "format_version": TRAINING_STATE_FORMAT_VERSION,
        "status": "complete",
        "checkpoint_format": state_plan["checkpoint_format"],
        "site_name": state_plan["site_name"],
        "round_index": state_plan["round_index"],
        "previous_manifest_sha256": state_plan["previous_manifest_sha256"],
        "configuration": state_plan["configuration"],
        "configuration_sha256": state_plan["configuration_sha256"],
        "trainable_schema": state_plan["trainable_schema"],
        "trainable_schema_sha256": state_plan["trainable_schema_sha256"],
        "incoming_global_state_sha256": _trainable_state_sha256(incoming_state),
        "updated_local_state_sha256": _trainable_state_sha256(updated_state),
        "counters": counters,
        "payload": payload,
    }
    manifest_path = candidate_dir / TRAINING_STATE_MANIFEST_FILENAME
    _write_state_manifest(manifest_path, manifest)
    total_bytes = sum(item["bytes"] for item in payload) + manifest_path.stat().st_size
    stable_dir = state_plan["stable_dir"]
    os.replace(candidate_dir, stable_dir)
    try:
        directory_fd = os.open(state_plan["state_root"], os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        # The same-filesystem rename above is the atomic visibility boundary;
        # not every supported filesystem permits fsync on a directory.
        pass
    return {
        "persistent_training_state": 1.0,
        "persistent_training_state_resumed": float(state_plan["load_dir"] is not None),
        "persistent_training_state_mebibytes": total_bytes / (1024 * 1024),
    }


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
    peft_mode: str,
    lora_dim: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: tuple[str, ...],
) -> Any:
    """Build the upstream classifier configuration with round-local checkpointing disabled."""

    if peft_mode not in ("lora", "head-only"):
        raise ValueError(f"Unsupported peft_mode {peft_mode!r}; expected 'lora' or 'head-only'.")
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
        use_lora=peft_mode == "lora",
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


def validate_model_trainable_boundary(model, peft_mode: str) -> dict[str, int]:
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
    if peft_mode == "lora" and not lora_names:
        raise ValueError(
            "LoRA mode produced no trainable adapter parameters. Check --lora-target-modules against the pinned model."
        )
    if peft_mode == "head-only" and lora_names:
        raise ValueError(f"Head-only mode unexpectedly produced trainable LoRA parameters: {lora_names}.")
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
    peft_mode: str,
    load_on_data_init: bool = False,
    train_sample_offset: int | None = None,
    expected_counters: Mapping[str, int] | None = None,
):
    """Create a Megatron Bridge callback that imports and exports trainable tensors."""

    if train_sample_offset is not None and train_sample_offset < 0:
        raise ValueError("train_sample_offset must be non-negative.")
    if train_sample_offset is not None and expected_counters is not None:
        raise ValueError("Manual sample offsets and persistent training-state counters are mutually exclusive.")
    if expected_counters is not None:
        required_counter_names = set(_expected_counters(0, 1, 1))
        if set(expected_counters) != required_counter_names:
            raise ValueError(
                "expected_counters must contain exactly the persistent training-state counter schema; "
                f"received {sorted(expected_counters)}."
            )

    from megatron.bridge.training.callbacks import Callback
    from megatron.core.utils import unwrap_model

    class NVFlareExchangeCallback(Callback):
        def __init__(self):
            self.initial_state = None
            self.updated_state = None
            self.metrics: dict[str, float] = {}
            self.frozen_fingerprint = None

        def _validate_persistent_counters(self, context, phase: str) -> None:
            if expected_counters is None:
                return
            if phase == "start":
                expected_step = expected_counters["start_step"]
                expected_samples = expected_counters["start_train_samples"]
                expected_skipped = expected_counters["start_skipped_train_samples"]
                expected_scheduler = expected_counters["start_scheduler_steps"]
            else:
                expected_step = expected_counters["end_step"]
                expected_samples = expected_counters["end_train_samples"]
                expected_skipped = expected_counters["end_skipped_train_samples"]
                expected_scheduler = expected_counters["end_scheduler_steps"]
            train_state = context.state.train_state
            observed = {
                "step": _counter_value(train_state.step, f"{phase} train step"),
                "consumed_train_samples": _counter_value(
                    train_state.consumed_train_samples, f"{phase} consumed train samples"
                ),
                "skipped_train_samples": _counter_value(
                    train_state.skipped_train_samples, f"{phase} skipped train samples"
                ),
                "scheduler_steps": _counter_value(context.scheduler.num_steps, f"{phase} scheduler steps"),
            }
            expected = {
                "step": expected_step,
                "consumed_train_samples": expected_samples,
                "skipped_train_samples": expected_skipped,
                "scheduler_steps": expected_scheduler,
            }
            if observed != expected:
                raise RuntimeError(
                    f"Persistent client training-state counters are invalid at {phase}: "
                    f"expected {expected}, received {observed}."
                )
            for name, value in observed.items():
                self.metrics[f"training_state_{phase}_{name}"] = float(value)

        def on_data_init_start(self, context) -> None:
            if train_sample_offset is not None:
                # Every FL task is a fresh process, so MBridge otherwise starts its
                # cyclic sampler from consumed_train_samples=0 on every round. The
                # hook fires immediately before the loaders are built, making this
                # the narrowest place to resume only the data cursor while keeping
                # optimizer, scheduler, and RNG checkpoint state local and fresh.
                context.state.train_state.consumed_train_samples = train_sample_offset
                self.metrics["train_sample_offset"] = float(train_sample_offset)
            self._validate_persistent_counters(context, "start")
            model = unwrap_model(context.model)
            boundary = validate_model_trainable_boundary(model, peft_mode)
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
                self._validate_persistent_counters(context, "end")
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
    peft_mode: str,
    lora_dim: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: tuple[str, ...],
) -> Mapping[str, torch.Tensor]:
    """Instantiate Evo2 once and return its deterministic initial LoRA/head state."""

    upstream = load_classifier_module(classifier_path)
    callback = make_exchange_callback(None, extract_after_training=False, peft_mode=peft_mode)
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
        peft_mode=peft_mode,
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
    peft_mode: str,
    lora_dim: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: tuple[str, ...],
    training_state_dir: str | None = None,
    site_name: str | None = None,
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor], dict[str, float]]:
    """Return the rebased full state, exact model-boundary DIFF, and metrics for one local round."""

    if round_index < 0:
        raise ValueError("round_index must be non-negative.")
    if local_steps <= 0:
        raise ValueError("local_steps must be positive.")
    upstream = load_classifier_module(classifier_path)
    persistent = training_state_dir is not None
    train_target_steps = (round_index + 1) * local_steps if persistent else local_steps
    config = build_classifier_config(
        upstream,
        base_checkpoint=base_checkpoint,
        train_file=train_file,
        validation_file=validation_file,
        test_file=None,
        result_dir=result_dir,
        experiment_name="local_train",
        train_iters=train_target_steps,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        learning_rate=learning_rate,
        min_learning_rate=min_learning_rate,
        warmup_iters=warmup_iters,
        eval_interval=local_steps,
        eval_iters=eval_iters,
        seed=seed,
        peft_mode=peft_mode,
        lora_dim=lora_dim,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
    )
    state_plan = None
    if persistent:
        resolved_classifier_path = getattr(upstream, "__file__", None) or classifier_path or DEFAULT_CLASSIFIER_PATH
        configuration = _persistent_configuration(
            base_checkpoint=base_checkpoint,
            classifier_path=resolved_classifier_path,
            train_file=train_file,
            validation_file=validation_file,
            local_steps=local_steps,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            learning_rate=learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_iters=warmup_iters,
            eval_iters=eval_iters,
            seed=seed,
            peft_mode=peft_mode,
            lora_dim=lora_dim,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            checkpoint_format=_checkpoint_format(config),
        )
        state_plan = _prepare_persistent_training_state(
            training_state_dir,
            site_name=site_name or "",
            round_index=round_index,
            local_steps=local_steps,
            global_batch_size=global_batch_size,
            configuration=configuration,
            incoming_state=incoming_state,
        )
        _configure_persistent_checkpoint(config, state_plan)
        train_sample_offset = None
        expected_counters = state_plan["counters"]
    else:
        train_sample_offset = round_train_sample_offset(round_index, local_steps, global_batch_size)
        expected_counters = None
    callback = make_exchange_callback(
        incoming_state,
        extract_after_training=True,
        peft_mode=peft_mode,
        train_sample_offset=train_sample_offset,
        expected_counters=expected_counters,
    )
    pretrain_error = None
    try:
        upstream.pretrain(config, upstream.classifier_forward_step, callbacks=[callback])
    except BaseException as exc:
        pretrain_error = exc
        raise
    finally:
        if persistent:
            try:
                _shutdown_mcore_checkpoint_results_manager()
            except Exception as cleanup_error:
                if pretrain_error is None:
                    raise
                message = f"MCore checkpoint worker cleanup also failed: {cleanup_error}"
                if hasattr(pretrain_error, "add_note"):
                    pretrain_error.add_note(message)
                print(message, file=sys.stderr, flush=True)
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
    if state_plan is not None:
        callback.metrics.update(
            _commit_persistent_training_state(
                state_plan,
                incoming_state=incoming_state,
                updated_state=updated_state,
            )
        )
    return updated_state, model_delta, callback.metrics
