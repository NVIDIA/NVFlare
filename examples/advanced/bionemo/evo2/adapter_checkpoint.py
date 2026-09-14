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
"""PyTorch-only helpers for exchanging Evo2 LoRA and classification-head parameters."""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import nn

LORA_PARAMETER_MARKERS = (".adapter.",)
CLASSIFICATION_HEAD_MARKER = ".classification_head."
DEFAULT_TRAIN_CONF = {"train": {"model": "Evo2LoRAClassifier"}}
EXCHANGE_DTYPE = torch.float32
EXCHANGE_DTYPE_NAME = "float32"


class _IncompatibleKeys(NamedTuple):
    """Return value compatible with ``torch.nn.Module.load_state_dict``."""

    missing_keys: list[str]
    unexpected_keys: list[str]


class ValidatedTrainableStateSchema:
    """Reusable exact tensor schema built from a fully validated reference state."""

    def __init__(self, reference: Mapping[str, torch.Tensor], *, context: str = "Reference trainable state"):
        reference_tensors = _validate_exchange_mapping(reference, context)
        self._specs = tuple((name, tuple(tensor.shape), tensor.dtype) for name, tensor in reference_tensors.items())
        self._keys = frozenset(reference_tensors)

    def validate(
        self, incoming: Mapping[str, torch.Tensor], *, context: str = "Incoming trainable state"
    ) -> OrderedDict[str, torch.Tensor]:
        """Validate exact keys, shapes, dtypes, and finite values without rescanning the reference."""

        incoming_tensors = _tensor_mapping(incoming, context)
        incoming_keys = set(incoming_tensors)
        missing = sorted(self._keys - incoming_keys)
        unexpected = sorted(incoming_keys - self._keys)
        if missing or unexpected:
            raise KeyError(f"{context} keys do not match reference; missing={missing}, unexpected={unexpected}.")

        errors = []
        for name, expected_shape, expected_dtype in self._specs:
            received = incoming_tensors[name]
            if tuple(received.shape) != expected_shape:
                errors.append(f"{name!r} shape: expected {expected_shape}, received {tuple(received.shape)}")
            if received.dtype != expected_dtype:
                errors.append(f"{name!r} dtype: expected {expected_dtype}, received {received.dtype}")
        if errors:
            raise ValueError(f"{context} is incompatible with reference: " + "; ".join(errors) + ".")
        _validate_finite_tensors(incoming_tensors, context)
        return incoming_tensors


def is_trainable_parameter(name: str) -> bool:
    """Return whether ``name`` belongs to Evo2 LoRA or its classification head.

    BioNeMo nests each LoRA tensor below an ``adapter`` module. Requiring that
    module-path segment keeps wrapper prefixes intact without treating ordinary
    backbone modules named ``linear_in`` or ``linear_out`` as adapters.
    """

    if not isinstance(name, str):
        raise TypeError(f"Parameter name must be a string, got {type(name).__name__}.")
    padded_name = f".{name.strip('.')}."
    return CLASSIFICATION_HEAD_MARKER in padded_name or any(marker in padded_name for marker in LORA_PARAMETER_MARKERS)


def _cpu_clone(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().clone()


def _model_boundary_clone(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not tensor.is_floating_point():
        raise ValueError(f"Trainable model parameter {name!r} must use a real floating dtype, received {tensor.dtype}.")
    return tensor.detach().to(device="cpu", dtype=EXCHANGE_DTYPE).clone()


def _single_module(model: nn.Module | list[nn.Module] | tuple[nn.Module, ...]) -> nn.Module:
    if isinstance(model, (list, tuple)):
        if len(model) != 1:
            raise ValueError(
                "The single-GPU Evo2 example supports exactly one model chunk; " f"received {len(model)} chunks."
            )
        model = model[0]
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected a torch.nn.Module or one-element model chunk sequence, got {type(model).__name__}.")
    return model


def _tensor_mapping(state: Mapping[str, torch.Tensor], label: str) -> OrderedDict[str, torch.Tensor]:
    if not isinstance(state, Mapping):
        raise TypeError(f"{label} must be a mapping, got {type(state).__name__}.")

    tensors = OrderedDict()
    for name, value in state.items():
        if not isinstance(name, str):
            raise TypeError(f"{label} contains a non-string key of type {type(name).__name__}.")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{label} parameter {name!r} must be a torch.Tensor, got {type(value).__name__}.")
        tensors[name] = value
    return tensors


def _validate_finite_tensors(tensors: Mapping[str, torch.Tensor], label: str) -> None:
    non_finite = []
    for name, tensor in tensors.items():
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor.detach()).all().item():
            non_finite.append(name)
    if non_finite:
        raise ValueError(f"{label} contains non-finite values in tensors: {non_finite}.")


def _validate_exchange_mapping(state: Mapping[str, torch.Tensor], label: str) -> OrderedDict[str, torch.Tensor]:
    tensors = _tensor_mapping(state, label)
    if not tensors:
        raise ValueError(f"{label} is empty; expected Evo2 LoRA or classification-head tensors.")

    unsupported = [name for name in tensors if not is_trainable_parameter(name)]
    if unsupported:
        raise ValueError(
            f"{label} contains unsupported parameter names: {unsupported}. "
            "Only parameters below a LoRA adapter module or classification_head may be exchanged."
        )
    incompatible_dtypes = {
        name: str(tensor.dtype) for name, tensor in tensors.items() if tensor.dtype != EXCHANGE_DTYPE
    }
    if incompatible_dtypes:
        raise ValueError(
            f"{label} must contain only {EXCHANGE_DTYPE_NAME} tensors; received incompatible dtypes: "
            f"{incompatible_dtypes}."
        )
    _validate_finite_tensors(tensors, label)
    return tensors


def extract_trainable_state(
    source: nn.Module | list[nn.Module] | tuple[nn.Module, ...] | Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Extract the federated Evo2 parameters as detached CPU float32 clones.

    For modules, only supported parameters with ``requires_grad=True`` are
    included. Mapping inputs have no gradient metadata, so supported names are
    selected directly. A one-element model-chunk list or tuple is accepted for
    compatibility with Megatron Bridge; pipeline-parallel chunks are outside the
    scope of this single-GPU example.
    """

    selected = OrderedDict()
    if isinstance(source, Mapping):
        for name, value in source.items():
            if not isinstance(name, str):
                raise TypeError(f"State mapping contains a non-string key of type {type(name).__name__}.")
            if not is_trainable_parameter(name):
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Trainable state value for {name!r} must be a torch.Tensor, got {type(value).__name__}."
                )
            selected[name] = _model_boundary_clone(value, name)
    else:
        model = _single_module(source)
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and is_trainable_parameter(name):
                selected[name] = _model_boundary_clone(parameter, name)

    if not selected:
        raise ValueError(
            "No trainable Evo2 parameters found. Expected requires-grad parameters below a LoRA adapter module "
            "or classification_head."
        )
    return _validate_exchange_mapping(selected, "Extracted trainable state")


def copy_trainable_state(
    state: Mapping[str, torch.Tensor], *, context: str = "Trainable state"
) -> OrderedDict[str, torch.Tensor]:
    """Strictly validate an exchange payload and return detached CPU clones.

    Unlike :func:`extract_trainable_state`, this helper never filters a state
    mapping. It is intended for trust boundaries such as an NVFlare receive,
    where an unexpected backbone key must make the whole payload fail.
    """

    tensors = _validate_exchange_mapping(state, context)
    return OrderedDict((name, _cpu_clone(tensor)) for name, tensor in tensors.items())


def validate_trainable_state(
    incoming: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
    *,
    context: str = "Incoming trainable state",
) -> None:
    """Validate exact keys, shapes, dtypes, and finite floating values against ``reference``."""

    ValidatedTrainableStateSchema(reference).validate(incoming, context=context)


def load_trainable_state(
    model: nn.Module | list[nn.Module] | tuple[nn.Module, ...], incoming: Mapping[str, torch.Tensor]
) -> int:
    """Strictly copy an incoming federated state into a single Evo2 model chunk.

    Validation completes before any parameter is modified, so a malformed update
    cannot partially change the model. Returns the number of copied parameters.
    """

    module = _single_module(model)
    reference = extract_trainable_state(module)
    validate_trainable_state(incoming, reference)
    parameters = OrderedDict(module.named_parameters())

    with torch.no_grad():
        for name in reference:
            parameters[name].copy_(
                incoming[name].detach().to(device=parameters[name].device, dtype=parameters[name].dtype)
            )
    return len(reference)


def compute_trainable_diff(
    current: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor]
) -> OrderedDict[str, torch.Tensor]:
    """Compute the NVFlare ``DIFF`` update ``current - reference`` on CPU."""

    validate_trainable_state(current, reference, context="Current trainable state")
    diff = OrderedDict(
        (name, _cpu_clone(current[name]) - _cpu_clone(reference_tensor)) for name, reference_tensor in reference.items()
    )
    return _validate_exchange_mapping(diff, "Computed trainable DIFF")


def apply_trainable_diff(
    reference: Mapping[str, torch.Tensor], diff: Mapping[str, torch.Tensor]
) -> OrderedDict[str, torch.Tensor]:
    """Apply an aggregated round ``DIFF`` to a reference state on CPU."""

    validate_trainable_state(diff, reference, context="Trainable DIFF")
    updated = OrderedDict(
        (name, _cpu_clone(reference_tensor) + _cpu_clone(diff[name])) for name, reference_tensor in reference.items()
    )
    return _validate_exchange_mapping(updated, "Updated trainable state")


def state_dict_size_mb(state: Mapping[str, torch.Tensor]) -> float:
    """Return the tensor payload size in mebibytes."""

    tensors = _tensor_mapping(state, "State dict")
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors.values()) / (1024 * 1024)


def _metadata_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _metadata_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata_safe(item) for item in value]
    if isinstance(value, set):
        return sorted((_metadata_safe(item) for item in value), key=repr)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def save_nvflare_checkpoint(
    state: Mapping[str, torch.Tensor], path: str | os.PathLike[str], metadata: Mapping[str, Any] | None = None
) -> None:
    """Save trainable tensors in ``PTFileModelPersistor`` checkpoint format."""

    tensors = _validate_exchange_mapping(state, "Trainable checkpoint state")
    checkpoint = OrderedDict(
        model=OrderedDict((name, _cpu_clone(tensor)) for name, tensor in tensors.items()),
        train_conf=DEFAULT_TRAIN_CONF,
    )
    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise TypeError(f"metadata must be a mapping, got {type(metadata).__name__}.")
        checkpoint["meta_props"] = _metadata_safe(metadata)

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)


def _load_nvflare_checkpoint_data(path: str | os.PathLike[str]) -> Mapping[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"NVFlare checkpoint does not exist: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "Safe checkpoint loading requires a PyTorch version with torch.load(..., weights_only=True) support."
        ) from exc

    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"NVFlare checkpoint must contain a mapping, got {type(checkpoint).__name__}.")
    return checkpoint


def load_nvflare_checkpoint_metadata(path: str | os.PathLike[str]) -> dict:
    """Load JSON-safe metadata from an NVFlare checkpoint."""

    checkpoint = _load_nvflare_checkpoint_data(path)
    metadata = checkpoint.get("meta_props")
    if not isinstance(metadata, Mapping):
        raise ValueError("NVFlare checkpoint is missing required dictionary 'meta_props' metadata.")
    return _metadata_safe(metadata)


def load_nvflare_checkpoint(
    path: str | os.PathLike[str], *, schema: ValidatedTrainableStateSchema | None = None
) -> OrderedDict[str, torch.Tensor]:
    """Load and validate trainable tensors from an NVFlare checkpoint."""

    checkpoint = _load_nvflare_checkpoint_data(path)
    if "model" not in checkpoint:
        raise KeyError("NVFlare checkpoint is missing required 'model' state.")
    if schema is None:
        tensors = _validate_exchange_mapping(checkpoint["model"], "NVFlare checkpoint model state")
    else:
        if not isinstance(schema, ValidatedTrainableStateSchema):
            raise TypeError("schema must be a ValidatedTrainableStateSchema, " f"got {type(schema).__name__}.")
        tensors = schema.validate(checkpoint["model"], context="NVFlare checkpoint model state")
    return OrderedDict((name, _cpu_clone(tensor)) for name, tensor in tensors.items())


class TrainableStateModule(nn.Module):
    """Small CPU module exposing exact Evo2 exchange keys to ``PTFileModelPersistor``.

    PyTorch does not permit dots in registered parameter or buffer names. The
    tensors are therefore registered under private slot names while ``state_dict``
    and ``load_state_dict`` preserve the original MCore parameter namespace.
    """

    def __init__(self, initial_state: Mapping[str, torch.Tensor]):
        super().__init__()
        tensors = _validate_exchange_mapping(initial_state, "Initial trainable state")
        self._external_to_slot = OrderedDict()
        for index, (name, tensor) in enumerate(tensors.items()):
            slot = f"_exchange_tensor_{index}"
            self.register_buffer(slot, _cpu_clone(tensor), persistent=True)
            self._external_to_slot[name] = slot

    def state_dict(self, destination=None, prefix: str = "", keep_vars: bool = False):
        if destination is None:
            destination = OrderedDict()
        for external_name, slot in self._external_to_slot.items():
            tensor = getattr(self, slot)
            if not keep_vars:
                tensor = tensor.detach()
            destination[f"{prefix}{external_name}"] = tensor.cpu().clone()
        return destination

    def load_state_dict(
        self, state_dict: Mapping[str, torch.Tensor], strict: bool = True, assign: bool = False
    ) -> _IncompatibleKeys:
        # This model is deliberately strict even if a generic caller supplies
        # strict=False: silently dropping an adapter tensor would corrupt FedAvg.
        reference = self.state_dict()
        validate_trainable_state(state_dict, reference)
        with torch.no_grad():
            for external_name, slot in self._external_to_slot.items():
                target = getattr(self, slot)
                target.copy_(state_dict[external_name].detach().to(device=target.device))
        return _IncompatibleKeys(missing_keys=[], unexpected_keys=[])
