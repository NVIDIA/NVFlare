# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch
import torch.nn as nn

from nvflare.fuel.utils.log_utils import get_module_logger
from nvflare.security.logging import secure_format_exception


@dataclass(frozen=True)
class ParamShapeMismatch:
    key: str
    expected_shape: tuple
    received_shape: tuple


@dataclass(frozen=True)
class ModelParamMatchReport:
    """Summary of how an incoming parameter payload matches a local keyspace.

    The report is intentionally descriptive rather than prescriptive: callers can
    decide whether to warn, fail fast, or filter to `matched_keys`.

    Attributes:
        external_key_count: Number of keys present in the incoming payload.
        local_key_count: Number of keys present in the local model/checkpoint.
        external_key_sample: Up to the first five sorted keys from the incoming
            payload. This is only for diagnostics and does not imply ordering.
        local_key_sample: Up to the first five sorted keys from the local
            model/checkpoint keyspace. This is only for diagnostics.
        matched_keys: Incoming keys that exist locally and have compatible
            shapes. Partial overlap is allowed.
        unexpected_keys: Incoming keys that do not exist locally.
        shape_mismatches: Keys that exist in both places but whose shapes differ.
        prefix_hint: Optional hint for common wrapper drift such as ``model.``
            prefixes on all incoming keys.
    """

    external_key_count: int
    local_key_count: int
    external_key_sample: tuple[str, ...]
    local_key_sample: tuple[str, ...]
    matched_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[ParamShapeMismatch, ...]
    prefix_hint: Optional[str] = None

    def format_context(self) -> str:
        return (
            f" Incoming keys: {self.external_key_count} sample={list(self.external_key_sample)}."
            f" Local keys: {self.local_key_count} sample={list(self.local_key_sample)}."
        )

    def format_zero_match_error(self) -> str:
        msg = (
            f"None of the {self.external_key_count} incoming model parameter(s) matched the local model's "
            f"{self.local_key_count} parameter(s)."
        )
        msg += self.format_context()
        if self.unexpected_keys:
            msg += f" Unexpected keys sample: {_format_key_sample(self.unexpected_keys)}."
        if self.prefix_hint:
            msg += f" {self.prefix_hint}"
        return msg

    def format_shape_mismatch_error(self) -> str:
        sample = ", ".join(
            f"{m.key}: expected {m.expected_shape}, got {m.received_shape}" for m in self.shape_mismatches[:5]
        )
        more = "" if len(self.shape_mismatches) <= 5 else f" (+{len(self.shape_mismatches) - 5} more)"
        return f"Incoming model parameter shape mismatch: {sample}{more}.{self.format_context()}"

    def format_unexpected_keys_warning(self) -> str:
        msg = f"Ignoring {len(self.unexpected_keys)} unexpected model parameter(s): {_format_key_sample(self.unexpected_keys)}."
        msg += self.format_context()
        if self.prefix_hint:
            msg += f" {self.prefix_hint}"
        return msg

    def format_unexpected_keys_error(self) -> str:
        msg = f"Rejecting {len(self.unexpected_keys)} unexpected model parameter(s): {_format_key_sample(self.unexpected_keys)}."
        msg += self.format_context()
        if self.prefix_hint:
            msg += f" {self.prefix_hint}"
        return msg


def _format_key_sample(keys: tuple[str, ...], max_keys: int = 5) -> str:
    sample = list(keys[:max_keys])
    if len(keys) > max_keys:
        sample.append(f"... (+{len(keys) - max_keys} more)")
    return str(sample)


def _get_value_shape(value) -> Optional[tuple]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(shape)

    try:
        return tuple(torch.as_tensor(value).shape)
    except Exception:
        return None


def _detect_prefix_hint(local_keys: set[str], external_keys: tuple[str, ...]) -> Optional[str]:
    if not local_keys or not external_keys:
        return None

    prefixes = []
    for key in external_keys:
        if "." in key:
            prefix = key.split(".", 1)[0] + "."
            if prefix not in prefixes:
                prefixes.append(prefix)

    for prefix in prefixes:
        if not all(key.startswith(prefix) for key in external_keys):
            continue

        stripped_keys = {key[len(prefix) :] for key in external_keys}
        matches = stripped_keys.intersection(local_keys)
        if matches:
            return (
                f"Hint: stripping common prefix '{prefix}' would match "
                f"{len(matches)}/{len(external_keys)} incoming key(s)."
            )
    return None


def inspect_model_params(
    local_var_dict: Mapping[str, object], model_params: Optional[Mapping[str, object]]
) -> ModelParamMatchReport:
    """Compare incoming model parameters against a local model/checkpoint keyspace.

    This helper does not mutate either input. It only classifies the incoming
    keys into matches, unexpected keys, and shape mismatches, and captures small
    key samples to make diagnostics readable.

    Partial payloads are valid: callers can choose to accept a subset of keys as
    long as there is at least one compatible match. Empty or missing payloads are
    also valid and return an all-empty report.

    Args:
        local_var_dict: Local model or checkpoint mapping keyed by parameter name.
        model_params: Incoming parameter mapping to validate.

    Returns:
        A ``ModelParamMatchReport`` describing how the incoming payload relates
        to the local keyspace.
    """

    model_params = model_params or {}
    local_keys = set(local_var_dict)
    matched_keys = []
    unexpected_keys = []
    shape_mismatches = []

    for key, value in model_params.items():
        if key not in local_var_dict:
            unexpected_keys.append(key)
            continue

        expected_shape = _get_value_shape(local_var_dict[key])
        received_shape = _get_value_shape(value)
        if expected_shape is not None and received_shape is not None and expected_shape != received_shape:
            shape_mismatches.append(
                ParamShapeMismatch(key=key, expected_shape=expected_shape, received_shape=received_shape)
            )
            continue

        matched_keys.append(key)

    external_keys = tuple(sorted(model_params.keys()))
    local_keys_sorted = tuple(sorted(local_var_dict.keys()))
    return ModelParamMatchReport(
        external_key_count=len(model_params),
        local_key_count=len(local_var_dict),
        external_key_sample=external_keys[:5],
        local_key_sample=local_keys_sorted[:5],
        matched_keys=tuple(sorted(matched_keys)),
        unexpected_keys=tuple(sorted(unexpected_keys)),
        shape_mismatches=tuple(shape_mismatches),
        prefix_hint=_detect_prefix_hint(local_keys, external_keys) if unexpected_keys else None,
    )


def feed_vars(model: nn.Module, model_params):
    """Feed variable values from model_params to pytorch state_dict.

    Args:
        model (nn.Module): the local pytorch model
        model_params: incoming parameter mapping keyed by state-dict name.

    Returns:
        a list of params and a dictionary of vars to params

    Raises:
        RuntimeError: if a matching key has a shape mismatch, or if a non-empty
            incoming payload has zero compatible matches with the local model.

    Notes:
        Empty payloads are treated as a no-op. Partial payloads are accepted as
        long as at least one key matches; unknown keys are ignored with a warning
        instead of being applied to the local state dict. This is for loading a
        received model into a local PyTorch module. Server-side validation of
        learned client updates is handled by ``PTModelPersistenceFormatManager``
        and rejects keys outside the server checkpoint schema.
    """
    _logger = get_module_logger(__name__, "AssignVariables")
    _logger.debug("AssignVariables...")

    to_assign = []
    model_params = model_params or {}
    n_ext = len(model_params)
    _logger.debug(f"n_ext {n_ext}")

    local_var_dict = model.state_dict()
    report = inspect_model_params(local_var_dict, model_params)

    if report.shape_mismatches:
        raise RuntimeError(report.format_shape_mismatch_error())

    if n_ext > 0 and not report.matched_keys:
        raise RuntimeError(report.format_zero_match_error())

    if report.unexpected_keys:
        _logger.warning(report.format_unexpected_keys_warning())

    matched_key_set = set(report.matched_keys)
    for var_name in local_var_dict:
        try:
            if var_name in matched_key_set:
                nd = model_params[var_name]
                to_assign.append(nd)
                local_var_dict[var_name] = torch.as_tensor(
                    nd
                )  # update local state dict TODO: enable setting of datatype
        except Exception as e:
            _logger.error(f"feed_vars Exception: {secure_format_exception(e)}")
            raise RuntimeError(secure_format_exception(e))

    _logger.debug("Updated local variables to be assigned.")

    n_assign = len(to_assign)
    _logger.info(f"Vars {n_ext} of {n_assign} assigned.")
    return to_assign, local_var_dict
