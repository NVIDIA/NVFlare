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

"""LoRA helper shared across adapters.

This module exposes `apply_peft` — a default LoRA-application helper that
adapters can call from their `Backend.from_config`. Each adapter passes
its own `attr_priority` (the order of backbone-submodule attr names to
look for); the helper does the actual PEFT wrapping if `peft` is
installed.

Per the spec (§5.2), LoRA is owned by the Backend, not the framework.
`apply_peft` is a default; adapters are free to implement custom LoRA
logic instead.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoRAConfig:
    """Declarative LoRA spec. Maps directly to peft.LoraConfig fields, plus
    `attach_to`, which is ours: *which* submodule the adapters go on."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] | str = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    bias: str = "none"
    task_type: str | None = None  # most VLAs leave this unset; PEFT auto-detects
    attach_to: str | None = None  # dotted module path from the model root, e.g.
                                   # "action_head" or "backbone.model.language_model".
                                   # None = the adapter's own default attachment
                                   # point. `target_modules` says which leaves
                                   # inside get adapters; this says which part of
                                   # the network is up for adaptation at all —
                                   # the difference between adapting the VLM and
                                   # adapting the action head.


def resolve_peft_target(inner_model, attr_priority: Sequence[str]):
    """Return (attr_name_or_None, submodule_to_wrap).

    Falls through to wrapping the entire inner_model if no priority attr
    matches. Each adapter passes its own priority list.
    """
    for attr in attr_priority:
        if hasattr(inner_model, attr):
            return attr, getattr(inner_model, attr)
    return None, inner_model


def apply_peft(
    model,
    lora_cfg: LoRAConfig,
    attr_priority: Sequence[str] = (),
) -> Any:
    """Wrap `model` (or a submodule chosen by `attr_priority`) with PEFT LoRA.

    Raises `ImportError` with a clear message if `peft` is not installed
    — adapters that want LoRA should require the `[peft]` extra.
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "LoRA requested but `peft` is not installed. "
            "Install with: pip install -e '.[peft]'"
        ) from exc

    attr_name, target = resolve_peft_target(model, attr_priority)
    peft_cfg = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=lora_cfg.target_modules,
        bias=lora_cfg.bias,
        task_type=lora_cfg.task_type,
    )
    wrapped = get_peft_model(target, peft_cfg)
    if attr_name is not None:
        setattr(model, attr_name, wrapped)
        return model
    return wrapped
