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

from __future__ import annotations

from typing import Iterable

import torch


def is_lora_a_key(key: str) -> bool:
    return ".lora_A." in key


def is_lora_b_key(key: str) -> bool:
    return ".lora_B." in key


def is_lora_factor_key(key: str) -> bool:
    return is_lora_a_key(key) or is_lora_b_key(key)


def get_lora_base_key(key: str) -> str | None:
    if is_lora_a_key(key):
        return key.replace(".lora_A.", ".", 1)
    if is_lora_b_key(key):
        return key.replace(".lora_B.", ".", 1)
    return None


def get_lora_factor_pairs_and_base(keys: Iterable[str]) -> list[tuple[str, str, str]]:
    pairs = {}
    for key in keys:
        if is_lora_a_key(key):
            base_key = key.replace(".lora_A.", ".", 1)
            pairs.setdefault(base_key, {})["a"] = key
        elif is_lora_b_key(key):
            base_key = key.replace(".lora_B.", ".", 1)
            pairs.setdefault(base_key, {})["b"] = key

    missing = [base_key for base_key, value in pairs.items() if "a" not in value or "b" not in value]
    if missing:
        raise ValueError(f"Missing LoRA A/B factor pairs for keys: {missing[:3]}")

    return [(pairs[base_key]["a"], pairs[base_key]["b"], base_key) for base_key in sorted(pairs)]


def build_uniform_lora_rank_map(keys: Iterable[str], rank: int) -> dict[str, int]:
    if rank <= 0:
        raise ValueError(f"LoRA rank must be positive, got {rank}.")
    rank_map = {}
    for _a_key, _b_key, base_key in get_lora_factor_pairs_and_base(keys):
        rank_map[base_key] = rank
    return rank_map


def truncate_global_bank_for_site(
    global_state: dict[str, torch.Tensor], site_rank_map: dict[str, int]
) -> dict[str, torch.Tensor]:
    local_state = {}
    for key, value in global_state.items():
        base_key = get_lora_base_key(key)
        if base_key is None:
            local_state[key] = value
            continue

        rank = site_rank_map.get(base_key)
        if rank is None:
            raise KeyError(f"Missing local LoRA rank for layer {base_key}.")
        if rank <= 0:
            raise ValueError(f"Local LoRA rank must be positive for layer {base_key}, got {rank}.")

        if is_lora_a_key(key):
            local_state[key] = value[:rank, :].contiguous()
        else:
            local_state[key] = value[:, :rank].contiguous()
    return local_state
