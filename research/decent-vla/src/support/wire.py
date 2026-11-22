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

"""Shared wire-precision conversion for FLARE model exchange and payload sizes.

`fl.wire_dtype` accepts fp16/fp32 for both directions or a mapping such as
{broadcast: fp32, delta: fp16}. The server master and aggregation stay fp32.
NumPy cannot carry bfloat16, so those tensors are always converted to fp16.
"""

from __future__ import annotations

import torch

WIRE_DTYPES: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}
_BYTES = {torch.float16: 2, torch.float32: 4}

DEFAULT_WIRE_DTYPE = "fp16"
DIRECTIONS = ("broadcast", "delta")


def _one(name: object) -> torch.dtype:
    key = str(name).lower()
    if key not in WIRE_DTYPES:
        raise ValueError(f"fl.wire_dtype={name!r} invalid; use one of {sorted(set(WIRE_DTYPES))}")
    return WIRE_DTYPES[key]


def resolve(spec: object) -> tuple[torch.dtype, torch.dtype]:
    """``(broadcast_dtype, delta_dtype)`` for a scalar or per-direction spec."""
    if isinstance(spec, dict):
        unknown = set(spec) - set(DIRECTIONS)
        if unknown:
            raise ValueError(
                f"fl.wire_dtype has unknown direction(s) {sorted(unknown)}; " f"expected a subset of {list(DIRECTIONS)}"
            )
        return (
            _one(spec.get("broadcast", DEFAULT_WIRE_DTYPE)),
            _one(spec.get("delta", DEFAULT_WIRE_DTYPE)),
        )
    d = _one(spec)
    return d, d


def nbytes(spec: object) -> tuple[int, int]:
    """Bytes per element per direction — for the scope manifest's payload sizes."""
    b, d = resolve(spec)
    return _BYTES[b], _BYTES[d]


def cast(v: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Detach and cast floating tensors; convert bfloat16 to NumPy-compatible fp16."""
    v = v.detach()
    if not v.is_floating_point():
        return v
    if v.dtype == torch.bfloat16:
        return v.to(torch.float16)
    return v.to(dtype)
