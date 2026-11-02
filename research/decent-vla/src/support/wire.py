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

"""Wire precision — the ONE place that decides on-the-wire dtypes.

The NVFlare execution plane and the scope manifest resolve
`fl.wire_dtype` through this module, so the planes are consistent by
construction: a config that says fp16 means the same thing on both, in both
directions. Keeping the table and the parser here (rather than duplicated in
each codec) is what makes "same config => same bytes on the wire" checkable.

`fl.wire_dtype` accepts either

  * a scalar — ``fp16`` | ``fp32`` — applied to BOTH directions, or
  * a mapping — ``{broadcast: fp32, delta: fp16}`` — per direction, for the
    numerical-equivalence tests that need a lossless downlink.

Historically the knob was applied ONLY to the client->server delta, while the
server->client broadcast went out at the fp32 master dtype. That asymmetry was
an accident of where the cast was inserted (``update_to_flmodel`` had it,
``round_flmodel`` did not) rather than a decision: it was undocumented, it made
`wire_dtype` mean half of what its name says, and at 3B scale it made the
broadcast the single largest host-RAM term in the whole system (13.5 GB per
live copy for pi0.5's 3.62B params, against 6.7 GB for the delta). Both
directions now honour the knob.

Precision note: fp16 carries a 10-bit mantissa, so a round-trip perturbs
weights by at most 2**-11 = 4.9e-4 relative. Measured on the real pi05_base
checkpoint: max relative error 4.88e-4, zero overflow (max |w| = 283 against
fp16's 65504 ceiling), 0.013% of weights below fp16's subnormal floor and thus
flushed to zero. That is ~20x finer than the bf16 autocast arithmetic already
consuming these weights, and it does not compound: the server master stays
fp32 (the native controller upcasts replies) and clients diff against the same rounded base
they trained from, so each round's rounding is bounded, not accumulated.

NOTE numpy cannot carry bfloat16, so bf16 tensors are
ALWAYS downcast to fp16 regardless of this setting — never widened to fp32.
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
    """Detach and cast a tensor for the wire; integers pass through unchanged.

    bf16 is pinned to fp16 (see the module note) even when `dtype` is fp32 —
    widening it would double the payload for zero information gain, and the
    two codecs have always promised this downcast.
    """
    v = v.detach()
    if not v.is_floating_point():
        return v
    if v.dtype == torch.bfloat16:
        return v.to(torch.float16)
    return v.to(dtype)
