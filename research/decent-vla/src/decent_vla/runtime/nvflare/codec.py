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

"""decent-vla types <-> nvflare FLModel. The ONLY place tensors meet nvflare.

Lazy imports: nvflare is an optional extra; importing this module without
nvflare is fine, calling `round_flmodel`/`update_to_flmodel` is not (the
decoders are import-free). Params cross the wire as numpy arrays
(ExchangeFormat.NUMPY / FrameworkType.NUMPY on the ScriptRunner, so the
Client API applies no conversion or diffing of its own); the plan and the
update's non-tensor payload ride in FLModel.meta as JSON-safe dicts.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch

from decent_vla.core.strategy import ClientUpdate, RoundPlan
from decent_vla.runtime import wire

PLAN_KEY = "decent_vla_plan"
META_KEY = "decent_vla_meta"

# Precision for floating tensors crossing the nvflare boundary — BOTH ways.
# Resolved from ``fl.wire_dtype`` via runtime/wire.py, the single table shared
# with the scope manifest, so one config always yields the same bytes.
# Set once per process. NOTE numpy has no bfloat16, so bf16 is always downcast
# to fp16 regardless of this setting (see wire.cast).
_BROADCAST_DTYPE = wire.WIRE_DTYPES[wire.DEFAULT_WIRE_DTYPE]
_DELTA_DTYPE = wire.WIRE_DTYPES[wire.DEFAULT_WIRE_DTYPE]


def set_wire_dtype(spec) -> None:
    """Configure BOTH on-the-wire dtypes (idempotent per process).

    Accepts a scalar ("fp16"/"fp32") applied to both directions, or a mapping
    ``{broadcast: ..., delta: ...}`` for an asymmetric wire.
    """
    global _BROADCAST_DTYPE, _DELTA_DTYPE
    _BROADCAST_DTYPE, _DELTA_DTYPE = wire.resolve(spec)


def _delta_to_wire(v: torch.Tensor) -> torch.Tensor:
    """Cast a client-delta tensor to the configured DELTA dtype (floats only)."""
    return wire.cast(v, _DELTA_DTYPE)


def state_to_np(
    state: dict[str, torch.Tensor],
    dtype: torch.dtype | None = None,
    consume: bool = False,
) -> dict[str, np.ndarray]:
    """Tensors -> numpy at `dtype` (default: the configured BROADCAST dtype).

    `consume=True` drains `state` key by key, so each source tensor is released
    as its array is produced instead of holding the whole source dict and the
    whole output at once — at pi0.5's 3.62B params that turns a ~20 GB peak into
    ~13 GB. Safe even when the cast is a no-op (fp32->fp32): numpy then shares
    the tensor's storage and keeps it alive through the array's `.base`.
    """
    dt = _BROADCAST_DTYPE if dtype is None else dtype
    if not consume:
        return {k: wire.cast(v, dt).cpu().numpy() for k, v in state.items()}
    out: dict[str, np.ndarray] = {}
    for k in list(state):
        v = state.pop(k)
        out[k] = wire.cast(v, dt).cpu().numpy()
        del v
    return out


def np_to_state(params: dict[str, Any], consume: bool = False) -> dict[str, torch.Tensor]:
    """numpy params -> tensors.

    NOTE `np.ascontiguousarray` is a NO-OP for already-contiguous arrays, which
    is the normal case here — so the tensor SHARES the transport's buffer rather
    than copying it. (The original comment on this function claimed a defensive
    copy that does not actually happen.) Sharing is fine and cheaper; what
    matters is that the buffer ends up with a single owner.

    `consume=True` pops each entry out of `params` as it converts, so the
    transport envelope stops referencing the payload and the memory is genuinely
    reclaimable once the caller drops the tensors.
    """
    if not consume:
        return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in params.items()}
    out: dict[str, torch.Tensor] = {}
    for k in list(params):
        out[k] = torch.from_numpy(np.ascontiguousarray(params.pop(k)))
    return out


def plan_to_meta(plan: RoundPlan) -> dict[str, Any]:
    return {
        "round_num": int(plan.round_num),
        "local_steps": int(plan.local_steps),
        "selected_clients": json.dumps(list(plan.selected_clients)),
        "broadcast": json.dumps(plan.broadcast, default=str),
    }


def meta_to_plan(meta: dict[str, Any]) -> RoundPlan:
    d = meta[PLAN_KEY]
    return RoundPlan(
        selected_clients=json.loads(d["selected_clients"]),
        local_steps=int(d["local_steps"]),
        broadcast=json.loads(d["broadcast"]),
        round_num=int(d["round_num"]),
    )


def round_flmodel(state: dict[str, torch.Tensor], plan: RoundPlan):
    """Server -> client FLModel: master params at the BROADCAST wire dtype.

    NEVER consumes `state` — the server keeps its fp32 master and re-broadcasts
    it to every client (and again on the round-loop retry path).
    """
    from nvflare.app_common.abstract.fl_model import FLModel, ParamsType

    return FLModel(
        params=state_to_np(state),
        params_type=ParamsType.FULL,
        current_round=int(plan.round_num),
        meta={PLAN_KEY: plan_to_meta(plan)},
    )


def update_to_flmodel(update: ClientUpdate, consume: bool = False):
    """Client -> server FLModel: delta params at the DELTA wire dtype + meta.

    ClientUpdate.metrics may mix numbers and strings — numbers travel in
    FLModel.metrics, strings in meta.

    `consume=True` drains `update.delta` as the arrays are built, so the fp32
    delta and its fp16 copy never coexist. Only pass it when you own the update
    and will not read `.delta` again (the client loop does).
    """
    from nvflare.app_common.abstract.fl_model import FLModel, ParamsType

    nums: dict[str, Any] = {}
    strs: dict[str, Any] = {"num_samples": str(update.num_samples), "client_id": update.client_id}
    for k, v in update.metrics.items():
        (nums if isinstance(v, (int, float)) else strs)[k] = v
    return FLModel(
        params=state_to_np(update.delta, _DELTA_DTYPE, consume=consume),
        params_type=ParamsType.FULL,
        metrics=nums,
        meta={META_KEY: strs},
    )


def flmodel_to_update(model, consume: bool = False) -> ClientUpdate:
    """Client reply FLModel -> ClientUpdate. `consume=True` drains
    `model.params` as it decodes (see `np_to_state`) — safe for a per-round
    reply that nobody reads again."""
    meta = dict(model.meta.get(META_KEY, {}))
    num_samples = int(meta.pop("num_samples"))
    client_id = str(meta.pop("client_id", ""))
    metrics: dict[str, Any] = dict(model.metrics or {})
    metrics.update(meta)
    return ClientUpdate(
        delta=np_to_state(model.params, consume=consume),
        num_samples=num_samples,
        client_id=client_id,
        metrics=metrics,
    )


def flmodel_to_round(model) -> tuple[dict[str, torch.Tensor], RoundPlan]:
    """Client-side decode of the server's per-round FLModel."""
    return np_to_state(model.params), meta_to_plan(model.meta)
