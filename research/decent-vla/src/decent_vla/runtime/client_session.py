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

"""Transport-agnostic client-side FL session (spec §4.2).

The NVFlare client script is a thin envelope around
this module: they decode a (global_state, plan) message with their own codec,
call `handle_round`, and encode the returned ClientUpdate. Everything the
round does between decode and encode — building/reusing the warm ClientState,
scheduler fast-forward for partial participation, set_state, run_client_round,
global_step bookkeeping, opt-in disk persistence — lives here exactly once,
so every runtime matches LocalRunner by construction.

The module-level _STATES dict is the warm-state store; it lives as long as
the client process. Config travels via an explicit path (or the
DECENT_VLA_CONFIG env var upstream); clients compute their own data shard via
`partition_for` — deterministic, so it matches the server's.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from pathlib import Path

import torch

from decent_vla.runtime import wire
from decent_vla.runtime.builders import (
    build_backend_factory,
    build_strategy,
    partition_for,
    resolve_device,
)
from decent_vla.runtime.client_loop import run_client_round
from decent_vla.runtime.client_state import (
    ClientState,
    build_client_state,
    fast_forward_scheduler,
    make_minimal_ctx,
)

_STATES: dict[str, ClientState] = {}
# (cfg, factory, strategy, chunks) per config path. Built ONCE per process:
# re-loading every message would re-fetch dataset metadata each round and
# hand each round a FRESH strategy instance — diverging from LocalRunner,
# which uses one strategy object for all rounds.
_CACHE: dict[str, tuple] = {}


def load_session(config_path: str, configure: Callable | None = None) -> tuple:
    """(cfg, factory, strategy, chunks) for `config_path`, built once per process.

    `configure(cfg)` runs only on a cache miss — the hook where a runtime sets
    its per-process codec state (e.g. wire dtype) from the loaded config.
    """
    cached = _CACHE.get(config_path)
    if cached is None:
        from decent_vla.configs import load_run_config

        cfg = load_run_config(config_path)
        if configure is not None:
            configure(cfg)
        device = resolve_device(cfg.device)
        factory, cls = build_backend_factory(cfg, device)
        strategy = build_strategy(cfg)
        chunks, _ = partition_for(cfg, cls)
        cached = (cfg, factory, strategy, chunks)
        _CACHE[config_path] = cached
    return cached


def handle_round(
    cid: str,
    config_path: str,
    global_state: dict[str, torch.Tensor],
    plan,
    configure: Callable | None = None,
):
    """Run one client round and return the ClientUpdate (transport-free).

    A round that fails part-way may already have stepped the optimizer,
    scheduler, data iterator and personal weights. Training the server's retry
    from there would spend the round's budget twice, so a failed round drops
    the client's state: the next request rebuilds it as a client joining at
    that round, from its newest saved state before the round under
    fl.persist_client_state, otherwise with fresh optimizer state and the
    schedule moved forward to the round.
    """
    try:
        return _handle_round(cid, config_path, global_state, plan, configure)
    except Exception:
        _STATES.pop(cid, None)
        raise


def _handle_round(cid, config_path, global_state, plan, configure):
    cfg, factory, strategy, chunks = load_session(config_path, configure)
    # cids are always f"c{i}" (server and runtimes agree on this naming).
    idx = int(cid[1:])

    # Opt-in per-client persistence (fl.persist_client_state): restore on
    # (re)build — a remote runtime may run the client as a fresh process per
    # round, so this is also where a resumed run picks its client state back
    # up. DECENT_VLA_NO_RESUME skips the restore (mirrors the server session).
    _client_ckpt_dir = (
        Path(cfg.fl.output_dir) / "client_states" if cfg.fl.persist_client_state else None
    )
    from decent_vla.runtime.client_checkpoint import resume_disabled

    _load_dir = None if resume_disabled() else _client_ckpt_dir

    st = _STATES.get(cid)
    if st is None:
        # A state dropped after a failed round can still hold its tensors
        # through reference cycles; free them before building another.
        gc.collect()
        st = build_client_state(
            client_id=cid,
            backend_factory=factory,
            data_cfg=cfg.data,
            optim_cfg=cfg.backend.optim,
            partition_indices=chunks[idx],
            strategy=strategy,
            start_round=plan.round_num,
            local_steps=cfg.fl.local_steps,
            client_ckpt_dir=_load_dir,
        )
        _STATES[cid] = st

    # Partial participation: a client skipped by client-selection in earlier
    # rounds arrives with a stale scheduler; fast-forward to the cumulative
    # step (spec §6) — mirrors LocalRunner exactly.
    expected_step = plan.round_num * cfg.fl.local_steps
    if st.global_step < expected_step:
        fast_forward_scheduler(st.scheduler, expected_step - st.global_step)
        st.global_step = expected_step

    st.backend.set_state(global_state, scope="shared")
    update = run_client_round(
        backend=st.backend,
        strategy=strategy,
        plan=plan,
        dataloader=st.dataloader,
        client_id=cid,
        round_num=plan.round_num,
        global_state=global_state,
        optimizer=st.optimizer,
        scheduler=st.scheduler,
        persistent_state=st.persistent,
        grad_clip_norm=cfg.backend.optim.grad_clip_norm,
        # Narrow the delta on-device to the dtype it will cross the wire in, so
        # the host never holds a full-precision copy (see ClientContext).
        delta_dtype=wire.resolve(cfg.fl.wire_dtype)[1],
    )
    # Advance by the ACTUAL steps executed (an exhausted dataloader ends the
    # round early), not the planned count.
    st.global_step += update.extra.get("steps_executed", plan.local_steps)
    update.client_id = cid
    ctx = make_minimal_ctx(cid, plan.round_num, global_state, st.persistent)
    st.persistent = strategy.update_client_state(cid, ctx, update)
    # Opt-in: persist full client state at the server's cadence (and final
    # round) so a killed run resumes warm.
    if cfg.fl.persist_client_state and _client_ckpt_dir is not None:
        n = cfg.fl.save_every_n_rounds
        is_final = (plan.round_num + 1) >= cfg.fl.num_rounds
        if (n and (plan.round_num + 1) % n == 0) or is_final:
            from decent_vla.runtime.client_checkpoint import save_client_state

            save_client_state(
                _client_ckpt_dir, st, round_idx=plan.round_num,
                keep=cfg.fl.client_state_keep,
            )
    return update
