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

"""THE server-side FL round loop (spec §4) — written once against
Communicator; LocalRunner and the NVFlare controller both call it.

fp32 master copies (spec §6): the initial state is upcast once;
StrategyBase.aggregate casts client deltas to the master dtype, so
aggregation is always bf16-safe regardless of client dtype.

min_clients (spec §10): a round with fewer updates than min_clients
(default: all selected) is retried once, then aborts loudly. Never silent
partial aggregation.
"""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from decent_vla.core.communicator import Communicator
from decent_vla.core.strategy import ClientUpdate, RoundHistory, RoundPlan, Strategy
from decent_vla.runtime import observability as _obs

log = logging.getLogger(__name__)

CheckpointCb = Callable[[int, dict[str, torch.Tensor]], None]


class RoundFailure(RuntimeError):
    pass


def save_server_checkpoint(
    ckpt_dir: Path,
    round_idx: int,
    state: dict[str, torch.Tensor],
    strategy_state: dict[str, Any] | None = None,
) -> Path:
    """Write `server_round_NNNN.pt` atomically (tmp file + rename), so a
    crash never leaves a torn .pt. Shared by the CLI and the NVFlare controller.

    ``strategy_state`` (FedAdam moments etc.) is persisted alongside the model
    state so a resume restores the server optimizer instead of restarting it
    cold. Absent/empty for stateless strategies (FedAvg, FedProx).
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"server_round_{round_idx + 1:04d}.pt"
    tmp = path.with_suffix(".pt.tmp")
    blob: dict[str, Any] = {"state": state, "round": round_idx}
    if strategy_state:
        blob["strategy_state"] = strategy_state
    torch.save(blob, tmp)
    tmp.rename(path)
    return path


@dataclass
class RoundLoopConfig:
    num_rounds: int
    local_steps: int
    min_clients: int | None = None          # None => all selected clients
    round_timeout_s: float | None = None
    save_every_n_rounds: int | None = None
    start_round: int = 0
    # Mutate server-owned buffers instead of duplicating the fp32 master:
    # consume `initial_state` while upcasting, and aggregate in place. Both are
    # numerically identical to the copying path; both destroy caller-owned
    # dicts that no caller reuses. Off = the historical copying behaviour.
    # Driven by `fl.lean_host_memory`; required for 3B-scale full-FT on a
    # <=128 GB host, where each spare copy of the master is ~13.5 GB.
    lean_host_memory: bool = False


@dataclass
class RoundLoopOutcome:
    final_state: dict[str, torch.Tensor]
    per_round_metrics: list[dict[str, Any]] = field(default_factory=list)
    rounds_completed: int = 0


def run_rounds(
    strategy: Strategy,
    communicator: Communicator,
    initial_state: dict[str, torch.Tensor],
    client_ids: list[str],
    loop_cfg: RoundLoopConfig,
    checkpoint_cb: CheckpointCb | None = None,
    log_every: int = 1,
) -> RoundLoopOutcome:
    # fp32 master copy, detached, on server-side memory.
    if loop_cfg.lean_host_memory:
        # Drain `initial_state` key by key as we upcast: holding the bootstrap
        # dict AND the fp32 master at once doubles peak host RAM at startup
        # (~27 GB at pi0.5's 3.62B params). Insertion order is preserved
        # so checkpoint layout is unchanged. Neither caller — server_session
        # nor local_runner — touches its `initial` dict after this call.
        global_state: dict[str, torch.Tensor] = {}
        for k in list(initial_state):
            v = initial_state.pop(k)
            global_state[k] = v.detach().to("cpu", torch.float32)
            del v
        # A backend's get_state clones on ITS device, so the drained bootstrap
        # tensors may have lived on a GPU. Freed, their blocks still sit in
        # torch's caching allocator: on nvflare that is the server process
        # keeping a full model copy on card 0, next to client 0, for the whole
        # run (14 GB at pi0.5's 3.62B params). Hand them back. Numerically inert.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        global_state = {
            k: v.detach().to("cpu", torch.float32) for k, v in initial_state.items()
        }
    history = RoundHistory(round_num=loop_cfg.start_round)
    per_round: list[dict[str, Any]] = []

    for r in range(loop_cfg.start_round, loop_cfg.start_round + loop_cfg.num_rounds):
        history.round_num = r
        plan = strategy.configure_round(r, client_ids, history)
        plan.round_num = r
        if plan.local_steps <= 0:
            plan.local_steps = loop_cfg.local_steps
        expected = list(plan.selected_clients)
        need = loop_cfg.min_clients if loop_cfg.min_clients is not None else len(expected)

        updates = _collect_with_retry(
            communicator, global_state, plan, expected, need,
            loop_cfg.round_timeout_s, round_num=r)

        # round_loop owns these fresh, per-round `updates` and discards them
        # right after; draining each client's delta tensors here is safe and
        # is where the memory win actually pays off for large models. Under
        # lean_host_memory the fp32 master is mutated in place too, so the old
        # and new master never coexist (saves another full payload).
        global_state = strategy.aggregate(
            updates, global_state, consume=True,
            inplace=loop_cfg.lean_host_memory,
        )

        losses = {(u.client_id or f"#{i}"): u.metrics["loss"]
                  for i, u in enumerate(updates) if "loss" in u.metrics}
        metrics = {
            "round": r,
            "n_updates": len(updates),
            "client_losses": losses,
            "mean_client_loss": (sum(losses.values()) / len(losses)) if losses else None,
        }
        per_round.append(metrics)
        _obs.wandb_round_log(
            os.path.splitext(os.path.basename(os.environ.get("DECENT_VLA_CONFIG", "decent-vla-fl")))[0],
            r, losses, metrics["mean_client_loss"], {"n_updates": len(updates)},
        )
        for cid, lo in losses.items():
            history.client_losses.setdefault(cid, []).append(lo)

        if checkpoint_cb and loop_cfg.save_every_n_rounds and \
                (r + 1 - loop_cfg.start_round) % loop_cfg.save_every_n_rounds == 0:
            checkpoint_cb(r, global_state)

        if (r - loop_cfg.start_round) % log_every == 0:
            ml = metrics["mean_client_loss"]
            ml_str = f"{ml:.4f}" if ml is not None else "n/a"
            log.info("round %d  updates=%d  mean_loss=%s", r + 1, len(updates), ml_str)

    return RoundLoopOutcome(final_state=global_state, per_round_metrics=per_round,
                            rounds_completed=loop_cfg.num_rounds)


def _collect_with_retry(
    communicator: Communicator,
    state: dict[str, torch.Tensor],
    plan: RoundPlan,
    expected: list[str],
    need: int,
    timeout_s: float | None,
    round_num: int,
) -> list[ClientUpdate]:
    # The retry collects from the *missing* clients only. Re-running clients
    # that already responded is not idempotent: persistent clients (spec §6)
    # carry optimizer/scheduler state across rounds, so a second run would
    # double-step them within one round and silently skew training.
    def _missing(updates: list[ClientUpdate]) -> list[str]:
        responded = {u.client_id for u in updates}
        return sorted(cid for cid in expected if cid not in responded)

    t0 = time.time()
    communicator.broadcast_state(state, plan)
    updates = communicator.collect_updates(expected, timeout_s)
    if len(updates) >= need:
        return updates
    missing = _missing(updates)
    log.warning(
        "round %d attempt 1: got %d/%d updates (need %d) after %.1fs; "
        "missing clients: %s",
        round_num, len(updates), len(expected), need, time.time() - t0, missing)

    t0 = time.time()
    communicator.broadcast_state(state, plan)
    updates = updates + communicator.collect_updates(missing, timeout_s)
    if len(updates) >= need:
        return updates
    missing = _missing(updates)
    log.warning(
        "round %d attempt 2: got %d/%d updates (need %d) after %.1fs; "
        "missing clients: %s",
        round_num, len(updates), len(expected), need, time.time() - t0, missing)
    raise RoundFailure(
        f"round {round_num}: {len(updates)}/{len(expected)} updates after retry "
        f"(min_clients={need}); missing clients: {missing}. "
        f"Aborting — never silent partial aggregation.")
