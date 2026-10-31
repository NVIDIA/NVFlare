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

"""Transport-agnostic server-side FL orchestration (spec §4.2).

The NVFlare server app is a thin envelope around this module: it
constructs its transport's Communicator and hands it to
`run_fl_server`, which owns everything else — partition manifest, auto-resume
scan, initial-state bootstrap, and the shared `run_rounds` loop with periodic
server checkpoints. `write_final_artifacts` is the common end-of-run tail
(raw resume blob + backend-format eval checkpoint).
"""

from __future__ import annotations

import gc
import logging
import re
from collections.abc import Callable
from pathlib import Path

import torch

from decent_vla.runtime.builders import (
    build_backend_factory,
    build_strategy,
    partition_for,
    resolve_device,
)
from decent_vla.runtime.round_loop import (
    RoundLoopConfig,
    RoundLoopOutcome,
    run_rounds,
    save_server_checkpoint,
)

log = logging.getLogger(__name__)

_SERVER_CKPT_RE = re.compile(r"server_round_(\d+)\.pt")


def latest_server_checkpoint(ckpt_dir: Path) -> Path | None:
    """Highest-round `server_round_NNNN.pt` in `ckpt_dir`, or None.

    `server_round_final.pt` (the end-of-run blob written after the loop) is
    excluded by the digits-only pattern: it marks a *finished* run, not a
    resumable mid-run round.
    """
    best: tuple[int, Path] | None = None
    if not ckpt_dir.is_dir():
        return None
    for p in ckpt_dir.iterdir():
        m = _SERVER_CKPT_RE.fullmatch(p.name)
        if m and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), p)
    return best[1] if best else None


def run_fl_server(
    cfg,
    make_communicator: Callable[[], object],
) -> RoundLoopOutcome:
    """Partition, resume-scan, bootstrap, then run the shared round loop.

    `make_communicator` builds the transport's server-side Communicator; it is
    called after config/strategy setup so transports can capture the strategy-
    independent pieces they need at construction.
    """
    # Experiment tracking is declared in the run config (`wandb:`), so adopt it
    # here — this is the ONE process that owns the round loop under NVFlare,
    # and therefore the only one that should open a wandb run.
    from decent_vla.runtime import observability as _obs

    _obs.configure_wandb(cfg)

    device = resolve_device(cfg.device)
    factory, cls = build_backend_factory(cfg, device)
    strategy = build_strategy(cfg)
    chunks, manifest = partition_for(cfg, cls)
    out_dir = Path(cfg.fl.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.write(out_dir / "partition_manifest.json")

    # Auto-resume for chained jobs: pick up the latest periodic server
    # checkpoint in the (stable, explicit) output dir. DECENT_VLA_NO_RESUME=1
    # opts out and starts fresh. Blob shape mirrors cli/main.py's --resume.
    start_round = 0
    initial: dict[str, torch.Tensor] | None = None
    from decent_vla.runtime.client_checkpoint import resume_disabled

    if resume_disabled():
        log.info("DECENT_VLA_NO_RESUME set: skipping resume scan, starting fresh")
    else:
        resume = latest_server_checkpoint(out_dir / "checkpoints")
        if resume is not None:
            blob = torch.load(resume, map_location="cpu", weights_only=True)
            if not (isinstance(blob, dict) and "state" in blob and "round" in blob):
                raise RuntimeError(
                    f"resume expects a server_round_*.pt checkpoint with 'state'+'round'; "
                    f"got keys {list(blob)[:5]} from {resume}"
                )
            initial = blob["state"]
            start_round = int(blob["round"]) + 1
            if blob.get("strategy_state"):
                strategy.load_state_dict(blob["strategy_state"])
                log.info("restored server strategy state (%s) on resume", strategy.name)
            log.info("resuming FL run from round %d (checkpoint %s)", start_round, resume)
    if initial is None:
        _boot = factory()
        initial = _boot.get_state("shared")
        # Scope provenance (both axes, resolved names, payload bytes) — written
        # once at run start, while we have a built backend in hand.
        from decent_vla.runtime.scope_manifest import write_scope_manifest

        write_scope_manifest(out_dir, cfg.backend, cfg.fl, _boot)
        del _boot
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # cfg.fl.num_rounds is the TOTAL round count for the run, not "more rounds".
    remaining = cfg.fl.num_rounds - start_round
    if remaining <= 0:
        log.info(
            "nothing to do: resumed at round %d of %d total", start_round, cfg.fl.num_rounds
        )
        return RoundLoopOutcome(
            final_state={k: v.detach().clone().float() for k, v in initial.items()},
            rounds_completed=0,
        )

    client_ids = [f"c{i}" for i in range(cfg.data.num_clients)]

    def ckpt_cb(r, state):
        save_server_checkpoint(out_dir / "checkpoints", r, state, strategy.state_dict())

    return run_rounds(
        strategy=strategy,
        communicator=make_communicator(),
        initial_state=initial,
        client_ids=client_ids,
        loop_cfg=RoundLoopConfig(
            num_rounds=remaining,
            local_steps=cfg.fl.local_steps,
            min_clients=cfg.fl.min_clients,
            round_timeout_s=cfg.fl.round_timeout_s,
            save_every_n_rounds=cfg.fl.save_every_n_rounds,
            start_round=start_round,
            lean_host_memory=cfg.fl.lean_host_memory,
        ),
        checkpoint_cb=ckpt_cb,
    )


def write_final_artifacts(cfg, outcome: RoundLoopOutcome) -> Path:
    """End-of-run tail shared by remote runtimes; returns the eval checkpoint.

    Writes (a) a backend-format `round_{N:04d}.pt` for eval, mirroring the tail
    of cli/main.py's _run_fl_local exactly, and then (b) the raw
    `server_round_final.pt` resume blob — the `server_round_` prefix matters:
    cli/main.py's _latest_checkpoint excludes that prefix, so eval never tries
    to load this {state, round} dict as a backend checkpoint. (b) comes last
    because the nvflare entry takes a fresh server_round_final.pt as the sign
    that the server workflow finished. The run only reaches here once ALL
    cfg.fl.num_rounds rounds are done (auto-resume may have skipped some), so
    the absolute final round index is num_rounds - 1 regardless of how many ran
    in THIS process.
    """
    from decent_vla.core.types import CheckpointSpec

    out_dir = Path(cfg.fl.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_round = cfg.fl.num_rounds
    ckpt = ckpt_dir / f"round_{final_round:04d}.pt"

    def _export(device: torch.device) -> None:
        factory, _ = build_backend_factory(cfg, device)
        final_backend = factory()
        final_backend.set_state(outcome.final_state, scope="shared")
        final_backend.save_checkpoint(str(ckpt), spec=CheckpointSpec())

    try:
        _export(resolve_device(cfg.device))
    except torch.cuda.OutOfMemoryError:
        # On the nvflare simulator the server shares GPU 0 with client 0, which
        # can still hold its model when the run ends (a pi0.5 full fine-tuning
        # client keeps 70-90 GB there). Building a backend only to write a file
        # does not need the GPU: retry on the CPU.
        log.warning("final export ran out of GPU memory; exporting %s on the CPU", ckpt.name)
        gc.collect()
        torch.cuda.empty_cache()
        _export(torch.device("cpu"))

    final = ckpt_dir / "server_round_final.pt"
    tmp = final.with_suffix(".pt.tmp")
    torch.save({"state": outcome.final_state, "round": final_round - 1}, tmp)
    tmp.rename(final)
    return ckpt
