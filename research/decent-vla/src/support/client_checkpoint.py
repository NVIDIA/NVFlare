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

"""Optional client checkpoints for weights, optimizer, scheduler and global step.

Gated by ``fl.persist_client_state``; cadence reuses ``fl.save_every_n_rounds``.
Files live at ``<output_dir>/client_states/<cid>_round_NNNN.pt``. RNG and
data-iterator progress are not included, so this is not complete swap state.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import torch

_CLIENT_CKPT_GLOB = "{cid}_round_*.pt"


def resume_disabled() -> bool:
    """Single source of truth for DECENT_VLA_NO_RESUME on BOTH sides.

    '' and '0' mean resume normally; anything else disables restore. The
    server and client MUST agree — a split reading (server resumes, client
    skips restore) silently resets every personal parameter.
    """
    return os.environ.get("DECENT_VLA_NO_RESUME", "") not in ("", "0")


def _round_re(cid: str) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(cid)}_round_(\d+)\.pt$")


def save_client_state(ckpt_dir: str | Path, state, round_idx: int, keep: int | None = None) -> Path:
    """Atomically save client weights, optimizer/scheduler state and global step.

    ``keep`` retains only the highest-round files for this client after a
    successful write; ``None`` keeps all checkpoints.
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sched = getattr(state, "scheduler", None)
    blob = {
        "client_id": state.client_id,
        "round": round_idx,
        "global_step": state.global_step,
        "weights": state.backend.get_state("all"),
        "optimizer": state.optimizer.state_dict(),
        "scheduler": sched.state_dict() if (sched is not None and hasattr(sched, "state_dict")) else None,
    }
    path = ckpt_dir / f"{state.client_id}_round_{round_idx:04d}.pt"
    tmp = path.with_name(path.name + ".tmp")
    torch.save(blob, tmp)
    tmp.rename(path)
    if keep is not None and keep > 0:
        for _, old in client_checkpoints(ckpt_dir, state.client_id)[:-keep]:
            old.unlink(missing_ok=True)
    return path


def client_checkpoints(ckpt_dir: str | Path, cid: str) -> list[tuple[int, Path]]:
    """All ``<cid>_round_NNNN.pt`` files, sorted by round ascending."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return []
    rx = _round_re(cid)
    out: list[tuple[int, Path]] = []
    for p in ckpt_dir.glob(_CLIENT_CKPT_GLOB.format(cid=cid)):
        m = rx.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def load_client_state(path: str | Path) -> dict:
    """Load a client checkpoint blob (contains non-tensor state -> not weights_only)."""
    return torch.load(path, map_location="cpu", weights_only=False)
