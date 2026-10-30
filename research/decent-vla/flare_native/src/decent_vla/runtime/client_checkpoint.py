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

"""Per-client checkpoint persistence for warm resume (opt-in).

The server side already resumes from ``server_round_*.pt`` (see
``runtime.nvflare.persistor.VLAPersistor``). This is the client-side counterpart:
each client's weights + optimizer (Adam moments) + scheduler step + legacy
``persistent`` checkpoint field are written to disk so a killed FL run resumes *warm*
instead of cold-restarting every client's optimizer (audit #3/#9).

Gated by ``fl.persist_client_state`` (default off); cadence reuses
``fl.save_every_n_rounds``. Files live at
``<output_dir>/client_states/<cid>_round_NNNN.pt``.
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
    """Atomically write a client's full state (tmp file + rename).

    ``state`` is a ``ClientState`` (duck-typed: ``client_id``, ``backend``,
    ``optimizer``, ``scheduler``, ``persistent``, ``global_step``). Stores the
    full trainable weights plus the optimizer/scheduler/persistent state so the
    client can be reconstructed exactly.

    ``keep``: retain only the ``keep`` highest-round files for THIS client
    after a successful write (``fl.client_state_keep``). At 3B scale a blob is
    ~35 GB per client, so keep-all across 50 rounds is not viable. ``None``
    keeps everything (historical behavior).
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
        "persistent": state.persistent,
    }
    path = ckpt_dir / f"{state.client_id}_round_{round_idx:04d}.pt"
    tmp = path.with_name(path.name + ".tmp")
    torch.save(blob, tmp)
    tmp.rename(path)
    if keep is not None and keep > 0:
        rx = _round_re(state.client_id)
        rounds: list[tuple[int, Path]] = []
        for p in ckpt_dir.glob(_CLIENT_CKPT_GLOB.format(cid=state.client_id)):
            m = rx.match(p.name)
            if m:
                rounds.append((int(m.group(1)), p))
        for _, old in sorted(rounds)[:-keep]:
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


def latest_client_checkpoint(ckpt_dir: str | Path, cid: str) -> Path | None:
    """Highest-round ``<cid>_round_NNNN.pt`` for this client, or None."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return None
    rx = _round_re(cid)
    best: Path | None = None
    best_round = -1
    for p in ckpt_dir.glob(_CLIENT_CKPT_GLOB.format(cid=cid)):
        m = rx.match(p.name)
        if m and int(m.group(1)) > best_round:
            best_round = int(m.group(1))
            best = p
    return best


def load_client_state(path: str | Path) -> dict:
    """Load a client checkpoint blob (contains non-tensor state -> not weights_only)."""
    return torch.load(path, map_location="cpu", weights_only=False)
