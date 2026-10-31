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

"""Per-client checkpoint save/restore for resume (opt-in, spec 2026-06-24).

Mirrors round_loop's server-checkpoint tests but for the client side: a client
that trained (warm Adam, advanced step, changed weights) can be reconstructed
from disk so a killed run resumes warm instead of cold.
"""
from __future__ import annotations

import torch

from decent_vla.core.strategy import RoundPlan
from decent_vla.runtime.client_checkpoint import (
    latest_client_checkpoint,
    load_client_state,
    save_client_state,
)
from decent_vla.runtime.client_loop import run_client_round
from decent_vla.runtime.client_state import build_client_state
from decent_vla.strategies.fedavg import FedAvg


def _train(st, n):
    update = run_client_round(
        backend=st.backend,
        strategy=FedAvg(None),
        plan=RoundPlan(selected_clients=[st.client_id], local_steps=n),
        dataloader=st.dataloader,
        client_id=st.client_id,
        round_num=0,
        global_state=st.backend.get_state("shared"),
        optimizer=st.optimizer,
        scheduler=st.scheduler,
        persistent_state=st.persistent,
    )
    st.global_step += update.extra["steps_executed"]


def _new(cid, backend_factory, data_cfg, backend_cfg, start_round, ckpt_dir=None):
    return build_client_state(
        client_id=cid, backend_factory=backend_factory, data_cfg=data_cfg,
        optim_cfg=backend_cfg.optim, partition_indices=list(range(16)),
        strategy=FedAvg(None), start_round=start_round, local_steps=5,
        client_ckpt_dir=ckpt_dir,
    )


def test_save_keep_prunes_older_rounds_per_client(
    tmp_path, backend_factory, data_cfg, backend_cfg
):
    """keep=N retains only the N highest-round files PER CLIENT — at 3B scale
    a client blob is ~35GB, so keep-all across 50 rounds is not an option."""
    st0 = _new("c0", backend_factory, data_cfg, backend_cfg, start_round=0)
    st1 = _new("c1", backend_factory, data_cfg, backend_cfg, start_round=0)
    save_client_state(tmp_path, st0, round_idx=0, keep=2)
    save_client_state(tmp_path, st0, round_idx=5, keep=2)
    save_client_state(tmp_path, st1, round_idx=5, keep=2)
    p = save_client_state(tmp_path, st0, round_idx=10, keep=2)
    c0_files = sorted(f.name for f in tmp_path.glob("c0_round_*.pt"))
    assert c0_files == ["c0_round_0005.pt", "c0_round_0010.pt"]
    # Other clients' files are untouched.
    assert [f.name for f in tmp_path.glob("c1_round_*.pt")] == ["c1_round_0005.pt"]
    assert latest_client_checkpoint(tmp_path, "c0") == p


def test_save_keep_none_keeps_everything(
    tmp_path, backend_factory, data_cfg, backend_cfg
):
    st = _new("c0", backend_factory, data_cfg, backend_cfg, start_round=0)
    save_client_state(tmp_path, st, round_idx=0)
    save_client_state(tmp_path, st, round_idx=5)
    assert len(list(tmp_path.glob("c0_round_*.pt"))) == 2


def test_restore_falls_back_past_future_round_blobs(
    tmp_path, backend_factory, data_cfg, backend_cfg
):
    """A crash between client-save(r) and server-save(r) leaves the newest blob
    at round >= start_round. Restore must fall back to the newest ELIGIBLE blob
    (round < start_round) instead of silently cold-resetting — for
    personalization arms a cold reset wipes every personal parameter."""
    st = _new("c0", backend_factory, data_cfg, backend_cfg, start_round=0)
    _train(st, 3)
    with torch.no_grad():
        st.backend.model.fc1.weight.add_(7.0)
    want = st.backend.get_state("all")["fc1.weight"].clone()
    save_client_state(tmp_path, st, round_idx=3)
    with torch.no_grad():
        st.backend.model.fc1.weight.add_(1.0)
    save_client_state(tmp_path, st, round_idx=5)  # "future" vs start_round=4

    st2 = _new("c0", backend_factory, data_cfg, backend_cfg, start_round=4, ckpt_dir=tmp_path)
    # The round-3 blob is restored (weights prove it) — not skipped because the
    # NEWEST blob happened to be from a future round.
    assert torch.equal(st2.backend.get_state("all")["fc1.weight"], want)


def test_save_writes_per_client_file_and_latest_finds_highest_round(
    tmp_path, backend_factory, data_cfg, backend_cfg
):
    st = _new("c0", backend_factory, data_cfg, backend_cfg, start_round=0)
    _train(st, 3)
    p0 = save_client_state(tmp_path, st, round_idx=0)
    p1 = save_client_state(tmp_path, st, round_idx=5)
    assert p0.exists() and p1.exists()
    assert latest_client_checkpoint(tmp_path, "c0") == p1   # highest round wins
    assert latest_client_checkpoint(tmp_path, "cX") is None  # unknown client
    blob = load_client_state(p1)
    assert blob["client_id"] == "c0" and blob["round"] == 5


def test_build_client_state_restores_optimizer_step_and_weights(
    tmp_path, backend_factory, data_cfg, backend_cfg
):
    st = _new("c0", backend_factory, data_cfg, backend_cfg, start_round=0)
    _train(st, 3)                                   # warm Adam + changed weights
    st.global_step = 3
    saved_weights = st.backend.get_state("all")
    saved_exp_avg = next(
        s["exp_avg"].clone() for s in st.optimizer.state.values() if "exp_avg" in s
    )
    save_client_state(tmp_path, st, round_idx=0)

    # fresh client; cold start_step would be 2*5=10 -> restore must override to 3
    st2 = _new("c0", backend_factory, data_cfg, backend_cfg, start_round=2, ckpt_dir=tmp_path)
    assert st2.global_step == 3                                       # restored, not cold 10
    new_weights = st2.backend.get_state("all")
    for k in saved_weights:
        assert torch.allclose(new_weights[k], saved_weights[k])       # weights restored
    new_exp_avg = next(
        s["exp_avg"] for s in st2.optimizer.state.values() if "exp_avg" in s
    )
    assert torch.allclose(new_exp_avg, saved_exp_avg)                 # Adam moments restored


def test_no_ckpt_dir_is_cold_start(backend_factory, data_cfg, backend_cfg):
    """Default (flag OFF) = today's behavior: cold build, no restore."""
    st = _new("c0", backend_factory, data_cfg, backend_cfg, start_round=2)  # ckpt_dir=None
    assert st.global_step == 10                                        # cold: start_round*local_steps


def test_restore_skipped_when_checkpoint_at_or_after_start_round(
    tmp_path, backend_factory, data_cfg, backend_cfg
):
    """Only restore a checkpoint from a PRIOR round (catching up); a checkpoint
    at/after start_round is stale-forward and must be ignored (cold build)."""
    st = _new("c0", backend_factory, data_cfg, backend_cfg, start_round=0)
    _train(st, 3)
    st.global_step = 3
    save_client_state(tmp_path, st, round_idx=5)          # round 5 saved
    st2 = _new("c0", backend_factory, data_cfg, backend_cfg, start_round=1, ckpt_dir=tmp_path)
    assert st2.global_step == 5                            # cold (round5 >= start1 -> ignored)
