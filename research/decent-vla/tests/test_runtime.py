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

"""Runtime tests: hook ordering, local-mode FL E2E, partitioning."""
from __future__ import annotations

import pytest
import torch

from decent_vla.adapters._mock.backend import MockBackend
from decent_vla.configs.schemas import OptimConfig
from decent_vla.core.strategy import StrategyBase
from decent_vla.runtime.client_loop import run_client_round
from decent_vla.runtime.local_runner import LocalRunner
from decent_vla.runtime.partitioning import iid_partition, task_partition
from decent_vla.strategies import FedAvg


class _HookRecorder(StrategyBase):
    """Records the order in which hooks are called."""

    name = "hook-recorder"

    def __init__(self):
        self.calls: list[str] = []

    def on_round_start(self, ctx):
        self.calls.append("on_round_start")

    def modify_loss(self, loss, ctx):
        self.calls.append(f"modify_loss[{ctx.step_in_round}]")
        return loss

    def on_after_backward(self, ctx):
        self.calls.append(f"on_after_backward[{ctx.step_in_round}]")

    def on_round_end(self, ctx):
        self.calls.append("on_round_end")
        return super().on_round_end(ctx)


def test_hook_ordering(backend, data_cfg, device):
    """The canonical loop calls hooks in a fixed order, per spec §7.2."""
    from decent_vla.core.strategy import RoundPlan

    recorder = _HookRecorder()
    plan = RoundPlan(selected_clients=["c0"], local_steps=2)
    loader = backend.build_dataloader(data_cfg, partition_indices=None)
    optimizer, _ = backend.build_optimizer(OptimConfig(lr=1e-3))
    gs = backend.get_state("shared")
    run_client_round(
        backend=backend,
        strategy=recorder,
        plan=plan,
        dataloader=loader,
        client_id="c0",
        round_num=0,
        global_state=gs,
        optimizer=optimizer,
    )

    expected = [
        "on_round_start",
        "modify_loss[0]",
        "on_after_backward[0]",
        "modify_loss[1]",
        "on_after_backward[1]",
        "on_round_end",
    ]
    assert recorder.calls == expected


def test_local_runner_smoke(backend_factory, data_cfg, device):
    """Two clients × two rounds × MockBackend × FedAvg should produce a non-empty final state."""
    strategy = FedAvg()
    indices = iid_partition(MockBackend.SYNTHETIC_DATASET_SIZE, 2, seed=0)
    runner = LocalRunner(
        backend_factory=backend_factory,
        strategy=strategy,
        client_ids=["c0", "c1"],
        partition_indices=indices,
        data_cfg=data_cfg,
        num_rounds=2,
        local_steps=5,
        optim_cfg=OptimConfig(lr=1e-3),
        device=device,
    )
    outcome = runner.run()
    assert outcome.rounds_completed == 2
    assert outcome.final_state
    assert len(outcome.per_round_metrics) == 2
    # Mean loss should be finite.
    for m in outcome.per_round_metrics:
        assert m["mean_client_loss"] is not None
        assert torch.isfinite(torch.tensor(m["mean_client_loss"]))


def test_iid_partition_covers_all_indices():
    chunks = iid_partition(num_samples=100, num_clients=4, seed=0)
    flat = sorted(i for chunk in chunks for i in chunk)
    assert flat == list(range(100))


def test_iid_partition_seed_is_deterministic():
    a = iid_partition(100, 4, seed=7)
    b = iid_partition(100, 4, seed=7)
    assert a == b


def test_task_partition_does_not_split_tasks():
    task_ids = [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10
    chunks = task_partition(task_ids, num_clients=4, seed=0)
    # Each chunk should map to exactly one task id.
    for chunk in chunks:
        chunk_tasks = {task_ids[i] for i in chunk}
        assert len(chunk_tasks) == 1


def test_task_partition_rejects_too_few_tasks():
    import pytest

    with pytest.raises(ValueError):
        task_partition([0, 0, 1, 1], num_clients=4)


def test_localrunner_persists_optimizer_across_rounds(backend_factory, data_cfg, backend_cfg):
    """Cold-Adam fix: same optimizer object all rounds, moments accumulate."""
    from decent_vla.runtime.local_runner import LocalRunner
    from decent_vla.strategies.fedavg import FedAvg

    runner = LocalRunner(
        backend_factory=backend_factory, strategy=FedAvg(None),
        client_ids=["c0"], partition_indices=[list(range(32))],
        data_cfg=data_cfg, num_rounds=2, local_steps=3,
        optim_cfg=backend_cfg.optim, device=torch.device("cpu"))
    runner.run()
    st = runner.client_states["c0"]
    assert st.global_step == 6                       # 2 rounds x 3 steps cumulative
    assert len(st.optimizer.state) > 0               # Adam moments survived round 1


def test_localrunner_matches_round_loop_semantics(backend_factory, data_cfg, backend_cfg):
    from decent_vla.runtime.local_runner import LocalRunner
    from decent_vla.strategies.fedavg import FedAvg

    runner = LocalRunner(
        backend_factory=backend_factory, strategy=FedAvg(None),
        client_ids=["c0", "c1"], partition_indices=[list(range(16)), list(range(16, 32))],
        data_cfg=data_cfg, num_rounds=2, local_steps=2,
        optim_cfg=backend_cfg.optim, device=torch.device("cpu"))
    out = runner.run()
    assert out.rounds_completed == 2
    assert all(v.dtype == torch.float32 for v in out.final_state.values())
    assert out.per_round_metrics[0]["client_losses"].keys() == {"c0", "c1"}


def test_localrunner_broadcasts_fresh_state_each_round(backend_cfg, data_cfg, device):
    """Each round's broadcast is the new aggregate, not a stale copy."""

    class _SpyBackend(MockBackend):
        def __init__(self, cfg, dev):
            super().__init__(cfg, dev)
            self.set_state_norms: list[float] = []

        def set_state(self, state, scope="shared"):
            flat = torch.cat([v.detach().float().flatten() for v in state.values()])
            self.set_state_norms.append(float(flat.norm()))
            super().set_state(state, scope=scope)

    instances: list[_SpyBackend] = []

    def factory():
        b = _SpyBackend(backend_cfg, device)
        instances.append(b)
        return b

    runner = LocalRunner(
        backend_factory=factory, strategy=FedAvg(),
        client_ids=["c0", "c1"],
        partition_indices=[list(range(16)), list(range(16, 32))],
        data_cfg=data_cfg, num_rounds=2, local_steps=5,
        optim_cfg=OptimConfig(lr=1e-2), device=device)
    runner.run()

    initial = instances[0].get_state("shared")  # bootstrap backend, never trained
    clients = [b for b in instances if b.set_state_norms]
    assert len(clients) == 2
    for b in clients:
        # exactly one broadcast per round, and round-2's broadcast (the
        # round-1 aggregate) differs from round-1's (the initial state).
        assert len(b.set_state_norms) == 2
        assert b.set_state_norms[0] != b.set_state_norms[1]
        cur = b.get_state("shared")
        assert any(not torch.allclose(cur[k], initial[k]) for k in initial)


def test_localrunner_persistent_state_survives_rounds(backend_factory, data_cfg, backend_cfg):
    """st.persistent rides along across rounds (spec §6)."""

    class _Recorder(FedAvg):
        name = "rounds-recorder"

        def on_round_start(self, ctx):
            ctx.persistent_state.setdefault("rounds_seen", []).append(ctx.round_num)

    runner = LocalRunner(
        backend_factory=backend_factory, strategy=_Recorder(),
        client_ids=["c0"], partition_indices=[list(range(16))],
        data_cfg=data_cfg, num_rounds=3, local_steps=2,
        optim_cfg=backend_cfg.optim, device=torch.device("cpu"))
    runner.run()
    assert runner.client_states["c0"].persistent["rounds_seen"] == [0, 1, 2]


def test_zero_step_round_raises(backend, backend_cfg):
    """An empty or exhausted iterator with steps planned -> ValueError, not a silent 0-step update."""
    from decent_vla.core.strategy import RoundPlan

    optimizer, _ = backend.build_optimizer(backend_cfg.optim)
    gs = backend.get_state("shared")
    with pytest.raises(ValueError, match="0 of 3 planned steps"):
        run_client_round(
            backend=backend, strategy=FedAvg(),
            plan=RoundPlan(selected_clients=["c0"], local_steps=3),
            dataloader=iter([]), client_id="c0", round_num=1,
            global_state=gs, optimizer=optimizer)


def test_localrunner_advances_global_step_by_actual_steps(
    backend_cfg, data_cfg, device
):
    """A non-cycling dataloader runs short in round 2 (1 of 3 steps): st.global_step
    reflects ACTUAL executed steps, not the plan."""
    import itertools

    class _FiniteBackend(MockBackend):
        def build_dataloader(self, data_cfg, partition_indices):
            return iter(itertools.islice(
                super().build_dataloader(data_cfg, partition_indices), 4))

    runner = LocalRunner(
        backend_factory=lambda: _FiniteBackend(backend_cfg, device),
        strategy=FedAvg(), client_ids=["c0"], partition_indices=[list(range(16))],
        data_cfg=data_cfg, num_rounds=2, local_steps=3,
        optim_cfg=backend_cfg.optim, device=device)
    runner.run()
    assert runner.client_states["c0"].global_step == 4  # round 2 executed 1 step


def test_localrunner_fast_forwards_skipped_clients(
    backend_factory, data_cfg, backend_cfg, device
):
    """A client skipped by client-selection catches up to round * local_steps
    before training, keeping spec §6 cumulative-LR exact."""
    from decent_vla.core.strategy import RoundPlan

    class _SelectC0First(FedAvg):
        name = "select-c0-first"

        def configure_round(self, round_num, client_ids, history):
            selected = ["c0"] if round_num == 0 else list(client_ids)
            return RoundPlan(selected_clients=selected, local_steps=0)

    runner = LocalRunner(
        backend_factory=backend_factory, strategy=_SelectC0First(),
        client_ids=["c0", "c1"],
        partition_indices=[list(range(16)), list(range(16, 32))],
        data_cfg=data_cfg, num_rounds=2, local_steps=3,
        optim_cfg=backend_cfg.optim, device=device)
    runner.run()
    assert runner.client_states["c0"].global_step == 6
    # c1 first participates in round 1: gap fast-forward sets it to 3, then +3.
    assert runner.client_states["c1"].global_step == 6


def test_localrunner_moves_the_resident_client_off_the_gpu_before_building_the_next(
    backend_cfg, data_cfg, monkeypatch
):
    """Single-GPU juggling: the next client's model is built on the GPU, so the resident client
    must move to the CPU first. Runs on CPU tensors with the device moves recorded."""
    import decent_vla.runtime.client_state as client_state_mod
    import decent_vla.runtime.local_runner as local_runner_mod

    events = []
    real_build = client_state_mod.build_client_state

    def recording_build(**kw):
        events.append(("build", kw["client_id"]))
        return real_build(**kw)

    monkeypatch.setattr(client_state_mod, "build_client_state", recording_build)
    monkeypatch.setattr(local_runner_mod, "_move_client_state",
                        lambda st, dev: events.append(("move", st.client_id, dev.type)))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    cpu = torch.device("cpu")
    runner = LocalRunner(
        backend_factory=lambda: MockBackend(backend_cfg, cpu),
        strategy=FedAvg(), client_ids=["c0", "c1"],
        partition_indices=[list(range(16)), list(range(16, 32))],
        data_cfg=data_cfg, num_rounds=1, local_steps=1,
        optim_cfg=backend_cfg.optim, device=torch.device("cuda"))
    runner.run()
    assert events.index(("move", "c0", "cpu")) < events.index(("build", "c1"))
