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

from __future__ import annotations

import pytest
import torch

from decent_vla.core.communicator import LocalCommunicator, Topology
from decent_vla.core.strategy import ClientUpdate, RoundPlan, StrategyBase
from decent_vla.runtime.round_loop import RoundFailure, RoundLoopConfig, run_rounds


def test_local_communicator_runs_client_fn_sequentially():
    calls = []

    def client_fn(client_id, state, plan):
        calls.append(client_id)
        return ClientUpdate(delta={"w": torch.ones(2)}, num_samples=4)

    comm = LocalCommunicator(client_fn=client_fn)
    plan = RoundPlan(selected_clients=["c0", "c1"], local_steps=1)
    comm.broadcast_state({"w": torch.zeros(2)}, plan)
    updates = comm.collect_updates(expected=["c0", "c1"], timeout_s=None)
    assert calls == ["c0", "c1"]
    assert len(updates) == 2 and updates[0].num_samples == 4
    assert comm.topology() == Topology(kind="star")


class _CountingStrategy(StrategyBase):
    name = "counting"

    def __init__(self):
        self.configured, self.aggregated = 0, 0

    def configure_round(self, round_num, client_ids, history):
        self.configured += 1
        return super().configure_round(round_num, client_ids, history)

    def aggregate(self, updates, global_state, consume=False, inplace=False):
        self.aggregated += 1
        return super().aggregate(updates, global_state, consume=consume)


def _delta_client_fn(client_id, state, plan):
    return ClientUpdate(delta={"w": torch.ones_like(state["w"])}, num_samples=2,
                        metrics={"loss": 1.0})


def test_run_rounds_fp32_master_and_hooks():
    strat = _CountingStrategy()
    comm = LocalCommunicator(client_fn=_delta_client_fn)
    init = {"w": torch.zeros(3, dtype=torch.bfloat16)}     # bf16 in, fp32 master
    out = run_rounds(strategy=strat, communicator=comm, initial_state=init,
                     client_ids=["c0", "c1"],
                     loop_cfg=RoundLoopConfig(num_rounds=3, local_steps=1))
    assert strat.configured == 3 and strat.aggregated == 3
    assert out.final_state["w"].dtype == torch.float32     # spec §6 fp32 master
    assert torch.allclose(out.final_state["w"], torch.full((3,), 3.0))
    assert out.rounds_completed == 3
    assert len(out.per_round_metrics) == 3
    assert out.per_round_metrics[0]["mean_client_loss"] == 1.0


def test_lean_drain_returns_cached_device_blocks(monkeypatch):
    """The drained bootstrap may have lived on a GPU: the lean path must hand
    its cached blocks back (server process on card 0, next to client 0)."""
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
    comm = LocalCommunicator(client_fn=_delta_client_fn)
    out = run_rounds(strategy=_CountingStrategy(), communicator=comm,
                     initial_state={"w": torch.zeros(3)}, client_ids=["c0"],
                     loop_cfg=RoundLoopConfig(num_rounds=1, local_steps=1,
                                              lean_host_memory=True))
    assert calls and out.rounds_completed == 1


def test_min_clients_enforced():
    class TolerantComm(LocalCommunicator):
        def collect_updates(self, expected, timeout_s=None, min_responses=None):
            # c1 "dies" silently on every attempt
            return super().collect_updates(
                [cid for cid in expected if cid != "c1"], timeout_s)

    comm = TolerantComm(client_fn=_delta_client_fn)
    with pytest.raises(RoundFailure, match=r"round 0.*missing clients: \['c1'\]"):
        run_rounds(strategy=_CountingStrategy(), communicator=comm,
                   initial_state={"w": torch.zeros(2)}, client_ids=["c0", "c1"],
                   loop_cfg=RoundLoopConfig(num_rounds=1, local_steps=1))  # min=all


def test_retry_warnings_name_missing_clients(caplog):
    """Spec §10 diagnostics: the operator must see WHO is missing, not just
    counts — both in the retry warnings and in the final RoundFailure."""
    import logging

    class TolerantComm(LocalCommunicator):
        def collect_updates(self, expected, timeout_s=None, min_responses=None):
            # c1 and c2 die silently on every attempt
            return super().collect_updates(
                [cid for cid in expected if cid not in ("c1", "c2")], timeout_s)

    comm = TolerantComm(client_fn=_delta_client_fn)
    with caplog.at_level(logging.WARNING, logger="decent_vla.runtime.round_loop"):
        with pytest.raises(RoundFailure, match=r"missing clients: \['c1', 'c2'\]"):
            run_rounds(strategy=_CountingStrategy(), communicator=comm,
                       initial_state={"w": torch.zeros(2)},
                       client_ids=["c0", "c1", "c2"],
                       loop_cfg=RoundLoopConfig(num_rounds=1, local_steps=1))
    warnings = [r.getMessage() for r in caplog.records]
    assert len(warnings) == 2
    assert all("missing clients: ['c1', 'c2']" in w for w in warnings)


def test_quorum_is_passed_to_the_communicator():
    """A transport that waits for replies must wait for the round loop's quorum,
    and the retry only for the shortfall."""
    asked = []

    class RecordingComm(LocalCommunicator):
        def collect_updates(self, expected, timeout_s=None, min_responses=None):
            asked.append((list(expected), min_responses))
            # c1 and c2 die silently on every attempt
            return super().collect_updates(
                [cid for cid in expected if cid not in ("c1", "c2")], timeout_s)

    comm = RecordingComm(client_fn=_delta_client_fn)
    with pytest.raises(RoundFailure):
        run_rounds(strategy=_CountingStrategy(), communicator=comm,
                   initial_state={"w": torch.zeros(2)}, client_ids=["c0", "c1", "c2", "c3"],
                   loop_cfg=RoundLoopConfig(num_rounds=1, local_steps=1, min_clients=3))
    # First attempt: quorum 3 of all four. Retry: c0 and c3 answered, so 1 more
    # is needed from the two missing clients.
    assert asked == [(["c0", "c1", "c2", "c3"], 3), (["c1", "c2"], 1)]


def test_min_clients_relaxed_allows_partial():
    class TolerantComm(LocalCommunicator):
        def collect_updates(self, expected, timeout_s=None, min_responses=None):
            return super().collect_updates(
                [cid for cid in expected if cid != "c1"], timeout_s)

    comm = TolerantComm(client_fn=_delta_client_fn)
    out = run_rounds(strategy=_CountingStrategy(), communicator=comm,
                     initial_state={"w": torch.zeros(2)}, client_ids=["c0", "c1"],
                     loop_cfg=RoundLoopConfig(num_rounds=2, local_steps=1, min_clients=1))
    assert out.rounds_completed == 2
    assert out.per_round_metrics[0]["n_updates"] == 1


def test_checkpoint_callback_called_every_n():
    saved = []
    comm = LocalCommunicator(client_fn=_delta_client_fn)
    run_rounds(strategy=_CountingStrategy(), communicator=comm,
               initial_state={"w": torch.zeros(2)}, client_ids=["c0"],
               loop_cfg=RoundLoopConfig(num_rounds=4, local_steps=1,
                                        save_every_n_rounds=2),
               checkpoint_cb=lambda r, s: saved.append(r))
    assert saved == [1, 3]      # 0-indexed rounds: after rounds 2 and 4


def test_configured_local_steps_reaches_clients():
    """B1: StrategyBase's sentinel (0) must defer to RoundLoopConfig.local_steps."""
    seen_steps = []

    def client_fn(client_id, state, plan):
        seen_steps.append(plan.local_steps)
        return ClientUpdate(delta={"w": torch.ones_like(state["w"])}, num_samples=2)

    comm = LocalCommunicator(client_fn=client_fn)
    run_rounds(strategy=StrategyBase(), communicator=comm,
               initial_state={"w": torch.zeros(2)}, client_ids=["c0"],
               loop_cfg=RoundLoopConfig(num_rounds=1, local_steps=250))
    assert seen_steps == [250]


def test_strategy_local_steps_override_beats_config():
    """B1: a positive local_steps from configure_round overrides the runner config."""
    seen_steps = []

    class SevenStepStrategy(StrategyBase):
        def configure_round(self, round_num, client_ids, history):
            plan = super().configure_round(round_num, client_ids, history)
            plan.local_steps = 7
            return plan

    def client_fn(client_id, state, plan):
        seen_steps.append(plan.local_steps)
        return ClientUpdate(delta={"w": torch.ones_like(state["w"])}, num_samples=2)

    comm = LocalCommunicator(client_fn=client_fn)
    run_rounds(strategy=SevenStepStrategy(), communicator=comm,
               initial_state={"w": torch.zeros(2)}, client_ids=["c0"],
               loop_cfg=RoundLoopConfig(num_rounds=1, local_steps=250))
    assert seen_steps == [7]


def test_client_losses_keyed_by_client_id():
    """B3: per-round metrics key client losses by typed ClientUpdate.client_id."""
    comm = LocalCommunicator(client_fn=_delta_client_fn)
    out = run_rounds(strategy=_CountingStrategy(), communicator=comm,
                     initial_state={"w": torch.zeros(2)}, client_ids=["c0", "c1"],
                     loop_cfg=RoundLoopConfig(num_rounds=1, local_steps=1))
    assert list(out.per_round_metrics[0]["client_losses"].keys()) == ["c0", "c1"]


def test_retry_only_reruns_missing_clients():
    """B2: the retry must not re-run clients that already responded."""
    calls: dict[str, int] = {}

    def counting_client_fn(client_id, state, plan):
        calls[client_id] = calls.get(client_id, 0) + 1
        return ClientUpdate(delta={"w": torch.ones_like(state["w"])}, num_samples=2,
                            metrics={"loss": 1.0})

    class FlakyComm(LocalCommunicator):
        def __init__(self, client_fn):
            super().__init__(client_fn)
            self._attempt = 0

        def collect_updates(self, expected, timeout_s=None, min_responses=None):
            self._attempt += 1
            if self._attempt == 1:      # c1 "dies" on the first attempt only
                expected = [cid for cid in expected if cid != "c1"]
            return super().collect_updates(expected, timeout_s)

    comm = FlakyComm(client_fn=counting_client_fn)
    out = run_rounds(strategy=_CountingStrategy(), communicator=comm,
                     initial_state={"w": torch.zeros(2)}, client_ids=["c0", "c1"],
                     loop_cfg=RoundLoopConfig(num_rounds=1, local_steps=1))
    assert out.rounds_completed == 1
    assert calls == {"c0": 1, "c1": 1}                 # c0 ran exactly once
    assert out.per_round_metrics[0]["n_updates"] == 2  # both updates aggregated


def test_start_round_offsets_rounds():
    seen_rounds = []

    class RecordingStrategy(StrategyBase):
        def configure_round(self, round_num, client_ids, history):
            seen_rounds.append(round_num)
            return super().configure_round(round_num, client_ids, history)

    comm = LocalCommunicator(client_fn=_delta_client_fn)
    run_rounds(strategy=RecordingStrategy(), communicator=comm,
               initial_state={"w": torch.zeros(2)}, client_ids=["c0"],
               loop_cfg=RoundLoopConfig(num_rounds=2, local_steps=1, start_round=5))
    assert seen_rounds == [5, 6]
