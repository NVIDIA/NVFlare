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

"""Personalized eval for partial-federation (personalization) arms.

Each client's deployed model = the FINAL server shared subset overlaid on the
client's own persisted full state (its personal params). Every client is
evaluated on its OWN tasks only (from the partition manifest); the pooled
number is the "personalized" success rate. The existing generic eval
(server shared + pretrained rest, all tasks) is untouched.
"""
from __future__ import annotations

import json

import pytest
import torch

from decent_vla.adapters._mock.backend import MockBackend
from decent_vla.configs.schemas import BackendConfig
from decent_vla.runtime.personalized_eval import (
    assemble_personalized_state,
    match_task_names_to_ids,
    pool_client_results,
)


# ------------------------------------------------------------ model assembly

def _backend(shared):
    cfg = BackendConfig(
        name="mock", shared_params=shared, extra={"model_seed": 0}
    )
    return MockBackend.from_config(cfg, torch.device("cpu"))


def test_assemble_overlays_server_shared_on_client_full_state():
    b = _backend(["fc2.*"])
    client_weights = {k: torch.full_like(v, 1.0) for k, v in b.get_state("all").items()}
    server_shared = {k: torch.full_like(v, 2.0) for k, v in b.get_state("shared").items()}
    assemble_personalized_state(b, client_weights, server_shared)
    # Personal half comes from the client blob...
    assert torch.all(b.get_state("personal")["fc1.weight"] == 1.0)
    # ...shared half from the final server aggregate.
    assert torch.all(b.get_state("shared")["fc2.weight"] == 2.0)


def test_assemble_rejects_server_keys_outside_shared_scope():
    """A server blob whose keys spill outside the backend's shared scope means
    the eval config's shared_params don't match the training run — fail loudly
    instead of silently overwriting personal params."""
    b = _backend(["fc2.*"])
    client_weights = b.get_state("all")
    bad_server = {k: v.clone() for k, v in b.get_state("all").items()}  # too many keys
    with pytest.raises(ValueError, match="shared"):
        assemble_personalized_state(b, client_weights, bad_server)


# ------------------------------------------------------- task-name matching

def test_match_task_names_normalizes_underscores_and_case():
    suite_tasks = {
        0: "pick up the black bowl and place it on the plate",
        1: "open the top drawer of the cabinet",
    }
    names = ["Pick_up_the_black_bowl_and_place_it_on_the_plate"]
    assert match_task_names_to_ids(names, suite_tasks) == [0]


def test_match_task_names_unknown_name_raises():
    with pytest.raises(ValueError, match="pick up the moon"):
        match_task_names_to_ids(["pick up the moon"], {0: "open the drawer"})


# ----------------------------------------------------- benchmark task filter

def test_mock_benchmark_honors_task_filter():
    from decent_vla.benchmarks._mock import MockBenchmark
    from decent_vla.configs.schemas import BenchmarkConfig

    cfg = BenchmarkConfig(
        name="mock",
        suites=["default"],
        extra={"task_filter": {"default": [1]}},
    )
    bench = MockBenchmark.from_config(cfg)
    b = _backend(None)
    res = bench.evaluate(
        b.build_policy(), suites=["default"], n_episodes_per_task=2, seed=0
    )
    per_task = res.suites["default"].per_task
    assert [t.task_id for t in per_task] == [1]
    assert res.suites["default"].n_episodes == 2


def test_mock_benchmark_task_names_accessor():
    from decent_vla.benchmarks._mock import MockBenchmark
    from decent_vla.configs.schemas import BenchmarkConfig

    bench = MockBenchmark.from_config(BenchmarkConfig(name="mock", suites=["default"]))
    assert bench.task_names("default") == {0: "mock-task-0", 1: "mock-task-1"}


# ------------------------------------------------------------------ driver e2e

def test_run_personalized_eval_e2e(tmp_path, monkeypatch):
    """End-to-end on mock: 2 clients with disjoint tasks, personalization run
    artifacts on disk → per-client evals on OWN tasks only + pooled number."""
    import json

    from decent_vla.benchmarks._mock import MockBenchmark

    # Bypass entry-point discovery (no editable install in this test env).
    monkeypatch.setattr(
        "decent_vla.runtime.builders.build_benchmark",
        lambda eval_cfg: MockBenchmark.from_config(eval_cfg.benchmark),
    )

    from decent_vla.configs.schemas import (
        BenchmarkConfig,
        DataConfig,
        EvalConfig,
        FLConfig,
        RunConfig,
    )
    from decent_vla.core.partitioner import PartitionManifest
    from decent_vla.runtime.client_checkpoint import save_client_state
    from decent_vla.runtime.client_state import build_client_state
    from decent_vla.runtime.personalized_eval import run_personalized_eval
    from decent_vla.runtime.round_loop import save_server_checkpoint
    from decent_vla.strategies import FedAvg

    run_dir = tmp_path / "run"
    bcfg = BackendConfig(name="mock", shared_params=["fc2.*"], extra={"model_seed": 0})
    device = torch.device("cpu")

    def factory():
        return MockBackend.from_config(bcfg, device)

    # --- fake training artifacts ---
    data_cfg = DataConfig(name="mock", batch_size=4, num_clients=2, seed=42)
    for cid in ("c0", "c1"):
        st = build_client_state(
            client_id=cid, backend_factory=factory, data_cfg=data_cfg,
            optim_cfg=bcfg.optim, partition_indices=None, strategy=FedAvg(),
            start_round=0, local_steps=1,
        )
        save_client_state(run_dir / "client_states", st, round_idx=1)
    boot = factory()
    save_server_checkpoint(run_dir / "checkpoints", 1, boot.get_state("shared"), None)
    PartitionManifest(
        scheme="task_round_robin", num_clients=2, seed=42,
        dataset_repo_id=None, dataset_revision=None,
        held_out_task_indices=[], held_out_task_names=[],
        clients={
            "c0": {"task_names": ["mock-task-0"], "episode_indices": [0]},
            "c1": {"task_names": ["mock-task-3"], "episode_indices": [1]},
        },
        total_episodes=2,
    ).write(run_dir / "partition_manifest.json")

    cfg = RunConfig(
        name="pers-e2e", backend=bcfg, data=data_cfg,
        fl=FLConfig(output_dir=str(run_dir)),
        eval=EvalConfig(
            benchmark=BenchmarkConfig(name="mock", suites=["default", "extra"]),
            n_episodes_per_task=2,
        ),
        seed=42, device="cpu",
    )
    results_path = run_personalized_eval(cfg, factory, run_dir, run_dir / "eval")
    blob = json.loads(results_path.read_text())
    # Each client contributes exactly its own single task x 2 episodes.
    assert blob["per_client"]["c0"]["n_episodes"] == 2
    assert blob["per_client"]["c1"]["n_episodes"] == 2
    assert blob["personalized"]["n_episodes"] == 4
    # c0's task lives in suite "default", c1's in "extra".
    assert list(blob["per_client"]["c0"]["suites"]) == ["default"]
    assert list(blob["per_client"]["c1"]["suites"]) == ["extra"]


# ------------------------------------------------------------- aggregation

def test_pool_client_results_sums_episodes_and_successes():
    per_client = {
        "c0": {"n_episodes": 50, "n_success": 40},
        "c1": {"n_episodes": 50, "n_success": 20},
    }
    pooled = pool_client_results(per_client)
    assert pooled["n_episodes"] == 100
    assert pooled["n_success"] == 60
    assert pooled["success_rate"] == pytest.approx(0.6)


# ------------------------------------------------------- checkpoint choice


def _personalized_run(tmp_path, monkeypatch, client_rounds):
    """Mock personalization-run artifacts with client states saved at `client_rounds`."""
    from decent_vla.benchmarks._mock import MockBenchmark
    from decent_vla.configs.schemas import BenchmarkConfig, DataConfig, EvalConfig, FLConfig, RunConfig
    from decent_vla.core.partitioner import PartitionManifest
    from decent_vla.runtime.client_checkpoint import save_client_state
    from decent_vla.runtime.client_state import build_client_state
    from decent_vla.strategies import FedAvg

    monkeypatch.setattr(
        "decent_vla.runtime.builders.build_benchmark",
        lambda eval_cfg: MockBenchmark.from_config(eval_cfg.benchmark),
    )
    run_dir = tmp_path / "run"
    bcfg = BackendConfig(name="mock", shared_params=["fc2.*"], extra={"model_seed": 0})

    def factory():
        return MockBackend.from_config(bcfg, torch.device("cpu"))

    data_cfg = DataConfig(name="mock", batch_size=4, num_clients=2, seed=42)
    for cid in ("c0", "c1"):
        st = build_client_state(
            client_id=cid, backend_factory=factory, data_cfg=data_cfg,
            optim_cfg=bcfg.optim, partition_indices=None, strategy=FedAvg(),
            start_round=0, local_steps=1,
        )
        for r in client_rounds:
            save_client_state(run_dir / "client_states", st, round_idx=r)
    PartitionManifest(
        scheme="task_round_robin", num_clients=2, seed=42,
        dataset_repo_id=None, dataset_revision=None,
        held_out_task_indices=[], held_out_task_names=[],
        clients={
            "c0": {"task_names": ["mock-task-0"], "episode_indices": [0]},
            "c1": {"task_names": ["mock-task-3"], "episode_indices": [1]},
        },
        total_episodes=2,
    ).write(run_dir / "partition_manifest.json")
    cfg = RunConfig(
        name="pers", backend=bcfg, data=data_cfg, fl=FLConfig(output_dir=str(run_dir)),
        eval=EvalConfig(benchmark=BenchmarkConfig(name="mock", suites=["default", "extra"]),
                        n_episodes_per_task=2),
        seed=42, device="cpu",
    )
    return cfg, factory, run_dir


def test_personalized_eval_uses_the_final_checkpoint_and_the_same_round_client_states(tmp_path, monkeypatch):
    """A 2-round run saved every round: the latest periodic checkpoint is round 0's, the final
    blob is round 1's. Client states exist only for round 1, so only the final blob pairs up."""
    from decent_vla.runtime.personalized_eval import run_personalized_eval
    from decent_vla.runtime.round_loop import save_server_checkpoint

    cfg, factory, run_dir = _personalized_run(tmp_path, monkeypatch, client_rounds=[1])
    shared = factory().get_state("shared")
    save_server_checkpoint(run_dir / "checkpoints", 0, shared, None)
    torch.save({"state": shared, "round": 1}, run_dir / "checkpoints" / "server_round_final.pt")
    results = run_personalized_eval(cfg, factory, run_dir, run_dir / "eval")
    assert json.loads(results.read_text())["server_checkpoint"].endswith("server_round_final.pt")


def test_personalized_eval_skips_a_final_checkpoint_older_than_the_latest_periodic(tmp_path, monkeypatch):
    """A resumed run that stopped early leaves the previous run's final blob (round 0) behind;
    the later periodic checkpoint (round 2) is the one to evaluate."""
    from decent_vla.runtime.personalized_eval import run_personalized_eval
    from decent_vla.runtime.round_loop import save_server_checkpoint

    cfg, factory, run_dir = _personalized_run(tmp_path, monkeypatch, client_rounds=[2])
    shared = factory().get_state("shared")
    save_server_checkpoint(run_dir / "checkpoints", 2, shared, None)
    torch.save({"state": shared, "round": 0}, run_dir / "checkpoints" / "server_round_final.pt")
    results = run_personalized_eval(cfg, factory, run_dir, run_dir / "eval")
    assert json.loads(results.read_text())["server_checkpoint"].endswith("server_round_0003.pt")


def test_personalized_eval_honours_an_explicit_checkpoint(tmp_path, monkeypatch):
    """--ckpt picks the server checkpoint; its round decides which client states are needed."""
    from decent_vla.runtime.personalized_eval import run_personalized_eval
    from decent_vla.runtime.round_loop import save_server_checkpoint

    cfg, factory, run_dir = _personalized_run(tmp_path, monkeypatch, client_rounds=[1])
    shared = factory().get_state("shared")
    save_server_checkpoint(run_dir / "checkpoints", 0, shared, None)
    save_server_checkpoint(run_dir / "checkpoints", 1, shared, None)
    with pytest.raises(FileNotFoundError, match="round 0"):
        run_personalized_eval(cfg, factory, run_dir, run_dir / "eval",
                              ckpt_path=run_dir / "checkpoints" / "server_round_0001.pt")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        run_personalized_eval(cfg, factory, run_dir, run_dir / "eval",
                              ckpt_path=run_dir / "checkpoints" / "server_round_0009.pt")


def test_personalized_cache_misses_after_a_retrain_into_the_same_paths(tmp_path, monkeypatch):
    """Rewriting a checkpoint at the same path changes its identity, so cached scores are not reused."""
    import os

    from decent_vla.runtime.personalized_eval import run_personalized_eval
    from decent_vla.runtime.round_loop import save_server_checkpoint

    cfg, factory, run_dir = _personalized_run(tmp_path, monkeypatch, client_rounds=[0])
    save_server_checkpoint(run_dir / "checkpoints", 0, factory().get_state("shared"), None)
    run_personalized_eval(cfg, factory, run_dir, run_dir / "eval")
    client_json = run_dir / "eval" / "personalized_clients" / "c0.json"
    first = client_json.stat().st_mtime_ns
    ckpt = run_dir / "checkpoints" / "server_round_0001.pt"
    st = ckpt.stat()
    os.utime(ckpt, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))  # same path, "retrained"
    run_personalized_eval(cfg, factory, run_dir, run_dir / "eval")
    assert client_json.stat().st_mtime_ns != first


def test_cli_eval_passes_ckpt_to_personalized_eval(tmp_path, monkeypatch):
    """`decent-vla eval --ckpt X` on a personalized config hands X to run_personalized_eval."""
    import yaml

    from decent_vla.cli.main import main

    seen = {}

    def fake_run(cfg, backend_factory, run_dir, eval_dir, ckpt_path=None):
        seen["ckpt_path"] = ckpt_path
        return eval_dir / "personalized_results.json"

    monkeypatch.setattr("decent_vla.runtime.personalized_eval.run_personalized_eval", fake_run)
    cfg = {
        "name": "pers_cli", "device": "cpu",
        "backend": {"name": "mock", "shared_params": ["fc2.*"]},
        "data": {"name": "mock", "batch_size": 4, "num_clients": 2},
        "fl": {"strategy": {"name": "fedavg"}, "num_rounds": 1, "local_steps": 1,
               "persist_client_state": True, "output_dir": str(tmp_path / "out")},
        "eval": {"benchmark": {"name": "mock", "suites": ["default"]}, "personalized": True},
    }
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    ckpt = tmp_path / "out" / "checkpoints" / "server_round_0001.pt"
    assert main(["eval", "--config", str(cfg_path), "--ckpt", str(ckpt)]) == 0
    assert str(seen["ckpt_path"]) == str(ckpt)


def test_personalized_eval_takes_each_clients_newest_state_up_to_the_server_round(tmp_path, monkeypatch):
    """Client states from rounds 0 and 2, server checkpoint from round 1: each client uses its
    round-0 state (a client that sat out round 1 keeps it), never the later round-2 state."""
    from decent_vla.runtime.personalized_eval import run_personalized_eval
    from decent_vla.runtime.round_loop import save_server_checkpoint

    cfg, factory, run_dir = _personalized_run(tmp_path, monkeypatch, client_rounds=[0, 2])
    save_server_checkpoint(run_dir / "checkpoints", 1, factory().get_state("shared"), None)
    run_personalized_eval(cfg, factory, run_dir, run_dir / "eval")
    for cid in ("c0", "c1"):
        blob = json.loads((run_dir / "eval" / "personalized_clients" / f"{cid}.json").read_text())
        assert blob["provenance"]["client_checkpoint"].endswith(f"{cid}_round_0000.pt")
        assert blob["client_round"] == 0
