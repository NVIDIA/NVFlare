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

"""Smoke tests for the CLI: list, info, doctor, run (end-to-end with MockBackend)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from decent_vla.cli.main import build_parser, main


def test_cli_list_strategies(capsys):
    rc = main(["list", "strategies"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "fedavg" in captured.out
    assert "fedadam" in captured.out
    assert "fedprox" in captured.out


def test_cli_list_backends(capsys):
    rc = main(["list", "backends"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "mock" in captured.out


def test_cli_list_benchmarks(capsys):
    rc = main(["list", "benchmarks"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "mock" in captured.out


def test_cli_info(capsys):
    rc = main(["info", "strategy", "fedprox"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "fedprox" in captured.out


def test_cli_doctor(capsys, monkeypatch):
    monkeypatch.setenv("HF_HOME", "/nonexistent/hf_home_for_doctor_test")
    rc = main(["doctor"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Backends:" in captured.out
    assert "Strategies:" in captured.out
    # informational dependency checks (spec §10)
    assert "nvflare:" in captured.out
    assert "peft:" in captured.out
    assert (
        "WARN: HF_HOME=/nonexistent/hf_home_for_doctor_test does not exist"
        in captured.out
    )


def test_cli_run_e2e(tmp_path, monkeypatch, capsys):
    """End-to-end: a 2-round FedAvg FL run against MockBackend + MockBenchmark."""
    monkeypatch.chdir(tmp_path)
    repo_root = Path(__file__).resolve().parent.parent
    cfg_path = repo_root / "examples" / "mock_fl_fedavg.yaml"
    assert cfg_path.exists(), f"missing example config at {cfg_path}"
    rc = main(["run", "--config", str(cfg_path)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "FL training complete" in captured.out
    assert "Eval complete" in captured.out

    # results.json was written
    results_path = tmp_path / "outputs" / "mock_fl_fedavg" / "eval" / "results.json"
    assert results_path.exists()
    data = json.loads(results_path.read_text())
    assert data["schema_version"] == "1.2"
    assert data["backend"].startswith("mock")
    assert "default" in data["suites"]
    assert "success_rate" in data["overall"]


def test_parser_help_does_not_crash():
    p = build_parser()
    # We don't call parser.parse_args([]) because that would error on missing subcommand;
    # just verify the parser builds and has subcommands.
    assert p is not None
    actions = [a for a in p._actions if a.dest == "cmd"]
    assert len(actions) == 1
    assert {"list", "info", "doctor", "run", "train", "eval", "partition"} <= set(actions[0].choices)


# Every example must parse as a RunConfig (no backend build — pure config
# validation). The `*_pipeline` companion files (train_pipeline, eval_pipeline)
# are lerobot-native TrainPipelineConfig YAMLs, not decent-vla RunConfigs, so
# they are excluded.
_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
_EXAMPLE_CONFIGS = sorted(
    p for p in _EXAMPLES_DIR.glob("*.yaml") if "_pipeline" not in p.name
)


@pytest.mark.parametrize("example", _EXAMPLE_CONFIGS, ids=lambda p: p.name)
def test_example_config_parses(example):
    from decent_vla.configs.loader import load_run_config

    cfg = load_run_config(example)
    assert cfg.name
    assert cfg.backend.name


def test_example_configs_discovered():
    # Guard against the glob silently matching nothing (e.g. moved directory).
    assert len(_EXAMPLE_CONFIGS) >= 8


def test_config_validation_rejects_missing_train_and_fl(tmp_path):
    from decent_vla.configs.loader import ConfigError, load_run_config

    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        "name: bad\nbackend:\n  name: mock\ndata:\n  num_clients: 1\n"
    )
    with pytest.raises(ConfigError):
        load_run_config(cfg)


def test_config_validation_rejects_both_train_and_fl(tmp_path):
    from decent_vla.configs.loader import ConfigError, load_run_config

    cfg = tmp_path / "both.yaml"
    cfg.write_text(
        "name: both\nbackend:\n  name: mock\n"
        "data:\n  num_clients: 1\n"
        "train:\n  num_steps: 1\n"
        "fl:\n  num_rounds: 1\n  local_steps: 1\n"
    )
    with pytest.raises(ConfigError):
        load_run_config(cfg)


def test_config_validation_rejects_eval_all_tasks_false(tmp_path):
    """eval_all_tasks=false parses but has no implementation in v0.2 —
    reject it loudly instead of silently evaluating everything anyway."""
    from decent_vla.configs.loader import ConfigError, load_run_config

    cfg = tmp_path / "eat.yaml"
    cfg.write_text(
        "name: t\nbackend:\n  name: mock\n"
        "fl:\n  num_rounds: 1\n  local_steps: 1\n"
        "eval:\n  eval_all_tasks: false\n"
    )
    with pytest.raises(ConfigError, match="eval_all_tasks"):
        load_run_config(cfg)


def test_nested_partitioning_block_and_execution_alias(tmp_path):
    """Spec §9 YAML shape: nested data.partitioning dict + fl.execution alias."""
    import yaml

    from decent_vla.configs import load_run_config

    cfg_yaml = {
        "name": "t",
        "backend": {"name": "mock"},
        "data": {
            "repo_id": "mock/ds",
            "partitioning": {
                "scheme": "task_round_robin",
                "num_clients": 8,
                "held_out": "configs/libero_heldout_8.json",
                "seed": 7,
            },
        },
        "fl": {
            "strategy": {"name": "fedavg"},
            "rounds": 50,
            "local_steps": 250,
            "execution": "nvflare",
            "min_clients": 6,
            "round_timeout_s": 3600.0,
        },
    }
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg_yaml))
    cfg = load_run_config(p)
    assert cfg.data.partitioning == "task_round_robin"
    assert cfg.data.num_clients == 8
    assert cfg.data.held_out == "configs/libero_heldout_8.json"
    assert cfg.data.seed == 7
    assert cfg.fl.mode == "nvflare"
    assert cfg.fl.num_rounds == 50  # `rounds` alias
    assert cfg.fl.min_clients == 6
    assert cfg.fl.round_timeout_s == 3600.0


def test_flat_partitioning_string_still_works(tmp_path):
    import yaml

    from decent_vla.configs import load_run_config

    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "name": "t",
                "backend": {"name": "mock"},
                "data": {"partitioning": "iid", "num_clients": 2},
                "train": {"num_steps": 1},
            }
        )
    )
    cfg = load_run_config(p)
    assert cfg.data.partitioning == "iid"
    assert cfg.data.held_out is None


def test_fl_batch_size_sugar_lands_in_data(tmp_path):
    import yaml

    from decent_vla.configs import load_run_config

    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "name": "t",
                "backend": {"name": "mock"},
                "fl": {"strategy": {"name": "fedavg"}, "rounds": 2, "batch_size": 32},
            }
        )
    )
    cfg = load_run_config(p)
    assert cfg.data.batch_size == 32
    assert cfg.fl.num_rounds == 2


def test_null_data_block_with_fl_batch_size(tmp_path):
    """data: null (explicit YAML null) plus fl.batch_size must not raise TypeError."""
    import yaml

    from decent_vla.configs import load_run_config

    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "name": "t",
                "backend": {"name": "mock"},
                "data": None,
                "fl": {"strategy": {"name": "fedavg"}, "rounds": 2, "batch_size": 16},
            }
        )
    )
    cfg = load_run_config(p)
    assert cfg.data.batch_size == 16


# ----------------------------------------------------------------------------
# partition subcommand, partition manifest, server checkpoints, --resume
# ----------------------------------------------------------------------------


def test_partition_subcommand_writes_manifest(tmp_path):
    import yaml

    cfg = {"name": "t", "backend": {"name": "mock"},
           "data": {"partitioning": {"scheme": "task_round_robin", "num_clients": 2, "seed": 1},
                    "extra": {"n_tasks": 8, "eps_per_task": 4}},
           "fl": {"strategy": {"name": "fedavg"}, "rounds": 1, "local_steps": 1}}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    rc = main(["partition", "--config", str(p), "--output-dir", str(tmp_path / "out")])
    assert rc == 0
    man = json.loads((tmp_path / "out" / "partition_manifest.json").read_text())
    assert man["num_clients"] == 2 and man["total_episodes"] == 32
    eps = json.loads((tmp_path / "out" / "train_episodes.json").read_text())
    assert sorted(eps) == sorted(i for c in man["clients"].values()
                                 for i in c["episode_indices"])


def _write_fl_cfg(tmp_path: Path, fname: str, num_rounds: int, output_dir: Path) -> Path:
    import yaml

    cfg = {
        "name": "ckpt_run",
        "device": "cpu",
        "backend": {"name": "mock"},
        "data": {"name": "mock", "batch_size": 4, "num_clients": 2, "partitioning": "iid"},
        "fl": {"strategy": {"name": "fedavg"}, "num_rounds": num_rounds, "local_steps": 2,
               "save_every_n_rounds": 1, "mode": "local", "output_dir": str(output_dir)},
    }
    p = tmp_path / fname
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_fl_run_writes_manifest_and_atomic_server_checkpoints(tmp_path):
    import torch

    out = tmp_path / "out"
    cfg_path = _write_fl_cfg(tmp_path, "c.yaml", num_rounds=2, output_dir=out)
    assert main(["run", "--config", str(cfg_path)]) == 0

    assert (out / "partition_manifest.json").exists()
    ckpts = sorted((out / "checkpoints").glob("server_round_*.pt"))
    assert [p.name for p in ckpts] == ["server_round_0001.pt", "server_round_0002.pt"]
    assert not list((out / "checkpoints").glob("*.tmp"))  # atomic: no leftover temp files
    blob = torch.load(ckpts[-1], map_location="cpu", weights_only=True)
    assert blob["round"] == 1
    assert all(isinstance(v, torch.Tensor) for v in blob["state"].values())


def test_fl_run_resume_from_server_checkpoint(tmp_path):
    import torch

    out = tmp_path / "out"
    cfg_path = _write_fl_cfg(tmp_path, "c.yaml", num_rounds=2, output_dir=out)
    assert main(["run", "--config", str(cfg_path)]) == 0
    before = {p.name for p in (out / "checkpoints").glob("server_round_*.pt")}
    assert before == {"server_round_0001.pt", "server_round_0002.pt"}

    # Resume from round 2's checkpoint. cfg.fl.num_rounds is the TOTAL round
    # count for the run (consistent with the NVFlare path), so total=3 from
    # start_round=2 trains only round index 2 (no round-1 retrain) and writes
    # exactly one new checkpoint.
    cfg_path2 = _write_fl_cfg(tmp_path, "c2.yaml", num_rounds=3, output_dir=out)
    resume_from = out / "checkpoints" / "server_round_0002.pt"
    assert main(["run", "--config", str(cfg_path2), "--resume", str(resume_from)]) == 0

    after = {p.name for p in (out / "checkpoints").glob("server_round_*.pt")}
    assert after - before == {"server_round_0003.pt"}
    assert len(after) == 3
    blob = torch.load(out / "checkpoints" / "server_round_0003.pt",
                      map_location="cpu", weights_only=True)
    assert blob["round"] == 2


def test_fl_run_nvflare_mode_dispatches_to_entry(tmp_path, monkeypatch):
    """execution: nvflare must dispatch to the nvflare entry (not train
    in-process here, not fall through to the unknown-mode error)."""
    import sys
    import types

    import yaml

    calls = {}

    def fake_run_from_cli(cfg, config_path, resume=None):
        calls["mode"] = cfg.fl.mode
        calls["config_path"] = config_path
        calls["resume"] = resume

    # Stub the whole entry module in sys.modules: the CLI imports it lazily
    # inside the dispatch branch, and a plain setattr on the imported module
    # can be defeated by tests/nvflare's module-eviction fixture having left a
    # different module object behind.
    stub = types.ModuleType("decent_vla.runtime.nvflare.entry")
    stub.run_from_cli = fake_run_from_cli
    monkeypatch.setitem(sys.modules, "decent_vla.runtime.nvflare.entry", stub)
    cfg = {
        "name": "t",
        "device": "cpu",
        "backend": {"name": "mock"},
        "data": {"name": "mock", "batch_size": 4, "num_clients": 2, "partitioning": "iid"},
        "fl": {"strategy": {"name": "fedavg"}, "rounds": 2, "local_steps": 2,
               "execution": "nvflare", "output_dir": str(tmp_path / "out")},
    }
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("DECENT_VLA_CONFIG", str(p))
    rc = main(["run", "--config", str(p)])
    assert rc == 0
    assert calls["mode"] == "nvflare"
    assert calls["config_path"] == str(p)


def test_fl_run_unknown_mode_raises(tmp_path):
    import yaml

    cfg = {
        "name": "t",
        "device": "cpu",
        "backend": {"name": "mock"},
        "data": {"name": "mock", "batch_size": 4, "num_clients": 2, "partitioning": "iid"},
        "fl": {"strategy": {"name": "fedavg"}, "rounds": 1, "local_steps": 1,
               "mode": "ray", "output_dir": str(tmp_path / "out")},
    }
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(SystemExit, match="Unknown fl.execution mode"):
        main(["run", "--config", str(p)])


# ----------------------------------------------------------------------------
# _latest_checkpoint + --resume validation (checkpoint discovery hardening)
# ----------------------------------------------------------------------------


def test_latest_checkpoint_only_server_blobs_raises(tmp_path):
    """If only server_round_*.pt files exist, _latest_checkpoint must raise SystemExit."""
    from decent_vla.cli.main import _latest_checkpoint

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    import torch

    torch.save({"state": {}, "round": 2}, ckpt_dir / "server_round_0002.pt")
    with pytest.raises(SystemExit, match="Only server-state checkpoints"):
        _latest_checkpoint(tmp_path)


def test_latest_checkpoint_prefers_backend_ckpt_over_server_blob(tmp_path):
    """With both round_*.pt and server_round_*.pt present, return the backend one."""
    import torch

    from decent_vla.cli.main import _latest_checkpoint

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    torch.save({"state": {}, "round": 2}, ckpt_dir / "server_round_0002.pt")
    # backend checkpoint — just needs to exist as a file
    backend_ckpt = ckpt_dir / "round_0002.pt"
    torch.save({}, backend_ckpt)
    result = _latest_checkpoint(tmp_path)
    assert result == backend_ckpt


def test_latest_checkpoint_empty_dir_returns_none(tmp_path):
    """No checkpoints at all → return None (no crash)."""
    from decent_vla.cli.main import _latest_checkpoint

    (tmp_path / "checkpoints").mkdir()
    assert _latest_checkpoint(tmp_path) is None


def test_resume_missing_round_key_raises(tmp_path):
    """--resume pointed at a blob without 'round' must raise SystemExit mentioning --resume."""
    import torch
    import yaml

    out = tmp_path / "out"
    cfg = {
        "name": "t",
        "device": "cpu",
        "backend": {"name": "mock"},
        "data": {"name": "mock", "batch_size": 4, "num_clients": 2, "partitioning": "iid"},
        "fl": {
            "strategy": {"name": "fedavg"},
            "num_rounds": 1,
            "local_steps": 2,
            "mode": "local",
            "output_dir": str(out),
        },
    }
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    bogus_ckpt = tmp_path / "bogus.pt"
    torch.save({"weights": {}, "epoch": 5}, bogus_ckpt)

    from decent_vla.cli.main import main

    with pytest.raises(SystemExit, match="--resume expects"):
        main(["run", "--config", str(cfg_path), "--resume", str(bogus_ckpt)])


# ----------------------------------------------------------------------------
# Results v1.1: tags, aggregates, idempotent per-suite eval
# ----------------------------------------------------------------------------


def test_run_eval_v11_tags_aggregates_and_idempotent_resume(tmp_path, caplog):
    """E2E mock run: v1.1 results carry tags + aggregates; re-eval skips done suites."""
    import logging

    import yaml

    out = tmp_path / "out"
    cfg = {
        "name": "v11_run",
        "device": "cpu",
        "backend": {"name": "mock"},
        "data": {"name": "mock", "batch_size": 4, "num_clients": 2, "partitioning": "iid"},
        "fl": {"strategy": {"name": "fedavg"}, "num_rounds": 1, "local_steps": 2,
               "save_every_n_rounds": 1, "mode": "local", "output_dir": str(out)},
        "eval": {"benchmark": {"name": "mock", "suites": ["default"]},
                 "n_episodes_per_task": 2, "seed": 0},
    }
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    assert main(["run", "--config", str(cfg_path)]) == 0

    data = json.loads((out / "eval" / "results.json").read_text())
    assert data["schema_version"] == "1.2"
    assert data["aggregates"]["overall"]["n_episodes"] > 0
    assert data["partition_manifest_hash"] is not None  # FL run wrote a manifest
    for s in data["suites"].values():
        assert s["per_task"], "per_task must not be empty"
        for t in s["per_task"]:
            assert t["tags"] == ["seen"]  # no held-out spec in the mock run

    suite_file = out / "eval" / "suites" / "default.json"
    assert suite_file.exists()

    # Re-invoke eval-only with an explicit --ckpt: the suite must be SKIPPED.
    mtime_before = suite_file.stat().st_mtime_ns
    ckpt = sorted((out / "checkpoints").glob("round_*.pt"))[-1]
    with caplog.at_level(logging.INFO):
        assert main(["eval", "--config", str(cfg_path), "--ckpt", str(ckpt)]) == 0
    assert suite_file.stat().st_mtime_ns == mtime_before
    assert any(
        "skipping default: already evaluated" in r.getMessage() for r in caplog.records
    )
    # eval-only re-run must not have trained any further rounds
    assert {p.name for p in (out / "checkpoints").glob("server_round_*.pt")} == {
        "server_round_0001.pt"
    }


def _write_train_eval_cfg(
    tmp_path: Path,
    fname: str,
    num_rounds: int,
    output_dir: Path,
    local_steps: int = 2,
    suites: list[str] | None = None,
    benchmark_extra: dict | None = None,
) -> Path:
    import yaml

    cfg = {
        "name": "prov_run",
        "device": "cpu",
        "backend": {"name": "mock"},
        "data": {"name": "mock", "batch_size": 4, "num_clients": 2, "partitioning": "iid"},
        "fl": {"strategy": {"name": "fedavg"}, "num_rounds": num_rounds,
               "local_steps": local_steps,
               "save_every_n_rounds": 1, "mode": "local", "output_dir": str(output_dir)},
        "eval": {"benchmark": {"name": "mock", "suites": suites or ["default"],
                               "extra": benchmark_extra or {}},
                 "n_episodes_per_task": 2, "seed": 0},
    }
    p = tmp_path / fname
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_stale_suite_cache_invalidated_by_new_checkpoint(tmp_path, caplog):
    """A1: train+eval twice into the SAME output_dir; the second run produces a
    new checkpoint, so the cached suite must be re-evaluated, not reused."""
    import logging

    out = tmp_path / "out"
    cfg1 = _write_train_eval_cfg(tmp_path, "c1.yaml", num_rounds=1, output_dir=out)
    assert main(["run", "--config", str(cfg1)]) == 0
    first = json.loads((out / "eval" / "results.json").read_text())
    first_ckpt = first["checkpoint"]
    assert first_ckpt is not None

    # Second train+eval, same output_dir: trains 2 rounds, so the latest
    # backend checkpoint is a NEW file (round_0002.pt).
    cfg2 = _write_train_eval_cfg(tmp_path, "c2.yaml", num_rounds=2, output_dir=out)
    with caplog.at_level(logging.INFO):
        assert main(["run", "--config", str(cfg2)]) == 0
    second = json.loads((out / "eval" / "results.json").read_text())
    assert second["checkpoint"] != first_ckpt
    assert second["checkpoint"].endswith("round_0002.pt")
    messages = [r.getMessage() for r in caplog.records]
    assert any("re-evaluating default: provenance changed" in m for m in messages)
    assert not any("skipping default: already evaluated" in m for m in messages)
    # The suite file on disk must carry the NEW provenance.
    suite = json.loads((out / "eval" / "suites" / "default.json").read_text())
    assert suite["provenance"]["checkpoint"] == second["checkpoint"]
    # results.json suites must NOT leak the provenance cache key.
    assert "provenance" not in second["suites"]["default"]


def test_stale_suite_cache_invalidated_by_retrain_at_same_path(tmp_path, caplog):
    """The silent-wrong-results case: retraining with DIFFERENT hyperparameters
    but the same `num_rounds` overwrites `round_{num_rounds:04d}.pt` — same path,
    different weights. The cache must miss, or the new run inherits the old
    policy's success rate while results.json advertises the new config_hash."""
    import hashlib
    import logging

    out = tmp_path / "out"
    cfg1 = _write_train_eval_cfg(tmp_path, "c1.yaml", num_rounds=1, output_dir=out, local_steps=2)
    assert main(["run", "--config", str(cfg1)]) == 0
    ckpt = out / "checkpoints" / "round_0001.pt"
    suite_file = out / "eval" / "suites" / "default.json"
    first_md5 = hashlib.md5(ckpt.read_bytes()).hexdigest()
    first_mtime = suite_file.stat().st_mtime_ns
    first_hash = json.loads((out / "eval" / "results.json").read_text())["config_hash"]

    # Same output_dir, same num_rounds (so the SAME checkpoint filename), but a
    # different training config → genuinely different weights.
    cfg2 = _write_train_eval_cfg(tmp_path, "c2.yaml", num_rounds=1, output_dir=out, local_steps=7)
    with caplog.at_level(logging.INFO):
        assert main(["run", "--config", str(cfg2)]) == 0

    second = json.loads((out / "eval" / "results.json").read_text())
    # Preconditions: the path really is identical and the weights really changed.
    assert second["checkpoint"] == str(ckpt)
    assert hashlib.md5(ckpt.read_bytes()).hexdigest() != first_md5
    assert second["config_hash"] != first_hash
    # …so the suite must have been re-evaluated, not served from cache.
    messages = [r.getMessage() for r in caplog.records]
    assert any("re-evaluating default: provenance changed" in m for m in messages)
    assert not any("skipping default: already evaluated" in m for m in messages)
    assert suite_file.stat().st_mtime_ns != first_mtime


def test_suite_cache_survives_training_only_config_edit(tmp_path, caplog):
    """Deliberate scope of the key: it identifies the *evaluated model*, not the
    whole run config. Editing a training knob without retraining must NOT throw
    away finished (expensive) rollouts."""
    import logging

    out = tmp_path / "out"
    cfg1 = _write_train_eval_cfg(tmp_path, "c1.yaml", num_rounds=1, output_dir=out, local_steps=2)
    assert main(["run", "--config", str(cfg1)]) == 0
    suite_file = out / "eval" / "suites" / "default.json"
    mtime_before = suite_file.stat().st_mtime_ns

    # local_steps changed but no retraining happens (eval-only) → same weights.
    cfg2 = _write_train_eval_cfg(tmp_path, "c2.yaml", num_rounds=1, output_dir=out, local_steps=99)
    ckpt = out / "checkpoints" / "round_0001.pt"
    with caplog.at_level(logging.INFO):
        assert main(["eval", "--config", str(cfg2), "--ckpt", str(ckpt)]) == 0
    assert any(
        "skipping default: already evaluated" in r.getMessage() for r in caplog.records
    )
    assert suite_file.stat().st_mtime_ns == mtime_before


def test_suite_cache_invalidated_by_eval_protocol_change(tmp_path, caplog):
    """Same weights, different eval protocol (benchmark.extra) → cache must miss:
    the cached numbers were produced under a protocol this run isn't asking for."""
    import logging

    out = tmp_path / "out"
    cfg1 = _write_train_eval_cfg(tmp_path, "c1.yaml", num_rounds=1, output_dir=out)
    assert main(["run", "--config", str(cfg1)]) == 0
    suite = json.loads((out / "eval" / "suites" / "default.json").read_text())
    assert len(suite["per_task"]) == 2

    # task_filter narrows the suite to one task — cached 2-task numbers would be wrong.
    cfg2 = _write_train_eval_cfg(
        tmp_path, "c2.yaml", num_rounds=1, output_dir=out,
        benchmark_extra={"task_filter": {"default": [0]}},
    )
    ckpt = out / "checkpoints" / "round_0001.pt"
    with caplog.at_level(logging.INFO):
        assert main(["eval", "--config", str(cfg2), "--ckpt", str(ckpt)]) == 0
    assert any(
        "re-evaluating default: provenance changed" in r.getMessage() for r in caplog.records
    )
    refreshed = json.loads((out / "eval" / "suites" / "default.json").read_text())
    assert len(refreshed["per_task"]) == 1


def test_adding_a_suite_does_not_invalidate_finished_suites(tmp_path, caplog):
    """`eval.benchmark.suites` is deliberately OUT of the key: the cache is
    per-suite, so extending the suite list must resume, not re-run everything."""
    import logging

    out = tmp_path / "out"
    cfg1 = _write_train_eval_cfg(tmp_path, "c1.yaml", num_rounds=1, output_dir=out,
                                 suites=["default"])
    assert main(["run", "--config", str(cfg1)]) == 0
    mtime_before = (out / "eval" / "suites" / "default.json").stat().st_mtime_ns

    cfg2 = _write_train_eval_cfg(tmp_path, "c2.yaml", num_rounds=1, output_dir=out,
                                 suites=["default", "extra"])
    ckpt = out / "checkpoints" / "round_0001.pt"
    with caplog.at_level(logging.INFO):
        assert main(["eval", "--config", str(cfg2), "--ckpt", str(ckpt)]) == 0
    assert any(
        "skipping default: already evaluated" in r.getMessage() for r in caplog.records
    )
    assert (out / "eval" / "suites" / "default.json").stat().st_mtime_ns == mtime_before
    assert (out / "eval" / "suites" / "extra.json").exists()


def test_eval_with_a_missing_ckpt_raises_instead_of_evaluating_the_base_model(tmp_path):
    """A mistyped or not-yet-written --ckpt used to be skipped silently: the
    base model was evaluated and results.json named the missing path."""
    out = tmp_path / "out"
    cfg_path = _write_train_eval_cfg(tmp_path, "c.yaml", num_rounds=1, output_dir=out)
    missing = out / "checkpoints" / "round_0100.pt"
    with pytest.raises(FileNotFoundError, match="round_0100.pt"):
        main(["eval", "--config", str(cfg_path), "--ckpt", str(missing)])
    assert not (out / "eval").exists()


def test_eval_without_a_ckpt_still_evaluates_the_base_model(tmp_path):
    """No --ckpt and no training output: evaluating the base model is intended."""
    out = tmp_path / "out"
    cfg_path = _write_train_eval_cfg(tmp_path, "c.yaml", num_rounds=1, output_dir=out)
    assert main(["eval", "--config", str(cfg_path)]) == 0
    data = json.loads((out / "eval" / "results.json").read_text())
    assert data["checkpoint"] is None
    assert data["provenance"]["checkpoint"] is None
    assert data["overall"]["n_episodes"] > 0


def test_cli_train_does_not_eval(tmp_path):
    """A2: `decent-vla train` with an eval block in the config must not eval."""
    out = tmp_path / "out"
    cfg_path = _write_train_eval_cfg(tmp_path, "c.yaml", num_rounds=1, output_dir=out)
    assert main(["train", "--config", str(cfg_path)]) == 0
    # Training happened…
    assert list((out / "checkpoints").glob("round_*.pt"))
    # …but no eval artifacts were produced.
    assert not (out / "eval").exists()
    assert not list(out.rglob("results.json"))


def test_unknown_partitioning_keys_raise_config_error(tmp_path):
    """Typo'd keys inside data.partitioning (e.g. 'sceme') must raise ConfigError."""
    import yaml

    from decent_vla.configs.loader import ConfigError, load_run_config

    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "name": "t",
                "backend": {"name": "mock"},
                "data": {
                    "partitioning": {
                        "sceme": "task",  # typo: should be 'scheme'
                        "num_clients": 4,
                    }
                },
                "fl": {"strategy": {"name": "fedavg"}, "rounds": 2, "local_steps": 1},
            }
        )
    )
    with pytest.raises(ConfigError, match="sceme"):
        load_run_config(p)


# --- unknown-key rejection on the field-by-field blocks ---------------------
# Same rationale as the `wandb:` guard: a key that is read with `.get()` and
# never matched is indistinguishable from one that was never written. The
# stakes here are higher than a misrouted curve — `save_every_n_round` (no
# trailing s) silently produces zero checkpoints for a whole multi-day run.


def _cfg_file(tmp_path, body: dict):
    import yaml

    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(body))
    return p


def test_unknown_fl_key_raises_config_error(tmp_path):
    """`save_every_n_round` (missing the trailing s) must not be silently dropped."""
    from decent_vla.configs.loader import ConfigError, load_run_config

    p = _cfg_file(tmp_path, {
        "name": "t",
        "backend": {"name": "mock"},
        "fl": {"rounds": 2, "local_steps": 1, "save_every_n_round": 5},
    })
    with pytest.raises(ConfigError, match=r"fl.*save_every_n_round"):
        load_run_config(p)


def test_unknown_backend_key_raises_config_error(tmp_path):
    from decent_vla.configs.loader import ConfigError, load_run_config

    p = _cfg_file(tmp_path, {
        "name": "t",
        "backend": {"name": "mock", "pretrained_pth": "/ckpt"},  # typo: pretrained_path
        "fl": {"rounds": 2, "local_steps": 1},
    })
    with pytest.raises(ConfigError, match=r"backend.*pretrained_pth"):
        load_run_config(p)


def test_unknown_eval_key_raises_config_error(tmp_path):
    from decent_vla.configs.loader import ConfigError, load_run_config

    p = _cfg_file(tmp_path, {
        "name": "t",
        "backend": {"name": "mock"},
        "fl": {"rounds": 2, "local_steps": 1},
        "eval": {"n_episodes_per_taks": 10},  # typo: n_episodes_per_task
    })
    with pytest.raises(ConfigError, match=r"eval.*n_episodes_per_taks"):
        load_run_config(p)


def test_unknown_eval_benchmark_key_raises_config_error(tmp_path):
    from decent_vla.configs.loader import ConfigError, load_run_config

    p = _cfg_file(tmp_path, {
        "name": "t",
        "backend": {"name": "mock"},
        "fl": {"rounds": 2, "local_steps": 1},
        "eval": {"benchmark": {"name": "mock", "suits": ["default"]}},  # typo: suites
    })
    with pytest.raises(ConfigError, match=r"eval\.benchmark.*suits"):
        load_run_config(p)


def test_unknown_key_error_lists_the_expected_keys(tmp_path):
    from decent_vla.configs.loader import ConfigError, load_run_config

    p = _cfg_file(tmp_path, {
        "name": "t",
        "backend": {"name": "mock"},
        "fl": {"rounds": 2, "local_steps": 1, "client_gpu_sprad": True},
    })
    with pytest.raises(ConfigError, match="client_gpu_spread") as exc:
        load_run_config(p)
    assert "client_gpu_sprad" in str(exc.value)


def test_nested_blocks_are_not_counted_as_unknown_at_the_parent(tmp_path):
    """fl.strategy / fl.nvflare / backend.lora / backend.optim are dispatched
    to their own builders — they must still be recognised as parent keys."""
    from decent_vla.configs import load_run_config

    p = _cfg_file(tmp_path, {
        "name": "t",
        "backend": {
            "name": "mock",
            "lora": {"r": 8},
            "optim": {"lr": 1e-4},
        },
        "fl": {
            "rounds": 2,
            "local_steps": 1,
            "strategy": {"name": "fedavg"},
            "nvflare": {"env": "simulator"},
        },
        "eval": {"benchmark": {"name": "mock"}},
    })
    cfg = load_run_config(p)
    assert cfg.backend.lora is not None
    assert cfg.fl.strategy.name == "fedavg"
    assert cfg.fl.nvflare.env == "simulator"
    assert cfg.eval.benchmark.name == "mock"
