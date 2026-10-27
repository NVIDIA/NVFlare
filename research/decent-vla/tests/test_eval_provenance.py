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

"""Audit-hardening: eval artifacts identify what was evaluated, at write time.

The 2026-08-14 forensic audit of the pi0.5 r100 results found the numbers
correct — but proving that depended entirely on the Slurm job logs surviving
(the `[eval] ckpt=` echo lines). The artifacts themselves carried no
checkpoint identity, no action horizon, no config identity.

These tests pin the fix in both eval paths:

  * decent-vla path (`runtime/eval_runner.run_eval`): `results.json` gains a
    `provenance` block — checkpoint path + size + mtime_ns, requested and
    EFFECTIVE `n_action_steps` (the horizon has silently diverged from the
    config twice; recording the requested value alone would re-create that
    trap inside the artifact), `eval_config_hash`, and the config path.

  * lerobot-harness path (`lerobot-eval`):
    `eval_info.json` is written by upstream lerobot, so the identity lands in
    a `provenance.json` sidecar in the same output dir, stamped by
    `python -m decent_vla.core.provenance` before the rollouts start.

The per-suite cache key in `eval_runner` is deliberately NOT extended — new
key fields would invalidate every finished suite on the clusters and re-run
hours of LIBERO rollouts (see `_EVAL_CACHE_IGNORED`).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from decent_vla.cli.main import main

# ---------------------------------------------------------------------------
# decent-vla eval path: results.json `provenance` block
# ---------------------------------------------------------------------------


def _mock_run_cfg(tmp_path: Path, out: Path) -> Path:
    cfg = {
        "name": "prov_block_run",
        "device": "cpu",
        "backend": {"name": "mock"},
        "data": {"name": "mock", "batch_size": 4, "num_clients": 2, "partitioning": "iid"},
        "fl": {"strategy": {"name": "fedavg"}, "num_rounds": 1, "local_steps": 2,
               "save_every_n_rounds": 1, "mode": "local", "output_dir": str(out)},
        "eval": {"benchmark": {"name": "mock", "suites": ["default"]},
                 "n_episodes_per_task": 2, "seed": 0},
    }
    p = tmp_path / "prov.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_results_json_carries_provenance_block(tmp_path):
    """results.json must identify the evaluated checkpoint + horizon + config
    without any job log: path, stat identity, eval_config_hash, config path."""
    out = tmp_path / "out"
    cfg_path = _mock_run_cfg(tmp_path, out)
    assert main(["run", "--config", str(cfg_path)]) == 0

    data = json.loads((out / "eval" / "results.json").read_text())
    assert data["schema_version"] == "1.2"
    prov = data["provenance"]

    ckpt = Path(data["checkpoint"])
    st = ckpt.stat()
    assert prov["checkpoint"] == data["checkpoint"]
    assert prov["checkpoint_size"] == st.st_size
    assert prov["checkpoint_mtime_ns"] == st.st_mtime_ns

    # Same identity the per-suite cache keys on — one artifact, one story.
    suite_prov = json.loads(
        (out / "eval" / "suites" / "default.json").read_text()
    )["provenance"]
    assert prov["eval_config_hash"] == suite_prov["eval_config_hash"]

    assert prov["config_path"] == str(cfg_path)
    # Mock config requests no horizon and MockPolicy exposes none.
    assert prov["n_action_steps_requested"] is None
    assert prov["n_action_steps_effective"] is None


def test_results_json_provenance_records_effective_horizon(tmp_path):
    """When the config pins a horizon, the artifact must record the value the
    policy ACTUALLY ran at, read back from the policy after it was applied."""
    out = tmp_path / "out"
    cfg = yaml.safe_load(_mock_run_cfg(tmp_path, out).read_text())
    cfg["eval"]["n_action_steps"] = 10
    cfg_path = tmp_path / "prov_nact.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    # MockPolicy has no config.n_action_steps, so a pinned horizon must raise
    # (apply_action_horizon's contract) — the artifact never gets written with
    # an un-honored horizon.
    with pytest.raises(ValueError, match="n_action_steps"):
        main(["run", "--config", str(cfg_path)])


def test_effective_action_horizon_probes_wrapper_then_policy():
    from decent_vla.runtime.eval_runner import effective_action_horizon

    bare = SimpleNamespace(config=SimpleNamespace(n_action_steps=10))
    assert effective_action_horizon(bare) == 10

    wrapper = SimpleNamespace(_policy=bare)  # benchmark-facing adapter wrapper
    assert effective_action_horizon(wrapper) == 10

    assert effective_action_horizon(object()) is None
    no_int = SimpleNamespace(config=SimpleNamespace(n_action_steps=None))
    assert effective_action_horizon(no_int) is None


def test_schema_12_reads_11_files_without_provenance(tmp_path):
    """v1.1 results (pre-provenance) must stay readable; provenance -> None."""
    from decent_vla.core.results import SCHEMA_VERSION, read_results

    assert SCHEMA_VERSION == "1.2"
    legacy = {"schema_version": "1.1", "run_id": "r", "backend": "mock:mock",
              "checkpoint": None, "benchmark": "mock", "config_hash": "sha256:x",
              "git_commit": None, "started_at": "t", "finished_at": "t",
              "wall_time_s": 1.0, "device": {"type": "cpu", "name": "cpu", "count": 1},
              "n_episodes_per_task": 1, "seed": 0, "suites": {}, "overall": {},
              "extra": {}, "aggregates": {}, "partition_manifest_hash": None}
    p = tmp_path / "r.json"
    p.write_text(json.dumps(legacy))
    assert read_results(p).provenance is None


# ---------------------------------------------------------------------------
# lerobot-harness path: `python -m decent_vla.core.provenance` sidecar stamper
# ---------------------------------------------------------------------------


def test_file_identity_stat_only(tmp_path):
    from decent_vla.core.provenance import file_identity

    f = tmp_path / "ckpt.pt"
    f.write_bytes(b"weights" * 100)
    ident = file_identity(f)
    st = f.stat()
    assert ident == {"path": str(f), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def test_file_identity_with_sha256(tmp_path):
    from decent_vla.core.provenance import file_identity

    payload = b"server_round_0100 weights"
    f = tmp_path / "ckpt.pt"
    f.write_bytes(payload)
    ident = file_identity(f, sha256=True)
    assert ident["sha256"] == hashlib.sha256(payload).hexdigest()


def test_stamper_cli_writes_sidecar(tmp_path):
    from decent_vla.core.provenance import main as stamp_main

    ckpt = tmp_path / "server_round_0100.pt"
    ckpt.write_bytes(b"\x00" * 64)
    cfg = tmp_path / "arm.yaml"
    cfg.write_text("name: arm\n")
    out = tmp_path / "eval_logs" / "arm" / "server_round_0100"

    rc = stamp_main([
        "--output-dir", str(out),
        "--ckpt", str(ckpt),
        "--config", str(cfg),
        "--n-action-steps", "10",
        "--extra", "suites=libero_spatial,libero_object",
        "--extra", "slurm_job_id=5960632",
    ])
    assert rc == 0

    prov = json.loads((out / "provenance.json").read_text())
    assert prov["schema"] == "decent-vla/eval-provenance@1"
    assert prov["checkpoint"]["path"] == str(ckpt)
    assert prov["checkpoint"]["size"] == ckpt.stat().st_size
    assert prov["checkpoint"]["mtime_ns"] == ckpt.stat().st_mtime_ns
    # The config file is small: always content-hashed.
    assert prov["config"]["sha256"] == hashlib.sha256(cfg.read_bytes()).hexdigest()
    assert prov["config"]["path"] == str(cfg)
    assert prov["n_action_steps"] == 10
    assert prov["extra"]["suites"] == "libero_spatial,libero_object"
    assert prov["extra"]["slurm_job_id"] == "5960632"
    assert "written_at" in prov


def test_stamper_cli_sha256_flag_hashes_checkpoint(tmp_path):
    from decent_vla.core.provenance import main as stamp_main

    payload = b"\x01\x02" * 512
    ckpt = tmp_path / "server_round_0050.pt"
    ckpt.write_bytes(payload)
    out = tmp_path / "out"
    assert stamp_main(["--output-dir", str(out), "--ckpt", str(ckpt), "--sha256"]) == 0
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["checkpoint"]["sha256"] == hashlib.sha256(payload).hexdigest()


def test_stamper_cli_records_absolute_paths(tmp_path, monkeypatch):
    """Slurm scripts pass $CKPT as typed — usually relative to the submit dir.
    A relative path in the sidecar is ambiguous the moment eval_logs/ moves;
    the stamp must record the resolved absolute path."""
    from decent_vla.core.provenance import main as stamp_main

    monkeypatch.chdir(tmp_path)
    (tmp_path / "checkpoints").mkdir()
    ckpt = tmp_path / "checkpoints" / "server_round_0100.pt"
    ckpt.write_bytes(b"\x00" * 8)
    cfg = tmp_path / "arm.yaml"
    cfg.write_text("name: arm\n")

    assert stamp_main(["--output-dir", "out",
                       "--ckpt", "checkpoints/server_round_0100.pt",
                       "--config", "arm.yaml"]) == 0
    prov = json.loads((tmp_path / "out" / "provenance.json").read_text())
    assert prov["checkpoint"]["path"] == str(ckpt)
    assert prov["config"]["path"] == str(cfg)


def test_stamper_cli_refuses_missing_checkpoint(tmp_path):
    """A provenance stamp for a path that doesn't exist would be a fabricated
    audit record — fail loudly instead."""
    from decent_vla.core.provenance import main as stamp_main

    with pytest.raises(SystemExit, match="checkpoint"):
        stamp_main(["--output-dir", str(tmp_path / "o"),
                    "--ckpt", str(tmp_path / "ghost.pt")])
