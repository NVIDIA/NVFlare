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

"""FedJob wiring + CLI-entry dispatch, against the fake nvflare namespace."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def run_cfg(tmp_path):
    def make(nvflare_block=None):
        cfg = {
            "name": "t",
            "backend": {"name": "mock"},
            "data": {"partitioning": "iid", "num_clients": 2, "batch_size": 4},
            "fl": {
                "strategy": {"name": "fedavg"},
                "rounds": 2,
                "local_steps": 2,
                "execution": "nvflare",
                "output_dir": str(tmp_path / "out"),
                **({"nvflare": nvflare_block} if nvflare_block else {}),
            },
        }
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(cfg))
        from decent_vla.configs import load_run_config

        return load_run_config(p), p

    return make


def test_build_job_wires_controller_and_script_runner(run_cfg):
    from decent_vla.configs import load_run_config  # noqa: F401
    from decent_vla.runtime.nvflare import client_script
    from decent_vla.runtime.nvflare.controller import DecentVLAController
    from decent_vla.runtime.nvflare.job import build_job

    cfg, p = run_cfg()
    job = build_job(cfg, str(p))

    assert job.min_clients == 2
    (ctrl,) = job.server_components
    assert isinstance(ctrl, DecentVLAController)
    assert ctrl.config_path == str(p.resolve())
    assert ctrl.output_dir == str(Path(cfg.fl.output_dir).resolve())
    assert ctrl.workdir == str(Path.cwd())
    (runner,) = job.client_components
    assert runner.kwargs["script"] == str(Path(client_script.__file__).resolve())
    assert runner.kwargs["script_args"] == f"--config {p.resolve()} --workdir {Path.cwd()}"
    assert runner.kwargs["launch_external_process"] is False
    assert runner.kwargs["params_transfer_type"] == "FULL"


def test_run_simulator_defaults(run_cfg):
    from decent_vla.runtime.nvflare.job import run_simulator

    cfg, p = run_cfg()
    run_simulator(cfg, str(p))
    from tests.nvflare.conftest import FakeFedJob

    (job,) = FakeFedJob.instances
    (call,) = job.sim_calls
    assert call["clients"] == ["c0", "c1"]
    assert call["threads"] == 2  # = num_clients: warm client state stays in memory
    assert call["gpu"] is None
    assert call["workspace"] == str((Path(cfg.fl.output_dir) / "nvflare_workspace").resolve())


def test_run_simulator_honours_nvflare_block(run_cfg, tmp_path):
    from decent_vla.runtime.nvflare.job import run_simulator

    cfg, p = run_cfg({"threads": 1, "gpu": "0,1", "workspace": str(tmp_path / "ws")})
    run_simulator(cfg, str(p))
    from tests.nvflare.conftest import FakeFedJob

    (job,) = FakeFedJob.instances
    (call,) = job.sim_calls
    assert call["threads"] == 1
    assert call["gpu"] == "0,1"
    assert call["workspace"] == str((tmp_path / "ws").resolve())


def test_export_job_writes_job_dir(run_cfg, tmp_path):
    from decent_vla.runtime.nvflare.job import export_job

    cfg, p = run_cfg({"job_dir": str(tmp_path / "jobs")})
    out = export_job(cfg, str(p))
    from tests.nvflare.conftest import FakeFedJob

    (job,) = FakeFedJob.instances
    assert job.exported == [str((tmp_path / "jobs").resolve())]
    assert out == (tmp_path / "jobs").resolve()


def _write_final_blob(cfg):
    from pathlib import Path

    ckpt_dir = Path(cfg.fl.output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "server_round_final.pt").write_bytes(b"blob")


def test_entry_simulator_env_runs_simulator(run_cfg, monkeypatch):
    from decent_vla.runtime.nvflare import entry, job

    cfg, p = run_cfg()
    calls = []

    def fake_sim(c, cp):
        calls.append(cp)
        _write_final_blob(c)  # what a successful controller run leaves behind

    monkeypatch.setattr(job, "run_simulator", fake_sim)
    entry.run_from_cli(cfg, str(p))
    assert calls == [str(p)]


def test_entry_simulator_fails_without_fresh_final_checkpoint(run_cfg, monkeypatch):
    """The simulator subprocess exits 0 even when the server workflow died;
    the entry must turn a missing/stale success artifact into a hard failure
    so scripted runs never silently 'succeed'."""
    import os
    import time

    from decent_vla.runtime.nvflare import entry, job

    cfg, p = run_cfg()
    monkeypatch.setattr(job, "run_simulator", lambda c, cp: None)  # crashed workflow
    with pytest.raises(SystemExit, match="server workflow failed"):
        entry.run_from_cli(cfg, str(p))

    # A STALE final blob (from an earlier finished run) must not count either.
    _write_final_blob(cfg)
    final = Path(cfg.fl.output_dir) / "checkpoints" / "server_round_final.pt"
    old = time.time() - 3600
    os.utime(final, (old, old))
    with pytest.raises(SystemExit, match="server workflow failed"):
        entry.run_from_cli(cfg, str(p))


def test_entry_export_env_exits_with_instructions(run_cfg, tmp_path, capsys):
    from decent_vla.runtime.nvflare import entry

    cfg, p = run_cfg({"env": "export", "job_dir": str(tmp_path / "jobs")})
    with pytest.raises(SystemExit) as ei:
        entry.run_from_cli(cfg, str(p))
    assert ei.value.code == 0
    out = capsys.readouterr().out
    assert "job folder written" in out
    assert "nvflare job submit" in out


def test_entry_rejects_unknown_env(run_cfg):
    from decent_vla.runtime.nvflare import entry

    cfg, p = run_cfg({"env": "poc"})
    with pytest.raises(SystemExit, match="Unknown fl.nvflare.env"):
        entry.run_from_cli(cfg, str(p))


def test_entry_rejects_resume_flag(run_cfg):
    from decent_vla.runtime.nvflare import entry

    cfg, p = run_cfg()
    with pytest.raises(SystemExit, match="auto-resumes"):
        entry.run_from_cli(cfg, str(p), resume="some.pt")
