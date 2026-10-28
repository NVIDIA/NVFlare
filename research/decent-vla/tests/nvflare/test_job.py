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


def test_build_job_ships_its_files_and_no_host_paths(run_cfg):
    """An exported job names only files inside itself: the packaged run config
    and client script. No submitter path reaches the server or the clients."""
    from decent_vla.runtime.nvflare.controller import DecentVLAController
    from decent_vla.runtime.nvflare.job import CLIENT_SCRIPT, RUN_CONFIG, JobConfigFiles, build_job, staged_job

    cfg, p = run_cfg()
    with staged_job(str(p)) as staged:
        stage, config_files = staged
        assert sorted(f.name for f in stage.iterdir()) == sorted([CLIENT_SCRIPT, RUN_CONFIG])
        assert config_files == [RUN_CONFIG]
        job = build_job(cfg, staged)

    assert job.min_clients == 2
    assert job.app_packages == ["decent_vla"]  # no partial copy of the package in custom/
    ctrl, server_files = job.server_components
    assert isinstance(ctrl, DecentVLAController)
    assert (ctrl.config_path, ctrl.output_dir, ctrl.workdir) == (RUN_CONFIG, None, None)
    client_files, runner = job.client_components
    # The run config goes in config/ (JobConfigFiles), the script in custom/ (ScriptRunner).
    assert isinstance(server_files, JobConfigFiles) and isinstance(client_files, JobConfigFiles)
    assert server_files.paths == client_files.paths == [str(stage / RUN_CONFIG)]
    assert runner.kwargs["script"] == CLIENT_SCRIPT
    assert runner.kwargs["script_args"] == f"--config {RUN_CONFIG}"
    assert runner.kwargs["launch_external_process"] is False
    assert runner.kwargs["params_transfer_type"] == "FULL"


def test_build_job_for_the_simulator_carries_this_hosts_paths(run_cfg):
    from decent_vla.runtime.nvflare.job import RUN_CONFIG, build_job, staged_job

    cfg, p = run_cfg()
    with staged_job(str(p)) as staged:
        job = build_job(cfg, staged, workdir="/submit/cwd", output_dir="/submit/out")
    ctrl, _ = job.server_components
    assert (ctrl.output_dir, ctrl.workdir) == ("/submit/out", "/submit/cwd")
    _, runner = job.client_components
    assert runner.kwargs["script_args"] == f"--config {RUN_CONFIG} --workdir /submit/cwd"


def test_job_min_sites_follow_the_quorum(run_cfg):
    """FLARE aborts a job once fewer than its min_sites are alive; that limit
    is the round loop's quorum, not every client."""
    from decent_vla.runtime.nvflare.job import build_job, staged_job

    cfg, p = run_cfg()
    cfg.fl.min_clients = 1
    with staged_job(str(p)) as staged:
        job = build_job(cfg, staged)
    assert job.min_clients == 1


def test_staged_job_leaves_out_submit_side_settings(run_cfg, tmp_path):
    from decent_vla.runtime.nvflare.job import RUN_CONFIG, staged_job

    cfg, p = run_cfg({"env": "export", "job_dir": str(tmp_path / "jobs"), "workspace": str(tmp_path / "ws")})
    with staged_job(str(p)) as (stage, _):
        packaged = yaml.safe_load((stage / RUN_CONFIG).read_text())
    assert packaged["fl"]["nvflare"] == {"env": "export"}


def test_staged_job_packages_the_files_the_config_names(tmp_path, monkeypatch):
    import os

    from decent_vla.runtime.nvflare.job import RUN_CONFIG, staged_job

    (tmp_path / "examples").mkdir()
    pipeline = tmp_path / "examples" / "pipeline.yaml"
    pipeline.write_text("policy: {type: smolvla}\n")
    cfg = tmp_path / "examples" / "run.yaml"
    cfg.write_text(yaml.safe_dump({"name": "t", "backend": {
        "name": "lerobot", "extra": {"train_config_path": "examples/pipeline.yaml", "dtype": "bfloat16"}}}))
    monkeypatch.chdir(tmp_path)  # relative paths in a run config resolve against the cwd

    with staged_job(str(cfg)) as (stage, config_files):
        assert config_files == [RUN_CONFIG, "pipeline.yaml"]
        assert os.getcwd() == str(stage)
        assert (stage / "pipeline.yaml").read_text() == pipeline.read_text()
        packaged = yaml.safe_load((stage / RUN_CONFIG).read_text())
        assert packaged["backend"]["extra"] == {"train_config_path": "pipeline.yaml", "dtype": "bfloat16"}
    assert os.getcwd() == str(tmp_path)
    assert not stage.exists()


def test_staged_job_rejects_a_missing_packaged_file(tmp_path, monkeypatch):
    from decent_vla.runtime.nvflare.job import staged_job

    cfg = tmp_path / "run.yaml"
    cfg.write_text(yaml.safe_dump({"backend": {"name": "lerobot", "extra": {"train_config_path": "nope.yaml"}}}))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="train_config_path"):
        with staged_job(str(cfg)):
            pass


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
