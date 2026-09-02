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

import sys
from pathlib import Path

import pytest

from tests.unit_test.examples.fedcore_test_utils import fedcore_import_context


def test_full_mode_default_defers_gpu_allocation_to_qwen_example():
    with fedcore_import_context():
        import run_demo

        args = run_demo.define_parser().parse_args(["--mode", "full"])
        command = run_demo._build_predictor_command(
            args,
            Path("/repo/examples/advanced/qwen3-vl"),
            Path("/tmp/fedcore data"),
            Path("/tmp/fedcore workspace"),
        )

    assert args.gpu is None
    assert run_demo._first_gpu(args.gpu) == "0"
    assert "--gpu" not in command


def test_full_mode_forwards_explicit_gpu_allocation():
    with fedcore_import_context():
        import run_demo

        args = run_demo.define_parser().parse_args(["--mode", "full", "--gpu", "[3],[4],[5]"])
        command = run_demo._build_predictor_command(args, Path("/repo/qwen"), Path("/tmp/data"), Path("/tmp/work"))

    assert command[command.index("--gpu") + 1] == "[3],[4],[5]"


def test_run_directories_must_be_fresh(tmp_path):
    with fedcore_import_context():
        import run_demo

        output_dir = tmp_path / "output"
        workspace = output_dir / "workspace"
        run_demo._prepare_run_directories(output_dir, workspace)
        assert output_dir.is_dir()
        assert workspace.is_dir()
        with pytest.raises(FileExistsError, match="fresh path"):
            run_demo._prepare_run_directories(output_dir, workspace)


def test_existing_workspace_does_not_poison_fresh_output_path(tmp_path):
    with fedcore_import_context():
        import run_demo

        output_dir = tmp_path / "output"
        workspace = tmp_path / "existing-workspace"
        workspace.mkdir()
        with pytest.raises(FileExistsError, match="workspace"):
            run_demo._prepare_run_directories(output_dir, workspace)

    assert not output_dir.exists()


def test_run_command_ignores_process_workspace_redirect(tmp_path, monkeypatch):
    with fedcore_import_context():
        import run_demo

        captured = {}

        def fake_run(command, cwd, check, env):
            captured.update({"command": command, "cwd": cwd, "check": check, "env": env})

        monkeypatch.setenv("NVFLARE_SIMULATOR_WORKSPACE_ROOT", "/unexpected/workspace")
        monkeypatch.setattr(run_demo.subprocess, "run", fake_run)
        run_demo._run([sys.executable, "-c", "pass"], cwd=tmp_path)

    assert "NVFLARE_SIMULATOR_WORKSPACE_ROOT" not in captured["env"]


def test_run_command_preserves_symlinked_virtual_environment(tmp_path, monkeypatch):
    with fedcore_import_context():
        import run_demo

        captured = {}
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        venv_python = venv_bin / "python"
        venv_python.symlink_to(Path(sys.executable))

        def fake_run(command, cwd, check, env):
            captured.update({"command": command, "cwd": cwd, "check": check, "env": env})

        monkeypatch.setattr(run_demo.sys, "executable", str(venv_python))
        monkeypatch.setattr(run_demo.subprocess, "run", fake_run)
        run_demo._run(["python3", "-c", "pass"], cwd=tmp_path)

    assert Path(captured["env"]["PATH"].split(run_demo.os.pathsep)[0]) == venv_bin


def test_invalid_data_configuration_does_not_reserve_output(tmp_path, monkeypatch):
    with fedcore_import_context():
        import run_demo

        output_dir = tmp_path / "output"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_demo.py",
                "--output-dir",
                str(output_dir),
                "--train-samples-per-site",
                "8",
            ],
        )
        with pytest.raises(ValueError, match="cannot represent both correct and incorrect OCR"):
            run_demo.main()

    assert not output_dir.exists()


def test_negative_seed_is_rejected_before_output_is_reserved(tmp_path, monkeypatch):
    with fedcore_import_context():
        import run_demo

        output_dir = tmp_path / "output"
        monkeypatch.setattr(
            sys,
            "argv",
            ["run_demo.py", "--output-dir", str(output_dir), "--seed", "-1"],
        )
        with pytest.raises(SystemExit):
            run_demo.main()

    assert not output_dir.exists()


def test_uninformative_scenario_rejects_proxy_override(monkeypatch):
    with fedcore_import_context():
        import run_demo

        monkeypatch.setattr(
            sys,
            "argv",
            ["run_demo.py", "--scenario", "uninformative", "--proxy-strength", "0.8"],
        )
        with pytest.raises(ValueError, match="fixed at 0.5"):
            run_demo.main()
