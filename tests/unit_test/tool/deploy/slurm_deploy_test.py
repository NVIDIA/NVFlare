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

import json
import os
import shlex
import shutil
import subprocess

import pytest

from nvflare.app_opt.job_launcher.slurm import ClientSlurmJobLauncher
from nvflare.tool.deploy import slurm_deploy
from tests.unit_test.tool.deploy.deploy_commands_test import (
    _add_server_storage,
    _component,
    _make_client_kit,
    _make_server_kit,
    _run_prepare,
)


def _slurm_executables(tmp_path):
    result = {}
    for name in ("sbatch", "squeue", "sacct", "scancel"):
        path = tmp_path / "bin" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        path.chmod(0o700)
        result[name] = str(path)
    return result


def _slurm_config(tmp_path, **launcher_overrides):
    launcher = {
        "sandbox": "none",
        "python_path": "/usr/bin/python3",
        "executables": _slurm_executables(tmp_path),
    }
    launcher.update(launcher_overrides)
    return {
        "runtime": "slurm",
        "job_launcher": launcher,
    }


def test_prepare_slurm_preserves_file_transport_comm_config(tmp_path, capsys):
    kit = _make_client_kit(tmp_path)
    file_internal = {
        "scheme": "shared-file",
        "resources": {
            "root_dir": "/lustre/proj/cellnet",
            "poll_interval": 0.05,
            "lease_timeout": 30,
        },
    }
    (kit / "local" / "comm_config.json").write_text(
        json.dumps({"backbone": {"connect_generation": 1}, "internal": file_internal}), encoding="utf-8"
    )
    output = tmp_path / "site-1-slurm"

    _run_prepare(kit, output, _slurm_config(tmp_path, internal_port=9210))
    capsys.readouterr()

    comm_config = json.loads((output / "local" / "comm_config.json").read_text())
    assert comm_config["backbone"] == {"connect_generation": 1}
    resources = comm_config["internal"]["resources"]
    assert comm_config["internal"]["scheme"] == "shared-file"
    assert resources["root_dir"] == "/lustre/proj/cellnet"
    assert resources["connection_security"] == "clear"
    assert "host" not in resources and "port" not in resources


def test_prepare_slurm_rejects_relative_file_transport_root_dir(tmp_path, capsys):
    kit = _make_client_kit(tmp_path)
    (kit / "local" / "comm_config.json").write_text(
        json.dumps({"internal": {"scheme": "shared-file", "resources": {"root_dir": "relative/cellnet"}}}),
        encoding="utf-8",
    )
    output = tmp_path / "site-1-slurm"

    with pytest.raises(SystemExit):
        _run_prepare(kit, output, _slurm_config(tmp_path))
    assert "absolute internal.resources.root_dir" in capsys.readouterr().err


def test_render_template_is_one_pass(tmp_path, monkeypatch):
    template = tmp_path / "test.sh"
    template.write_text("@@NVFLARE_FIRST@@\n@@NVFLARE_SECOND@@\n", encoding="utf-8")
    output = tmp_path / "rendered.sh"
    monkeypatch.setattr(slurm_deploy, "TEMPLATES_DIR", tmp_path)

    slurm_deploy._render_template(
        "test.sh",
        output,
        {
            "@@NVFLARE_FIRST@@": "first @@NVFLARE_LITERAL@@",
            "@@NVFLARE_SECOND@@": "done",
        },
        mode=0o700,
    )

    assert output.read_text(encoding="utf-8") == "first @@NVFLARE_LITERAL@@\ndone\n"
    assert output.stat().st_mode & 0o777 == 0o700


def test_render_template_rejects_unresolved_placeholder(tmp_path, monkeypatch):
    (tmp_path / "test.sh").write_text("@@NVFLARE_EXPECTED@@\n", encoding="utf-8")
    output = tmp_path / "rendered.sh"
    monkeypatch.setattr(slurm_deploy, "TEMPLATES_DIR", tmp_path)

    with pytest.raises(SystemExit):
        slurm_deploy._render_template(
            "test.sh",
            output,
            {"@@NVFLARE_OTHER@@": "value"},
            mode=0o700,
        )

    assert not output.exists()


def test_prepare_slurm_generates_runtime_artifacts(tmp_path, capsys):
    kit = _make_client_kit(tmp_path)
    output = tmp_path / "site-1-slurm"
    config = _slurm_config(tmp_path, internal_port=9210, forward_env=["HTTP_PROXY"])

    _run_prepare(kit, output, config)
    capsys.readouterr()

    resources = json.loads((output / "local" / "resources.json.default").read_text())
    launcher = _component(resources, "slurm_launcher")
    assert launcher["path"].endswith("ClientSlurmJobLauncher")
    assert launcher["args"] == {
        "workspace_path": str(output),
        "sandbox": "none",
        "image": None,
        "internal_port": 9210,
        "sbatch_directives": {},
        "setup": "",
        "python_path": "/usr/bin/python3",
        "executables": {
            **{name: str((tmp_path / "bin" / name).resolve()) for name in ("sbatch", "squeue", "sacct", "scancel")},
            "apptainer": None,
            "srun": None,
        },
        "forward_env": ["HTTP_PROXY"],
        "parent_host": None,
        "poll_interval": 10,
        "pending_timeout": 600,
    }
    ClientSlurmJobLauncher(**launcher["args"])
    assert _component(resources, "resource_manager")["path"].endswith("PassthroughResourceManager")
    assert not (output / "local" / "resources.json").exists()

    comm_config = json.loads((output / "local" / "comm_config.json").read_text())
    assert comm_config["internal"] == {
        "scheme": "tcp",
        "resources": {"host": "0.0.0.0", "port": 9210, "connection_security": "clear"},
    }
    assert (output / "startup" / "sub_start.sh").exists()
    start_script = output / "startup" / "start_slurm.sh"
    start_text = start_script.read_text()
    assert "@@NVFLARE_" not in start_text
    assert 'exec "$NVFL_WORKSPACE/startup/sub_start.sh" --once' in start_text
    assert start_script.stat().st_mode & 0o777 == 0o700
    assert not (output / "startup" / "start.sh").exists()
    assert not (output / "startup" / "parent.slurm").exists()
    assert output.stat().st_mode & 0o777 == 0o700
    assert (output / "startup").is_dir() and not (output / "startup").is_symlink()
    assert (output / "local").is_dir() and not (output / "local").is_symlink()
    assert not (output / "kit").exists()
    assert not (output / "local" / "slurm_stage.json").exists()

    runtime_marker = output / "runtime-marker"
    runtime_marker.write_text("old workspace", encoding="utf-8")
    _run_prepare(kit, output, config)
    capsys.readouterr()
    assert not runtime_marker.exists()


def test_prepare_slurm_does_not_require_scheduler_commands_on_prepare_host(tmp_path, capsys, monkeypatch):
    kit = _make_client_kit(tmp_path)
    output = tmp_path / "site-1-slurm"
    config = _slurm_config(tmp_path, executables={})
    monkeypatch.setenv("PATH", "")

    _run_prepare(kit, output, config)
    capsys.readouterr()

    resources = json.loads((output / "local" / "resources.json.default").read_text())
    executables = _component(resources, "slurm_launcher")["args"]["executables"]
    assert executables == {
        "sbatch": None,
        "squeue": None,
        "sacct": None,
        "scancel": None,
        "apptainer": None,
        "srun": None,
    }


def test_prepare_slurm_preserves_explicit_stable_executable_paths(tmp_path, capsys):
    kit = _make_client_kit(tmp_path)
    output = tmp_path / "site-1-slurm"
    target_dir = tmp_path / "slurm-23.02" / "bin"
    stable_dir = tmp_path / "slurm-current" / "bin"
    target_dir.mkdir(parents=True)
    stable_dir.mkdir(parents=True)
    configured = {}
    for name in ("sbatch", "squeue", "sacct", "scancel"):
        target = target_dir / name
        target.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        target.chmod(0o700)
        stable = stable_dir / name
        stable.symlink_to(target)
        configured[name] = str(stable)
    config = _slurm_config(tmp_path, executables=configured)

    _run_prepare(kit, output, config)
    capsys.readouterr()

    resources = json.loads((output / "local" / "resources.json.default").read_text())
    persisted = _component(resources, "slurm_launcher")["args"]["executables"]
    assert {name: persisted[name] for name in configured} == configured


def test_prepare_slurm_start_script_uses_output_as_workspace(tmp_path, capsys):
    kit = _make_client_kit(tmp_path)
    marker = tmp_path / "parent-started"
    sub_start = kit / "startup" / "sub_start.sh"
    sub_start.write_text(
        "#!/usr/bin/env bash\n"
        'DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'SOURCE_WORKSPACE="$(cd "$DIR/.." && pwd)"\n'
        f'printf "%s|%s|%s" "$NVFL_WORKSPACE" "$SOURCE_WORKSPACE" "$1" > "{marker}"\n',
        encoding="utf-8",
    )
    sub_start.chmod(0o700)
    output = tmp_path / "site-1-slurm"
    config = _slurm_config(tmp_path)
    _run_prepare(kit, output, config)
    prepare_output = capsys.readouterr().out
    assert "deploy slurm stage" not in prepare_output
    assert (output / "startup" / "sub_start.sh").is_file()

    subprocess.run([str(output / "startup" / "start_slurm.sh")], check=True)
    assert marker.read_text(encoding="utf-8") == f"{output}|{output}|--once"


def test_prepare_slurm_rejects_configurable_connection_security(tmp_path, capsys):
    kit = _make_client_kit(tmp_path)
    config = _slurm_config(tmp_path, connection_security="clear")

    with pytest.raises(SystemExit):
        _run_prepare(kit, tmp_path / "site-1-slurm", config)

    assert "Unknown keys" in capsys.readouterr().err


def test_prepare_slurm_client_parent_scripts_use_validated_values(tmp_path, capsys):
    kit = _make_client_kit(tmp_path)
    output = tmp_path / "site-1-slurm"
    config = _slurm_config(tmp_path)
    config["parent"] = {
        "sbatch_directives": {"partition": "batch", "time": "1-00:00:00"},
        "environment_setup": "source /opt/nvflare/activate",
    }

    _run_prepare(kit, output, config)
    prepare_output = capsys.readouterr().out

    parent = (output / "startup" / "parent.slurm").read_text()
    assert "@@NVFLARE_" not in parent
    assert "#SBATCH --partition=batch" in parent
    assert "#SBATCH --time=1-00:00:00" in parent
    assert "source /opt/nvflare/activate" in parent
    assert f"WORKSPACE_ROOT={shlex.quote(str(output))}" in parent
    assert 'exec "$WORKSPACE_ROOT/startup/start_slurm.sh"' in parent
    assert (output / "startup" / "parent.slurm").stat().st_mode & 0o777 == 0o700

    submit = output / "startup" / "submit_parent.sh"
    assert not submit.exists()
    assert "submit_command:" in prepare_output
    assert "submit_command: sbatch --parsable" in prepare_output
    assert str(tmp_path / "bin" / "sbatch") not in prepare_output
    assert f"--output={output}/parent-slurm-%j.out" in prepare_output
    subprocess.run(["bash", "-n", str(output / "startup" / "start_slurm.sh")], check=True)
    subprocess.run(["bash", "-n", str(output / "startup" / "parent.slurm")], check=True)

    config["parent"]["sbatch_directives"]["partition"] = "batch --nodes=2"
    with pytest.raises(SystemExit):
        _run_prepare(kit, tmp_path / "unsafe-parent", config)
    assert "must not contain whitespace" in capsys.readouterr().err

    config["parent"]["sbatch_directives"]["partition"] = "batch"
    config["job_launcher"]["sbatch_directives"] = {"partition": "batch --uid=123"}
    with pytest.raises(SystemExit):
        _run_prepare(kit, tmp_path / "unsafe-worker", config)
    assert "must not contain whitespace" in capsys.readouterr().err


def test_parent_script_uses_workspace_path_after_sbatch_relocation(tmp_path, capsys):
    kit = _make_client_kit(tmp_path)
    output = tmp_path / "site-1-slurm"
    config = _slurm_config(tmp_path)
    config["parent"] = {}

    _run_prepare(kit, output, config)
    capsys.readouterr()

    marker = tmp_path / "parent-started"
    start_script = output / "startup" / "start_slurm.sh"
    start_script.write_text('#!/usr/bin/env bash\nprintf started > "$NVFLARE_TEST_MARKER"\n', encoding="utf-8")
    start_script.chmod(0o700)

    spool_dir = tmp_path / "slurm-spool" / "job00001"
    spool_dir.mkdir(parents=True)
    relocated_script = spool_dir / "slurm_script"
    shutil.copy2(output / "startup" / "parent.slurm", relocated_script)

    subprocess.run([str(relocated_script)], env={**os.environ, "NVFLARE_TEST_MARKER": str(marker)}, check=True)
    assert marker.read_text(encoding="utf-8") == "started"


def test_prepare_slurm_normalizes_optional_null_shell_blocks(tmp_path, capsys):
    kit = _make_client_kit(tmp_path)
    output = tmp_path / "site-1-slurm"
    config = _slurm_config(tmp_path, setup=None)
    config["parent"] = {"environment_setup": None}

    _run_prepare(kit, output, config)
    capsys.readouterr()

    resources = json.loads((output / "local" / "resources.json.default").read_text())
    assert _component(resources, "slurm_launcher")["args"]["setup"] == ""
    assert "None" not in (output / "startup" / "parent.slurm").read_text()


@pytest.mark.parametrize(
    "name",
    [
        "_nvfl_secret",
        "NVFL_SRUN",
        "SLURM_EXPORT_ENV",
        "NVFLARE_SLURM_HELPER_NNODES",
    ],
)
def test_prepare_slurm_rejects_renderer_owned_forward_env(tmp_path, capsys, name):
    kit = _make_client_kit(tmp_path)
    config = _slurm_config(tmp_path, forward_env=[name])

    with pytest.raises(SystemExit):
        _run_prepare(kit, tmp_path / "site-1-slurm", config)

    assert "launcher-owned" in capsys.readouterr().err


def test_prepare_slurm_server_relocates_storage_and_rejects_parent(tmp_path, capsys):
    kit = _make_server_kit(tmp_path)
    _add_server_storage(kit / "local" / "resources.json.default")
    output = tmp_path / "server-slurm"
    config = _slurm_config(tmp_path)

    _run_prepare(kit, output, config)
    capsys.readouterr()
    resources = json.loads((output / "local" / "resources.json.default").read_text())
    assert _component(resources, "slurm_launcher")["path"].endswith("ServerSlurmJobLauncher")
    assert resources["snapshot_persistor"]["args"]["storage"]["args"]["root_dir"] == (f"{output}/snapshot-storage")
    assert _component(resources, "job_manager")["args"]["uri_root"] == f"{output}/jobs-storage"

    config["parent"] = {}
    with pytest.raises(SystemExit):
        _run_prepare(kit, tmp_path / "server-parent", config)
    assert "do not support parent" in capsys.readouterr().err

    config.pop("parent")
    resources_path = kit / "local" / "resources.json.default"
    malformed = json.loads(resources_path.read_text())
    malformed["snapshot_persistor"] = {"args": {}}
    resources_path.write_text(json.dumps(malformed), encoding="utf-8")
    _run_prepare(kit, tmp_path / "server-malformed-storage", config)
    assert "could not relocate snapshot storage" in capsys.readouterr().out


def test_prepare_slurm_rejects_legacy_workspace_path_and_unsafe_output(tmp_path, capsys):
    kit = _make_client_kit(tmp_path)
    legacy = _slurm_config(tmp_path)
    legacy["workspace_path"] = str(tmp_path / "shared" / "site-1")
    with pytest.raises(SystemExit):
        _run_prepare(kit, tmp_path / "site-1-slurm", legacy)
    assert "Unknown keys" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        _run_prepare(kit, tmp_path / "unsafe:path-list-slurm", _slurm_config(tmp_path))
    assert "must not contain the path-list separator ':'" in capsys.readouterr().err
