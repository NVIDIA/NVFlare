#!/usr/bin/env python3
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

"""Ready-cluster capture/export orchestration, without running hardware collectors."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("tomllib", reason="CoCo deployment helpers require Python 3.11+")

ROOT = Path(__file__).resolve().parents[5]
TRUSTED = ROOT / "examples/devops/coco/trusted_system"
spec = importlib.util.spec_from_file_location("platform_capture", TRUSTED / "lib/platform-capture.py")
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)
helper_spec = importlib.util.spec_from_file_location("capture_installed", TRUSTED / "lib/installed-platform.py")
installed = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(installed)


@pytest.fixture(autouse=True)
def preserve_process_umask():
    # capture() is a command entry point; direct unit calls must not affect later tests.
    previous = os.umask(0o077)
    os.umask(previous)
    try:
        yield
    finally:
        os.umask(previous)


def capture_fixture(tmp_path, monkeypatch):
    output = tmp_path / "approved-platform"
    runtime = "kata-qemu-snp"
    env = {
        "RUNTIME_CLASS": runtime,
        "SNP_MIN_REPORTED_TCB_BOOTLOADER": "1",
        "SNP_MIN_REPORTED_TCB_TEE": "2",
        "SNP_MIN_REPORTED_TCB_SNP": "3",
        "SNP_MIN_REPORTED_TCB_MICROCODE": "4",
    }
    values = {
        "snp_launch_measurement": "a" * 96,
        "snp_min_reported_tcb_bootloader": 1,
        "snp_min_reported_tcb_tee": 2,
        "snp_min_reported_tcb_snp": 3,
        "snp_min_reported_tcb_microcode": 4,
    }
    artifacts = {key: "/existing/" + key for key in ("path", "firmware", "kernel", "initrd")}
    qemu = dict(
        artifacts,
        kernel_params="agent.guest_components_rest_api=all",
        confidential_guest=True,
        sev_snp_guest=True,
        default_vcpus=2,
        default_memory=4096,
    )
    snapshot = {
        "config": {"hypervisor": {"qemu": qemu}},
        "runtime_class": runtime,
        "installed_config": "/existing/config.toml",
        "files": {path: key + "-hash" for key, path in artifacts.items()},
        "node": {"name": "selected-worker", "uid": "node-uid"},
        "runtime_version": "3.29.0",
    }
    snapshot["files"][snapshot["installed_config"]] = "config-hash"
    calls = []

    def inspect(directory, supplied):
        workflow.write_json(directory / "installed-platform.json", snapshot)
        return snapshot

    def verify_snapshot(directory, supplied):
        calls.append("snapshot")
        return snapshot

    def collect(supplied, directory, received, profile):
        evidence = directory / "evidence"
        evidence.mkdir()
        inputs = {"kernel_command_line": qemu["kernel_params"], "smp": "2", "memory": "4096M"}
        actual = {
            "runtime_class": runtime,
            "cpu_tee": "amd_sev_snp",
            "gpu": "none",
            "node_name": "selected-worker",
            "node_uid": "node-uid",
            "platform_snapshot_sha256": installed.digest(directory / "installed-platform.json"),
            "launch_inputs": inputs,
            "launch_inputs_sha256": hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
            "pod_resources": [{}],
            "artifacts": {
                ("qemu_executable" if key == "path" else key): {"sha256": key + "-hash"} for key in artifacts
            },
        }
        actual["artifacts"]["kata_config"] = {"sha256": "config-hash"}
        workflow.write_json(evidence / "actual-launch.json", actual)
        workflow.write_json(evidence / "pod-result.json", {"spec": {"nodeName": "selected-worker"}})
        (evidence / "request-data.bin").write_bytes(b"a" * 64)
        calls.append("collect")
        return values, {"measurement": values["snp_launch_measurement"]}

    api = dict(
        vars(installed), read_config=lambda path, keys: env, inspect_platform=inspect, verify_snapshot=verify_snapshot
    )
    monkeypatch.setattr(workflow, "helpers", lambda: api)
    monkeypatch.setattr(workflow, "require_capture_tools", lambda: None)
    monkeypatch.setattr(workflow, "collect", collect)
    monkeypatch.setattr(workflow, "verify_evidence", lambda *args: (values, {}))
    config = tmp_path / "platform.env"
    config.touch()
    return output, config, env, values, calls


def test_single_capture_verifies_and_exports_both_handoffs(tmp_path, monkeypatch):
    output, config, env, values, calls = capture_fixture(tmp_path, monkeypatch)
    with patch.object(workflow, "verify_evidence", return_value=(values, {})) as verify:
        workflow.capture(config, output)
    assert calls.count("collect") == 1
    verify.assert_called_once()
    assert (output / "capture-summary.json").exists()
    assert json.loads((output / "platform-reference-values.json").read_text()) == values
    profile = json.loads((output / "approved-workload-launch-profile.json").read_text())
    assert "kata_deploy_image" not in profile
    assert profile["kata_version"] == "3.29.0"
    assert profile["workload_security_context"]["readOnlyRootFilesystem"] is False
    with pytest.raises(ValueError, match="new output directory"):
        workflow.capture(config, output)


@pytest.mark.parametrize("changed", ["workload-profile.json", "capture-inputs.json", "evidence/actual-launch.json"])
def test_changed_capture_cannot_be_exported(tmp_path, monkeypatch, changed):
    output, config, _, _, _ = capture_fixture(tmp_path, monkeypatch)
    export = workflow._export_verified_capture

    def changed_before_export(directory):
        source = directory / changed
        source.write_text(source.read_text() + "\n")
        return export(directory)

    monkeypatch.setattr(workflow, "_export_verified_capture", changed_before_export)
    with patch.object(workflow, "verify_evidence") as verify:
        with pytest.raises(ValueError, match="changed before export"):
            workflow.capture(config, output)
    verify.assert_not_called()
    assert not (output / "platform-reference-values.json").exists()


def test_capture_reverifies_raw_hardware_evidence_before_export(tmp_path, monkeypatch):
    output, config, _, _, _ = capture_fixture(tmp_path, monkeypatch)
    with patch.object(workflow, "verify_evidence", side_effect=ValueError("CPU report signature rejected")):
        with pytest.raises(ValueError, match="signature rejected"):
            workflow.capture(config, output)
    assert not (output / "approved-workload-launch-profile.json").exists()


def test_reverified_references_must_match_initial_capture(tmp_path, monkeypatch):
    output, config, _, values, _ = capture_fixture(tmp_path, monkeypatch)
    changed = dict(values, snp_launch_measurement="b" * 96)
    with patch.object(workflow, "verify_evidence", return_value=(changed, {})):
        with pytest.raises(ValueError, match="captured values"):
            workflow.capture(config, output)
    assert not (output / "platform-reference-values.json").exists()


def test_missing_independent_tcb_floor_fails_before_capture(tmp_path, monkeypatch):
    output, config, env, _, calls = capture_fixture(tmp_path, monkeypatch)
    env["SNP_MIN_REPORTED_TCB_TEE"] = ""
    with pytest.raises(ValueError, match="independently approved"):
        workflow.capture(config, output)
    assert "collect" not in calls
    assert not output.exists()


def test_capture_cli_has_no_separate_approval_step(tmp_path):
    result = subprocess.run(
        [sys.executable, str(TRUSTED / "capture-platform.py"), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "output_directory" in result.stdout
    assert "--approve" not in result.stdout
    assert not (TRUSTED / "approve-platform.py").exists()


@pytest.mark.parametrize(
    "runtime", ["kata-qemu-snp", "kata-qemu-nvidia-gpu-snp", "kata-qemu-tdx", "kata-qemu-nvidia-gpu-tdx"]
)
def test_diagnostic_collector_uses_selected_worker_and_target_resources(tmp_path, runtime):
    ca = tmp_path / "registry-ca.crt"
    ca.write_text('public certificate with "quotes"\nand multiple lines\n')
    env = {"RUNTIME_CLASS": runtime, "COLLECTOR_IMAGE": "registry.example.com/collector@sha256:" + "a" * 64}
    snapshot = {"registry_ca": str(ca), "node": {"name": "selected-worker"}}
    profile = {"pod_resources": workflow.fixed_workload(runtime)["spec"]["containers"][0]["resources"]}
    pod = workflow.make_collector(env, snapshot, profile, tmp_path, "diagnostic-namespace")
    assert pod["spec"]["nodeName"] == "selected-worker"
    assert pod["spec"]["runtimeClassName"] == runtime
    assert pod["spec"]["containers"][0]["image"] == env["COLLECTOR_IMAGE"]
    assert pod["spec"]["containers"][0]["resources"] == profile["pod_resources"]
    import tomllib

    initdata = tomllib.loads((tmp_path / "initdata.toml").read_text())
    cdh = tomllib.loads(initdata["data"]["cdh.toml"])
    assert cdh["image"]["extra_root_certificates"] == [ca.read_text()]


@pytest.mark.parametrize("runtime", ["kata-qemu-tdx", "kata-qemu-nvidia-gpu-tdx"])
@pytest.mark.parametrize("name", ["TDX", "tdx-Profile", "_tdx", "a" * 65, "tdx+profile"])
def test_invalid_tdx_directory_is_rejected_before_capture_side_effects(tmp_path, monkeypatch, runtime, name):
    _, config, env, _, calls = capture_fixture(tmp_path, monkeypatch)
    env["RUNTIME_CLASS"] = runtime
    output = tmp_path / name
    with patch.object(workflow, "require_capture_tools") as probe, patch.object(workflow, "collect") as collect:
        with pytest.raises(ValueError, match="TDX output directory name"):
            workflow.capture(config, output)
    probe.assert_not_called()
    collect.assert_not_called()
    assert not calls
    assert not output.exists()


@pytest.mark.parametrize("runtime", ["kata-qemu-tdx", "kata-qemu-nvidia-gpu-tdx"])
@pytest.mark.parametrize("name", ["tdx", "0", "a" * 64, "kata-3.29.0_tdx-ready"])
def test_valid_tdx_directory_reaches_preflight_with_schema_compatible_id(tmp_path, monkeypatch, runtime, name):
    _, config, env, _, _ = capture_fixture(tmp_path, monkeypatch)
    env["RUNTIME_CLASS"] = runtime
    # Stop at the preflight boundary: name acceptance must not need a real TDX worker.
    with patch.object(workflow, "require_capture_tools", side_effect=ValueError("preflight boundary")) as probe:
        with pytest.raises(ValueError, match="preflight boundary"):
            workflow.capture(config, tmp_path / name)
    probe.assert_called_once()
    fields = ("mr_td", "rtmr_0", "rtmr_1", "rtmr_2", "rtmr_3", "tdvfkernel", "tdvfkernelparams")
    values = {
        "schema": "coco-platform-reference-values/v2",
        "tee": "tdx",
        "profiles": [{"id": name, **{field: "a" * 96 for field in fields}, "xfam": "a" * 16}],
    }
    assert workflow.runpy.run_path(str(TRUSTED / "lib/platform-reference-schema.py"))["validate_values"](values)
    assert not (tmp_path / name).exists()


def test_snp_directory_keeps_existing_uppercase_names(tmp_path, monkeypatch):
    _, config, _, values, _ = capture_fixture(tmp_path, monkeypatch)
    output = tmp_path / "SNP_Profile"
    workflow.capture(config, output)
    assert json.loads((output / "platform-reference-values.json").read_text()) == values


@pytest.mark.parametrize("failure", ["missing", "sudo-denied", "sudo-unavailable", "timeout"])
def test_privileged_crictl_failure_stops_before_inspection_cache_or_namespace(tmp_path, monkeypatch, failure):
    probe = workflow.require_capture_tools
    output, config, _, _, calls = capture_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(workflow, "require_capture_tools", probe)
    if failure == "sudo-unavailable":
        error = FileNotFoundError("sudo")
    elif failure == "timeout":
        error = subprocess.TimeoutExpired(["sudo", "python3"], 30)
    else:
        error = subprocess.CalledProcessError(1, ["sudo", "python3"])
    with (
        patch.object(workflow.subprocess, "run", side_effect=error) as command,
        patch.object(workflow, "collect") as collect,
    ):
        with pytest.raises(ValueError, match=r"Capture requires Python 3\.11\+"):
            workflow.capture(config, output)
    command.assert_called_once()
    assert command.call_args.args[0][:3] == ["sudo", "python3", "-c"]
    assert command.call_args.kwargs["timeout"] == 30
    collect.assert_not_called()
    assert not calls
    assert not output.exists()


@pytest.mark.parametrize("available_to_sudo", [False, True])
def test_crictl_probe_checks_the_effective_sudo_python_path(tmp_path, monkeypatch, available_to_sudo):
    caller = tmp_path / "caller-bin"
    privileged = tmp_path / "sudo-bin"
    caller.mkdir()
    privileged.mkdir()
    available = privileged if available_to_sudo else caller
    binary = available / "crictl"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o700)
    monkeypatch.setenv("PATH", str(caller))
    run = subprocess.run

    def effective_sudo(command, **kwargs):
        assert command[:3] == ["sudo", "python3", "-c"]
        # Execute the actual probe program with a distinct effective PATH, as
        # sudo's secure_path does, without requiring sudo privileges in tests.
        return run([sys.executable, *command[2:]], env={**os.environ, "PATH": str(privileged)}, **kwargs)

    with patch.object(workflow.subprocess, "run", side_effect=effective_sudo):
        if available_to_sudo:
            workflow.require_capture_tools()
        else:
            with pytest.raises(ValueError, match=r"Capture requires Python 3\.11\+"):
                workflow.require_capture_tools()


@pytest.mark.parametrize("failure", ["old-python", "helper-import"])
def test_privileged_interpreter_failures_stop_capture_before_side_effects(tmp_path, monkeypatch, failure):
    probe = workflow.require_capture_tools
    output, config, _, _, calls = capture_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(workflow, "require_capture_tools", probe)
    helper_root = tmp_path / "probe-helper"
    (helper_root / "lib").mkdir(parents=True)
    (helper_root / "capture-running-launch.py").write_text("import missing_capture_dependency\n")
    runtime = workflow.runtime_api()
    monkeypatch.setattr(workflow, "runtime_api", lambda: runtime)
    monkeypatch.setattr(workflow, "HERE", helper_root / "lib")
    run = subprocess.run
    diagnostics = []

    def effective_sudo(command, **kwargs):
        program = command[3]
        if failure == "old-python":
            program = "import sys; sys.version_info = (3, 10)\n" + program
        try:
            return run([sys.executable, "-c", program, *command[4:]], **kwargs)
        except subprocess.CalledProcessError as error:
            diagnostics.append(error.stderr)
            raise

    with (
        patch.object(workflow.subprocess, "run", side_effect=effective_sudo),
        patch.object(workflow, "collect") as collect,
    ):
        with pytest.raises(ValueError, match=r"Capture requires Python 3\.11\+"):
            workflow.capture(config, output)
    expected = "Python 3.11+ is required" if failure == "old-python" else "missing_capture_dependency"
    assert len(diagnostics) == 1 and expected in diagnostics[0]
    collect.assert_not_called()
    assert not calls
    assert not output.exists()
