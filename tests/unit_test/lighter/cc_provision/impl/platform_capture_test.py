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

"""In-process CoCo policy and validation contracts."""

import json
import os
import subprocess
from unittest.mock import patch

import pytest

from tests.coco_test_utils.platform_capture import TRUSTED, capture_fixture, workflow

pytest.importorskip("tomllib", reason="CoCo deployment helpers require Python 3.11+")


@pytest.fixture(autouse=True)
def preserve_process_umask():
    # capture() is a command entry point; direct unit calls must not affect later tests.
    previous = os.umask(0o077)
    os.umask(previous)
    try:
        yield
    finally:
        os.umask(previous)


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
