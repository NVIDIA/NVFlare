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


"""Shared capture API and factories for unit and executable contracts."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("tomllib", reason="CoCo deployment helpers require Python 3.11+")

ROOT = Path(__file__).resolve().parents[2]

TRUSTED = ROOT / "examples/devops/coco/trusted_system"

spec = importlib.util.spec_from_file_location("platform_capture", TRUSTED / "lib/platform-capture.py")

workflow = importlib.util.module_from_spec(spec)

spec.loader.exec_module(workflow)

helper_spec = importlib.util.spec_from_file_location("capture_installed", TRUSTED / "lib/installed-platform.py")

installed = importlib.util.module_from_spec(helper_spec)

helper_spec.loader.exec_module(installed)


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
            "cpu_tee": "snp",
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
