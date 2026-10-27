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

"""Offline contract, launch capture and host-preflight tests; not hardware proof."""

import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from nvflare.lighter.cc_provision import kata_runtime_profile as runtime
from nvflare.lighter.cc_provision import workload_launch_profile as launch

ROOT = Path(__file__).resolve().parents[5] / "examples/devops/coco"
CONTEXT = {
    "privileged": False,
    "allowPrivilegeEscalation": False,
    "runAsNonRoot": True,
    "runAsUser": 65532,
    "runAsGroup": 65532,
    "readOnlyRootFilesystem": False,
    "capabilities": {"drop": ["ALL"]},
    "seccompProfile": {"type": "RuntimeDefault"},
}


def profile(runtime_class):
    target = runtime.runtime_target(runtime_class)
    return {
        "schema": runtime.PROFILE_SCHEMA_V4,
        "cpu_tee": target["cpu_tee"],
        "gpu": target["gpu"],
        "workload_security_context": CONTEXT,
        "profile_id": "reviewed",
        "runtime_class": runtime_class,
        "kata_version": "3.29.0",
        "guest_token_api": runtime.CAPABILITY,
        "kata_deploy_image": "registry/runtime@sha256:" + "a" * 64,
        "kata_config_sha256": "b" * 64,
        "launch_inputs_sha256": "c" * 64,
        "vm_defaults": {"vcpus": 1, "memory_mib": 8192},
        "pod_constraints": {
            "container_count": 1,
            "gpu_resource": "nvidia.com/pgpu" if target["gpu_count"] else None,
            "gpu_count": target["gpu_count"],
            "cpu_memory_resources": "omitted",
            "host_namespaces": False,
            "allowed_annotations": ["io.katacontainers.config.hypervisor.cc_init_data"],
        },
    }


def load(tmp_path, value):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(value))
    return launch.load_profile(path, hashlib.sha256(path.read_bytes()).hexdigest(), value["runtime_class"], "3.29.0")


def pod(runtime_class):
    resources = {"limits": {"nvidia.com/pgpu": 1}} if runtime.runtime_target(runtime_class)["gpu_count"] else {}
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": "test"},
        "spec": {
            "runtimeClassName": runtime_class,
            "containers": [{"name": "workload", "image": "image", "securityContext": CONTEXT, "resources": resources}],
        },
    }


@pytest.mark.parametrize("runtime_class", runtime.RUNTIME_TARGETS)
def test_v4_target_contract_and_pod(tmp_path, runtime_class):
    value = load(tmp_path, profile(runtime_class))
    launch.validate_pod(value, pod(runtime_class))
    config = runtime.runtime_target(runtime_class)["config_name"]
    assert config == "configuration-" + runtime_class.removeprefix("kata-") + ".toml"


@pytest.mark.parametrize("runtime_class", runtime.RUNTIME_TARGETS)
@pytest.mark.parametrize("field", ["cpu_tee", "gpu"])
def test_cross_target_rejected(tmp_path, runtime_class, field):
    value = profile(runtime_class)
    value[field] = "tdx" if field == "cpu_tee" and value[field] == "snp" else "snp" if field == "cpu_tee" else "invalid"
    with pytest.raises(ValueError, match="target"):
        load(tmp_path, value)


@pytest.mark.parametrize("runtime_class", runtime.RUNTIME_TARGETS)
def test_v3_cannot_expand_authority(tmp_path, runtime_class):
    value = profile(runtime_class)
    value["schema"] = "coco-approved-workload-launch/v3"
    del value["cpu_tee"], value["gpu"]
    if runtime_class == "kata-qemu-nvidia-gpu-snp":
        load(tmp_path, value)
    else:
        with pytest.raises(ValueError, match="legacy"):
            load(tmp_path, value)


@pytest.mark.parametrize("runtime_class", runtime.RUNTIME_TARGETS)
@pytest.mark.parametrize(
    "resources", [{"limits": {"cpu": "1"}}, {"requests": {"memory": "1Gi"}}, {"limits": {"nvidia.com/pgpu": 2}}]
)
def test_unapproved_resources_rejected(tmp_path, runtime_class, resources):
    value = load(tmp_path, profile(runtime_class))
    workload = pod(runtime_class)
    workload["spec"]["containers"][0]["resources"] = resources
    with pytest.raises(ValueError):
        launch.validate_pod(value, workload)


@pytest.mark.parametrize("runtime_class", runtime.RUNTIME_TARGETS)
def test_target_toml_confidential_guard(runtime_class):
    target = runtime.runtime_target(runtime_class)
    config = {"confidential_guest": True, "sev_snp_guest": target["cpu_tee"] == "snp", "machine_type": "q35"}
    assert runtime.require_confidential_config(config, runtime_class) == target
    config["confidential_guest"] = False
    with pytest.raises(ValueError, match="confidential_guest"):
        runtime.require_confidential_config(config, runtime_class)


def capture_helper():
    pytest.importorskip("tomllib", reason="host launch capture requires system Python 3.11+")
    return runpy.run_path(str(ROOT / "trusted_system/capture-running-launch.py"))["confidential_guest"]


@pytest.mark.parametrize("gpu", [False, True])
def test_tdx_capture_parses_json_and_excludes_workload_binding(gpu):
    helper = capture_helper()
    name = "kata-qemu-nvidia-gpu-tdx" if gpu else "kata-qemu-tdx"
    guest = {
        "qom-type": "tdx-guest",
        "id": "tdx",
        "debug": False,
        "mrconfigid": "one",
        "quote-generation-socket": {"type": "vsock"},
    }
    argv = ["qemu", "-machine", "q35,confidential-guest-support=tdx", "-object", json.dumps(guest)]
    target, first = helper(argv, name)
    assert target["gpu_count"] == int(gpu)
    guest["mrconfigid"] = "two"
    argv[-1] = json.dumps(guest)
    assert helper(argv, name)[1] == first
    assert first == {"qom-type": "tdx-guest", "id": "tdx", "debug": False}


@pytest.mark.parametrize("debug", [True, 1, 0, "on", "true", None])
def test_tdx_debug_rejected(debug):
    helper = capture_helper()
    guest = {"qom-type": "tdx-guest", "id": "tdx", "debug": debug}
    with pytest.raises(ValueError, match="debug"):
        helper(
            ["qemu", "-machine", "q35,confidential-guest-support=tdx", "-object", json.dumps(guest)], "kata-qemu-tdx"
        )


def test_snp_capture_keeps_legacy_encoding():
    helper = capture_helper()
    argv = [
        "qemu",
        "-machine",
        "q35,confidential-guest-support=snp",
        "-object",
        "sev-snp-guest,id=snp,policy=196608,host-data=workload",
    ]
    target, guest = helper(argv, "kata-qemu-snp")
    assert target["cpu_tee"] == "snp"
    assert guest == {"qom-type": "sev-snp-guest", "id": "snp", "policy": "196608"}
    with pytest.raises(ValueError, match="RuntimeClass"):
        helper(argv, "kata-qemu-tdx")


def test_duplicate_machine_guest_selector_rejected():
    helper = capture_helper()
    argv = [
        "qemu",
        "-machine",
        "q35,confidential-guest-support=tdx,confidential-guest-support=other",
        "-object",
        json.dumps({"qom-type": "tdx-guest", "id": "tdx"}),
    ]
    with pytest.raises(ValueError, match="duplicate QEMU machine"):
        helper(argv, "kata-qemu-tdx")


def test_tdx_target_import_does_not_require_tomllib():
    script = "import sys; sys.modules['tomllib'] = None; from nvflare.lighter.cc_provision import workload_launch_profile; from nvflare.lighter.cc_provision.kata_runtime_profile import runtime_target; assert runtime_target('kata-qemu-tdx')['cpu_tee'] == 'tdx'"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def captured_inputs(runtime_class):
    pytest.importorskip("tomllib", reason="host profile export requires Python 3.11+")
    capture = runpy.run_path(str(ROOT / "trusted_system/lib/platform-capture.py"))
    target = runtime.runtime_target(runtime_class)
    paths = {name: "/installed/ready-runtime/" + name for name in ("path", "firmware", "kernel", "initrd")}
    config_path = "/installed/ready-runtime/configuration.toml"
    qemu = {
        **paths,
        "kernel_params": runtime.REQUIRED,
        "confidential_guest": True,
        "sev_snp_guest": target["cpu_tee"] == "snp",
        "machine_type": "q35",
        "default_vcpus": 1,
        "default_memory": 8192,
    }
    snapshot = {
        "config": {"hypervisor": {"qemu": qemu}},
        "installed_config": config_path,
        "runtime_version": "3.29.0",
        "files": {config_path: "a" * 64},
    }
    snapshot["files"].update({path: hashlib.sha256(name.encode()).hexdigest() for name, path in paths.items()})
    source = capture["prepare_profile"](snapshot, runtime_class)
    actual = {
        "cpu_tee": target["cpu_tee"],
        "gpu": target["gpu"],
        "launch_inputs": {"kernel_command_line": runtime.REQUIRED, "smp": "1", "memory": "8192M"},
        "pod_resources": [source["pod_resources"]],
        "artifacts": {"kata_config": {"sha256": source["kata_config_sha256"]}},
    }
    actual["artifacts"].update(
        {
            "qemu_executable" if name == "path" else name: {"sha256": value["sha256"]}
            for name, value in source["artifacts"].items()
        }
    )
    actual["launch_inputs_sha256"] = hashlib.sha256(
        json.dumps(actual["launch_inputs"], sort_keys=True).encode()
    ).hexdigest()
    return capture, source, actual


@pytest.mark.parametrize("runtime_class", runtime.RUNTIME_TARGETS)
def test_export_preserves_target_and_installed_launch_hashes(tmp_path, runtime_class):
    capture, source, actual = captured_inputs(runtime_class)
    contract = load(tmp_path, capture["build_launch_profile"](source, actual, "approved"))
    target = runtime.runtime_target(runtime_class)
    assert contract["schema"] == runtime.PROFILE_SCHEMA_V4
    assert contract["cpu_tee"] == target["cpu_tee"] and contract["gpu"] == target["gpu"]
    assert contract["launch_inputs_sha256"] == actual["launch_inputs_sha256"]
    assert contract["kata_config_sha256"] == source["kata_config_sha256"]
    assert contract["workload_security_context"] == CONTEXT
    assert "kata_deploy_image" not in contract
    actual["cpu_tee"] = "invalid"
    with pytest.raises(ValueError, match="target differs"):
        capture["build_launch_profile"](source, actual, "approved")


@pytest.mark.parametrize("runtime_class", runtime.RUNTIME_TARGETS)
@pytest.mark.parametrize("artifact", ["kata_config", "qemu_executable", "firmware", "kernel", "initrd"])
def test_export_rejects_changed_installed_or_boot_artifacts(runtime_class, artifact):
    capture, source, actual = captured_inputs(runtime_class)
    actual["artifacts"][artifact]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="differs from the installed snapshot"):
        capture["build_launch_profile"](source, actual, "approved")


@pytest.mark.parametrize("runtime_class", runtime.RUNTIME_TARGETS)
@pytest.mark.parametrize("field,value", [("smp", "2"), ("memory", "4096M")])
def test_export_rejects_changed_vm_defaults_even_with_consistent_fingerprint(runtime_class, field, value):
    capture, source, actual = captured_inputs(runtime_class)
    actual["launch_inputs"][field] = value
    actual["launch_inputs_sha256"] = hashlib.sha256(
        json.dumps(actual["launch_inputs"], sort_keys=True).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="VM sizing differs"):
        capture["build_launch_profile"](source, actual, "approved")


def test_export_rejects_stale_launch_fingerprint():
    capture, source, actual = captured_inputs("kata-qemu-snp")
    actual["launch_inputs"]["memory"] = "4096M"
    with pytest.raises(ValueError, match="fingerprint is inconsistent"):
        capture["build_launch_profile"](source, actual, "approved")


@pytest.mark.parametrize("runtime_class", runtime.RUNTIME_TARGETS)
def test_export_retains_supported_runtime_version_guard(runtime_class):
    capture, source, actual = captured_inputs(runtime_class)
    with pytest.raises(ValueError, match="supported Kata/genpolicy"):
        capture["build_launch_profile"](source, actual, "approved", runtime_version="3.30.0")


@pytest.mark.parametrize("runtime_class", runtime.RUNTIME_TARGETS)
@pytest.mark.parametrize("pin", [None, ""])
def test_received_launch_profile_needs_no_external_checksum(tmp_path, runtime_class, pin):
    path = tmp_path / "received.json"
    value = profile(runtime_class)
    path.write_text(json.dumps(value))
    assert launch.load_profile(path, pin, runtime_class, "3.29.0") == value


@pytest.mark.parametrize("pin", ["bad", "0" * 64, False, 0])
def test_launch_profile_rejects_invalid_supplied_checksum(tmp_path, pin):
    path = tmp_path / "received.json"
    path.write_text(json.dumps(profile("kata-qemu-snp")))
    with pytest.raises(ValueError, match="SHA-256"):
        launch.load_profile(path, pin, "kata-qemu-snp", "3.29.0")


@pytest.mark.parametrize("pin_mode", ["omitted", "empty", "provided"])
def test_launch_profile_cli_keeps_supplied_and_omitted_checksum_forms(tmp_path, monkeypatch, pin_mode):
    path, output = tmp_path / "received.json", tmp_path / "snapshot.json"
    raw = json.dumps(profile("kata-qemu-snp")).encode()
    path.write_bytes(raw)
    pin_args = [] if pin_mode == "omitted" else ["" if pin_mode == "empty" else hashlib.sha256(raw).hexdigest()]
    monkeypatch.setattr(
        sys, "argv", ["profile-check", str(path), *pin_args, "kata-qemu-snp", "3.29.0", "--snapshot", str(output)]
    )
    launch.main()
    assert output.read_bytes() == raw


@pytest.mark.parametrize("change_before_validation", [False, True])
def test_launch_profile_snapshot_uses_only_validated_bytes(tmp_path, monkeypatch, change_before_validation):
    path, output = tmp_path / "received.json", tmp_path / "snapshot.json"
    original = profile("kata-qemu-snp")
    raw = json.dumps(original).encode()
    path.write_bytes(raw)
    changed = {**original, "profile_id": "changed"}
    load_profile = launch.load_profile

    def replace_file(*args):
        if change_before_validation:
            path.write_text(json.dumps(changed))
        value = load_profile(*args)
        path.write_text(json.dumps(changed))
        return value

    monkeypatch.setattr(launch, "load_profile", replace_file)
    monkeypatch.setattr(sys, "argv", ["profile-check", str(path), "kata-qemu-snp", "3.29.0", "--snapshot", str(output)])
    if change_before_validation:
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            launch.main()
        assert not output.exists()
    else:
        launch.main()
        assert output.read_bytes() == raw
