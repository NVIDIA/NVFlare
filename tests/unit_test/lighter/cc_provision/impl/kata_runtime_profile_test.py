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

"""Read-only runtime capability and launch-contract checks."""

import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from nvflare.lighter.cc_provision import kata_runtime_profile, workload_launch_profile

ROOT = Path(__file__).resolve().parents[5] / "examples/devops/coco"
API = vars(kata_runtime_profile)
ADMIN = vars(workload_launch_profile)
CONFIG = (
    "[hypervisor.qemu]\n"
    'kernel_params = "pci=one pci=two quiet agent.guest_components_rest_api=all"\n'
    "confidential_guest = true\nsev_snp_guest = true\n"
    "default_vcpus = 1\ndefault_memory = 8192\n"
)

SECURITY_CONTEXT = {
    "privileged": False,
    "allowPrivilegeEscalation": False,
    "runAsNonRoot": True,
    "runAsUser": 65532,
    "runAsGroup": 65532,
    "readOnlyRootFilesystem": False,
    "capabilities": {"drop": ["ALL"]},
    "seccompProfile": {"type": "RuntimeDefault"},
}


@pytest.mark.parametrize(
    "runtime,tee,gpu,count",
    [
        ("kata-qemu-snp", "snp", "none", 0),
        ("kata-qemu-nvidia-gpu-snp", "snp", "nvidia", 1),
        ("kata-qemu-tdx", "tdx", "none", 0),
        ("kata-qemu-nvidia-gpu-tdx", "tdx", "nvidia", 1),
    ],
)
def test_supported_runtime_mapping(runtime, tee, gpu, count):
    target = API["runtime_target"](runtime)
    assert (target["cpu_tee"], target["gpu"], target["gpu_count"]) == (tee, gpu, count)
    assert target["config_name"] == "configuration-" + runtime.removeprefix("kata-") + ".toml"


@pytest.mark.parametrize("runtime", [None, True, 1, "", "kata-qemu-tdx-gpu", "kata-qemu-snp-extra"])
def test_unknown_runtime_is_not_inferred_from_its_name(runtime):
    with pytest.raises(ValueError, match="Unsupported confidential RuntimeClass"):
        API["runtime_target"](runtime)


def test_token_capability_accepts_existing_setting_without_changing_other_parameters():
    cmdline = "pci=one pci=two quiet " + API["REQUIRED"]
    API["require_token_api"](cmdline)
    assert cmdline == "pci=one pci=two quiet agent.guest_components_rest_api=all"


@pytest.mark.parametrize(
    "value",
    [
        None,
        False,
        [],
        "",
        "quiet",
        "agent.guest_components_rest_api",
        "agent.guest_components_rest_api=resource",
        "agent.guest_components_rest_api=attestation",
        "agent.guest_components_rest_api=unknown",
        "agent.guest_components_rest_api=all agent.guest_components_rest_api=all",
        "agent.guest_components_rest_api=all agent.guest_components_rest_api=resource",
    ],
)
def test_missing_disabled_duplicate_launch_option(value):
    with pytest.raises(ValueError):
        API["require_token_api"](value)


@pytest.mark.parametrize("runtime", API["RUNTIME_TARGETS"])
def test_confidential_guard_only_reads_existing_settings(runtime):
    tee = API["runtime_target"](runtime)["cpu_tee"]
    qemu = {"confidential_guest": True, "sev_snp_guest": tee == "snp", "machine_type": "q35"}
    before = json.dumps(qemu, sort_keys=True)
    assert API["require_confidential_config"](qemu, runtime)["cpu_tee"] == tee
    assert json.dumps(qemu, sort_keys=True) == before
    qemu["confidential_guest"] = False
    with pytest.raises(ValueError, match="confidential_guest"):
        API["require_confidential_config"](qemu, runtime)


@pytest.mark.parametrize(
    "path",
    [
        ROOT / "shared/kata-runtime-profile.py",
        ROOT / "trusted_system/lib/kata-runtime-profile.py",
    ],
)
def test_checkout_adapter_exports_current_read_only_guards(path):
    source = runpy.run_path(str(path))
    assert source["runtime_target"] is API["runtime_target"]
    source["require_token_api"](API["REQUIRED"])


@pytest.mark.parametrize(
    "path",
    [
        ROOT / "shared/kata-runtime-profile.py",
        ROOT / "trusted_system/lib/kata-runtime-profile.py",
        Path(kata_runtime_profile.__file__),
    ],
)
@pytest.mark.parametrize(
    "arguments", [[], ["target", "kata-qemu-snp", "--field", "cpu_tee"], ["enable", "unused.toml"]]
)
def test_removed_runtime_cli_rejects_direct_execution(path, arguments):
    result = subprocess.run([sys.executable, str(path), *arguments], capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert not result.stdout
    assert "Kata runtime configuration CLI has been removed" in result.stderr
    assert "agent.guest_components_rest_api=all" in result.stderr


def test_admin_rejects_old_or_missing_capability(tmp_path):
    path = tmp_path / "contract.json"
    contract = {
        "schema": "coco-approved-workload-launch/v3",
        "workload_security_context": SECURITY_CONTEXT,
        "profile_id": "reviewed",
        "runtime_class": "kata-qemu-nvidia-gpu-snp",
        "kata_version": "3.29.0",
        "guest_token_api": API["CAPABILITY"],
        "kata_deploy_image": "registry/runtime@sha256:" + "a" * 64,
        "kata_config_sha256": "b" * 64,
        "launch_inputs_sha256": "c" * 64,
        "vm_defaults": {"vcpus": 1, "memory_mib": 8192},
        "pod_constraints": {
            "container_count": 1,
            "gpu_resource": "nvidia.com/pgpu",
            "gpu_count": 1,
            "cpu_memory_resources": "omitted",
            "host_namespaces": False,
            "allowed_annotations": ["io.katacontainers.config.hypervisor.cc_init_data"],
        },
    }

    def load(value):
        path.write_text(json.dumps(value))
        return ADMIN["load_profile"](
            path, hashlib.sha256(path.read_bytes()).hexdigest(), contract["runtime_class"], "3.29.0"
        )

    assert load(contract)["guest_token_api"] == API["CAPABILITY"]
    with pytest.raises(ValueError):
        load({**contract, "schema": "coco-approved-workload-launch/v1"})
    with pytest.raises(ValueError):
        load({**contract, "schema": "coco-approved-workload-launch/v2"})
    with pytest.raises(ValueError):
        load({k: v for k, v in contract.items() if k != "workload_security_context"})
    with pytest.raises(ValueError):
        load({k: v for k, v in contract.items() if k != "guest_token_api"})
    with pytest.raises(ValueError):
        load({**contract, "guest_token_api": "resource-only"})


def test_ready_launch_contract_and_measured_option(tmp_path):
    tomllib = pytest.importorskip("tomllib", reason="installed TOML capture requires Python 3.11+")
    capture = runpy.run_path(str(ROOT / "trusted_system/lib/platform-capture.py"))
    runtime = "kata-qemu-nvidia-gpu-snp"
    resources = {"limits": {"nvidia.com/pgpu": "1"}, "requests": {"nvidia.com/pgpu": "1"}}
    installed = CONFIG
    profile = {
        "runtime_class": runtime,
        "cpu_tee": "snp",
        "gpu": "nvidia",
        "kata_version": "3.29.0",
        "guest_token_api": API["CAPABILITY"],
        "workload_security_context": SECURITY_CONTEXT,
        "kata_config": tomllib.loads(installed),
        "kata_config_sha256": hashlib.sha256(installed.encode()).hexdigest(),
        "pod_resources": resources,
        "runtime_default_vcpus": 1,
        "runtime_default_memory_mib": 8192,
        "artifacts": {
            name: {"sha256": hashlib.sha256(name.encode()).hexdigest()}
            for name in ("path", "firmware", "kernel", "initrd")
        },
    }
    actual = {
        "cpu_tee": "snp",
        "gpu": "nvidia",
        "launch_inputs": {"kernel_command_line": API["REQUIRED"], "smp": "1", "memory": "8192M"},
        "pod_resources": [resources],
        "artifacts": {"kata_config": {"sha256": profile["kata_config_sha256"]}},
    }
    actual["artifacts"].update(
        {"qemu_executable" if key == "path" else key: value for key, value in profile["artifacts"].items()}
    )
    actual["launch_inputs_sha256"] = hashlib.sha256(
        json.dumps(actual["launch_inputs"], sort_keys=True).encode()
    ).hexdigest()
    value = capture["build_launch_profile"](profile, actual, "reviewed")
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps(value))
    assert ADMIN["load_profile"](contract, None, runtime, "3.29.0")
    assert value["workload_security_context"] == SECURITY_CONTEXT
    assert "kata_deploy_image" not in value
    del profile["workload_security_context"]
    with pytest.raises((KeyError, ValueError)):
        capture["build_launch_profile"](profile, actual, "reviewed")
    profile["workload_security_context"] = {**SECURITY_CONTEXT, "privileged": True}
    with pytest.raises(ValueError, match="fixed NVFlare"):
        capture["build_launch_profile"](profile, actual, "reviewed")
    profile["workload_security_context"] = {**SECURITY_CONTEXT, "readOnlyRootFilesystem": True}
    with pytest.raises(ValueError, match="fixed NVFlare"):
        capture["build_launch_profile"](profile, actual, "reviewed")
    profile["workload_security_context"] = SECURITY_CONTEXT
    actual["launch_inputs"]["kernel_command_line"] = "quiet"
    with pytest.raises(ValueError, match="guest_components_rest_api"):
        capture["build_launch_profile"](profile, actual, "reviewed")
    actual["launch_inputs"]["kernel_command_line"] = API["REQUIRED"]
    del profile["guest_token_api"]
    with pytest.raises(ValueError, match="guest-local token API"):
        capture["build_launch_profile"](profile, actual, "reviewed")
