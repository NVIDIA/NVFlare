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

"""Read-only runtime selection and capability guards for CoCo workloads."""

CLI_REMOVED_MESSAGE = (
    "The Kata runtime configuration CLI has been removed. Use a preconfigured CoCo-ready runtime "
    "with exactly one agent.guest_components_rest_api=all kernel parameter; "
    "this module provides read-only capture helpers."
)

OPTION = "agent.guest_components_rest_api"
REQUIRED = OPTION + "=all"
CAPABILITY = "guest-local-aa-token/v1"
PROFILE_SCHEMA_V4 = "coco-approved-workload-launch/v4"
RUNTIME_TARGETS = {
    "kata-qemu-nvidia-gpu-snp": ("amd_sev_snp", "nvidia"),
    "kata-qemu-snp": ("amd_sev_snp", "none"),
    "kata-qemu-nvidia-gpu-tdx": ("intel_tdx", "nvidia"),
    "kata-qemu-tdx": ("intel_tdx", "none"),
}


def runtime_target(runtime):
    """Resolve a reviewed RuntimeClass, never infer a TEE from a substring."""
    if not isinstance(runtime, str) or runtime not in RUNTIME_TARGETS:
        raise ValueError("Unsupported confidential RuntimeClass")
    tee, gpu = RUNTIME_TARGETS[runtime]
    return {
        "cpu_tee": tee,
        "gpu": gpu,
        "gpu_count": 1 if gpu == "nvidia" else 0,
        "config_name": "configuration-" + runtime.removeprefix("kata-") + ".toml",
        "node_label": (
            "amd.feature.node.kubernetes.io/snp" if tee == "amd_sev_snp" else "intel.feature.node.kubernetes.io/tdx"
        ),
    }


def require_runtime_target(profile, runtime):
    """Keep legacy approvals limited to their original SNP+GPU interpretation."""
    target = runtime_target(runtime)
    legacy = "cpu_tee" not in profile and "gpu" not in profile
    if legacy and runtime == "kata-qemu-nvidia-gpu-snp":
        return target
    if profile.get("cpu_tee") != target["cpu_tee"] or profile.get("gpu") != target["gpu"]:
        raise ValueError("Approved CPU/GPU target differs from RuntimeClass")
    return target


def require_confidential_config(qemu, runtime):
    """Reject non-confidential or cross-TEE settings in an existing runtime."""
    target = runtime_target(runtime)
    if qemu.get("confidential_guest") is not True:
        raise ValueError("Kata configuration must enable confidential_guest")
    if qemu.get("sev_snp_guest", False) is not (target["cpu_tee"] == "amd_sev_snp"):
        raise ValueError("Kata sev_snp_guest does not match approved CPU TEE")
    if target["cpu_tee"] == "intel_tdx" and qemu.get("machine_type") != "q35":
        raise ValueError("The approved TDX profile requires QEMU q35")
    return target


def require_token_api(cmdline):
    if not isinstance(cmdline, str):
        raise ValueError("Missing kernel command line")
    values = [part for part in cmdline.split() if part == OPTION or part.startswith(OPTION + "=")]
    if values != [REQUIRED]:
        raise ValueError("Require exactly one " + REQUIRED)


if __name__ == "__main__":
    raise SystemExit(CLI_REMOVED_MESSAGE)
