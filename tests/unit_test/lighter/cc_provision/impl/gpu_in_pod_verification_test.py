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

"""Optional GPU diagnostic orchestration tests, not hardware-attestation tests."""

import copy
import importlib.util
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[5]
TRUSTED = ROOT / "examples/devops/coco/trusted_system"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


workflow = load_module("gpu_in_pod_verification", TRUSTED / "verify-gpu-in-pod.py")
tdx = load_module("gpu_diagnostic_tdx_reference", TRUSTED / "tdx-reference.py")
GPU_RUNTIMES = ["kata-qemu-nvidia-gpu-snp", "kata-qemu-nvidia-gpu-tdx"]


def collector_profile(tmp_path, runtime):
    first = tmp_path / "rehearsal-collector-build"
    first.mkdir()
    resources = {
        "limits": {"cpu": "2", "memory": "8Gi", "nvidia.com/pgpu": "1"},
        "requests": {"cpu": "2", "memory": "8Gi", "nvidia.com/pgpu": "1"},
    }
    if runtime == "kata-qemu-nvidia-gpu-tdx":
        # Exercise the actual TDX producer, including its digest pin, InitData,
        # Python image entrypoint and retained rehearsal filesystem layout.
        tls = first / "registry-tls"
        tls.mkdir()
        (tls / "ca.crt").write_text("fixture certificate")
        workload = tmp_path / "workload.yaml"
        workload.write_text(
            yaml.safe_dump({"spec": {"runtimeClassName": runtime, "containers": [{"resources": resources}]}})
        )
        pod = tdx.make_pod(
            {"REHEARSAL_WORKLOAD_YAML": str(workload), "RUNTIME_CLASS": runtime},
            first,
            "trusted-tdx-rehearsal",
            "192.0.2.1:5443",
            "192.0.2.1:5443/coco-tdx-rehearsal@sha256:" + "a" * 64,
        )
    else:
        pod = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": "snp-evidence-fixture",
                "namespace": "trusted-snp-rehearsal",
                "annotations": {"io.katacontainers.config.hypervisor.cc_init_data": "fixture-TLS-init-data"},
            },
            "spec": {
                "runtimeClassName": runtime,
                "automountServiceAccountToken": False,
                "enableServiceLinks": False,
                "restartPolicy": "Never",
                "containers": [
                    {
                        "name": "collector",
                        "image": "192.0.2.1:5443/coco-snp-rehearsal:snpguest-fixture",
                        "imagePullPolicy": "Never",
                        "securityContext": {"privileged": True, "runAsUser": 0, "runAsGroup": 0},
                        "resources": resources,
                        "volumeMounts": [{"name": "challenge", "mountPath": "/challenge", "readOnly": True}],
                    }
                ],
                "volumes": [{"name": "challenge", "configMap": {"name": "trusted-challenge", "defaultMode": 256}}],
            },
        }
    (first / "collector-pod.yaml").write_text(yaml.safe_dump(pod))
    return first, pod


@pytest.mark.parametrize("runtime", GPU_RUNTIMES)
def test_gpu_pod_preserves_selected_collector_launch_and_tls(tmp_path, runtime):
    first, original = collector_profile(tmp_path, runtime)
    result = workflow.prepare_pod(first, "gpu-diagnostic-fixture")
    spec = result["spec"]
    container = spec["containers"][0]
    assert result["metadata"]["annotations"] == original["metadata"]["annotations"]
    assert result["metadata"]["namespace"] == "gpu-diagnostic-fixture"
    assert spec["runtimeClassName"] == runtime
    for field in ("image", "imagePullPolicy", "resources"):
        assert container[field] == original["spec"]["containers"][0][field]
    assert spec["automountServiceAccountToken"] is False
    assert container["securityContext"] == {"privileged": False, "runAsUser": 0, "allowPrivilegeEscalation": False}
    assert container["command"] == ["/bin/bash", "-c"]
    assert "nvidia-smi -q" in container["args"][0]
    assert "/gpu-probe/cuda-probe" in container["args"][0]
    assert container["volumeMounts"] == [{"name": "gpu-probe", "mountPath": "/gpu-probe", "readOnly": True}]
    assert spec["volumes"] == [{"name": "gpu-probe", "configMap": {"name": "gpu-probe", "defaultMode": 0o555}}]
    assert yaml.safe_load((first / "collector-pod.yaml").read_text()) == original


@pytest.mark.parametrize("runtime", ["kata-qemu-snp", "kata-qemu-tdx", "kata-qemu", "unknown"])
def test_cpu_only_and_unknown_rejected_before_any_side_effect(tmp_path, runtime):
    collector_profile(tmp_path, runtime)
    before = sorted(tmp_path.rglob("*"))
    message = "not applicable to CPU-only" if runtime in workflow.CPU_ONLY_RUNTIMES else "supported confidential"
    with patch.object(workflow.subprocess, "run") as run, patch.object(workflow.subprocess, "check_output") as output:
        with patch.object(workflow.os, "umask") as umask, patch.object(Path, "mkdir") as mkdir:
            with pytest.raises(ValueError, match=message):
                workflow.main(tmp_path)
            mkdir.assert_not_called()
            umask.assert_not_called()
        run.assert_not_called()
        output.assert_not_called()
    assert sorted(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("runtime", GPU_RUNTIMES)
@pytest.mark.parametrize("count", [None, 0, 2, True])
def test_invalid_gpu_allocation_rejected_before_commands(tmp_path, runtime, count):
    first, pod = collector_profile(tmp_path, runtime)
    pod["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/pgpu"] = count
    (first / "collector-pod.yaml").write_text(yaml.safe_dump(pod))
    with patch.object(workflow.subprocess, "run") as run, patch.object(Path, "mkdir") as mkdir:
        with pytest.raises(ValueError, match="one passthrough GPU"):
            workflow.main(tmp_path)
        run.assert_not_called()
        mkdir.assert_not_called()


@pytest.mark.parametrize("field", ["hostNetwork", "hostPID", "hostIPC", "initContainers", "containers"])
def test_unexpected_launch_shape_rejected(tmp_path, field):
    first, pod = collector_profile(tmp_path, "kata-qemu-nvidia-gpu-tdx")
    if field == "containers":
        pod["spec"][field].append(copy.deepcopy(pod["spec"][field][0]))
    else:
        pod["spec"][field] = True
    (first / "collector-pod.yaml").write_text(yaml.safe_dump(pod))
    with patch.object(workflow.subprocess, "run") as run, patch.object(Path, "mkdir") as mkdir:
        with pytest.raises(ValueError):
            workflow.main(tmp_path)
        run.assert_not_called()
        mkdir.assert_not_called()


@pytest.mark.parametrize("runtime", GPU_RUNTIMES)
@pytest.mark.parametrize("phase", ["Succeeded", "Failed"])
def test_orchestration_retains_registry_tls_and_cleans_up_without_approving(tmp_path, runtime, phase):
    first, original = collector_profile(tmp_path, runtime)
    approved = tmp_path / "platform-reference-values.json"
    approved.write_text("approved reference fixture, never modified")
    with (
        patch.object(workflow.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run,
        patch.object(
            workflow.subprocess,
            "check_output",
            side_effect=[json.dumps({"status": {"phase": phase}}), "GPU_SMOKE_TEST_PASS\n"],
        ),
        patch.object(workflow.os, "access", return_value=True),
        patch.object(workflow.os, "umask"),
        patch.object(workflow.secrets, "token_hex", return_value="fixture"),
    ):
        if phase == "Failed":
            with pytest.raises(RuntimeError, match="In-Pod GPU check failed"):
                workflow.main(tmp_path)
        else:
            workflow.main(tmp_path)
    commands = [call.args[0] for call in run.call_args_list]
    registry = next(command for command in commands if command[:2] == ["docker", "run"])
    assert "REGISTRY_HTTP_ADDR=192.0.2.1:5443" in registry
    assert str(first / "registry-tls") + ":/tls:ro" in registry
    assert str(first / "registry-data") + ":/var/lib/registry" in registry
    curl = next(command for command in commands if command[0] == "curl")
    assert curl[curl.index("--cacert") + 1] == str(first / "registry-tls/ca.crt")
    assert "--insecure" not in curl
    kube = ["kubectl", "--kubeconfig", "/etc/kubernetes/admin.conf"]
    assert kube + ["delete", "namespace", "coco-gpu-check-fixture", "--wait=true", "--timeout=120s"] in commands
    assert ["docker", "rm", "-f", "coco-gpu-check-registry-fixture"] in commands
    emitted = yaml.safe_load((tmp_path / "gpu-verification-fixture/pod.yaml").read_text())
    assert emitted["spec"]["containers"][0]["image"] == original["spec"]["containers"][0]["image"]
    assert approved.read_text() == "approved reference fixture, never modified"
