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

"""Capture one verified platform reference from an existing CoCo-ready worker."""

import base64
import gzip
import hashlib
import json
import os
import re
import runpy
import secrets
import subprocess
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ENV_KEYS = [
    "NODE_NAME",
    "RUNTIME_CLASS",
    "INSTALLED_KATA_CONFIG",
    "KATA_RUNTIME",
    "KUBECONFIG_PATH",
    "COLLECTOR_IMAGE",
    "COLLECTOR_METADATA_FILE",
    "COLLECTOR_METADATA_SHA256",
    "COLLECTOR_REGISTRY_CA",
    "SNP_VERIFIER",
    "TDX_VERIFIER",
    "SNP_MIN_REPORTED_TCB_BOOTLOADER",
    "SNP_MIN_REPORTED_TCB_TEE",
    "SNP_MIN_REPORTED_TCB_SNP",
    "SNP_MIN_REPORTED_TCB_MICROCODE",
]


def helpers():
    return runpy.run_path(str(HERE / "installed-platform.py"))


def runtime_api():
    return runpy.run_path(str(HERE / "kata-runtime-profile.py"))


def evidence_api():
    return runpy.run_path(str(HERE / "platform-evidence.py"))


def security_api():
    return runpy.run_path(str(HERE / "workload-security-context.py"))


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def fixed_workload(runtime):
    target = runtime_api()["runtime_target"](runtime)
    context = {
        "privileged": False,
        "allowPrivilegeEscalation": False,
        "runAsNonRoot": True,
        "runAsUser": 65532,
        "runAsGroup": 65532,
        "readOnlyRootFilesystem": False,
        "capabilities": {"drop": ["ALL"]},
        "seccompProfile": {"type": "RuntimeDefault"},
    }
    resources = (
        {"limits": {"nvidia.com/pgpu": "1"}, "requests": {"nvidia.com/pgpu": "1"}} if target["gpu_count"] else {}
    )
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": "nvflare-workload-shape"},
        "spec": {
            "runtimeClassName": runtime,
            "restartPolicy": "Never",
            "automountServiceAccountToken": False,
            "enableServiceLinks": False,
            "containers": [
                {"name": "nvflare", "image": "nvflare-workload", "securityContext": context, "resources": resources}
            ],
        },
    }


def prepare_profile(snapshot, runtime):
    target = runtime_api()["runtime_target"](runtime)
    config = snapshot["config"]
    qemu = config["hypervisor"]["qemu"]
    runtime_api()["require_confidential_config"](qemu, runtime)
    runtime_api()["require_token_api"](qemu.get("kernel_params"))
    workload = fixed_workload(runtime)
    container = workload["spec"]["containers"][0]
    artifacts = {
        key: {"path": qemu[key], "sha256": snapshot["files"][qemu[key]]}
        for key in ("path", "firmware", "kernel", "initrd", "image")
        if qemu.get(key)
    }
    return {
        "runtime_class": runtime,
        "cpu_tee": target["cpu_tee"],
        "gpu": target["gpu"],
        "guest_token_api": runtime_api()["CAPABILITY"],
        "workload_security_context": container["securityContext"],
        "pod_resources": container["resources"],
        "kata_config": config,
        "kata_config_sha256": snapshot["files"][snapshot["installed_config"]],
        "kata_version": snapshot.get("runtime_version"),
        "runtime_default_vcpus": qemu["default_vcpus"],
        "runtime_default_memory_mib": qemu["default_memory"],
        "artifacts": artifacts,
    }


def build_launch_profile(profile, actual, profile_id, runtime_version=None):
    api = runtime_api()
    runtime = profile["runtime_class"]
    target = api["require_runtime_target"](profile, runtime)
    api["require_runtime_target"](actual, runtime)
    expected = fixed_workload(runtime)["spec"]["containers"][0]
    if (
        profile["workload_security_context"] != expected["securityContext"]
        or profile["pod_resources"] != expected["resources"]
        or actual["pod_resources"] != [expected["resources"]]
    ):
        raise ValueError("Only the fixed NVFlare security context and target resources are supported")
    security_api()["validate_context"](profile["workload_security_context"])
    if profile.get("guest_token_api") != api["CAPABILITY"]:
        raise ValueError("The installed profile lacks the guest-local token API")
    api["require_token_api"](profile["kata_config"]["hypervisor"]["qemu"].get("kernel_params"))
    api["require_token_api"](actual["launch_inputs"].get("kernel_command_line"))
    if actual["artifacts"]["kata_config"]["sha256"] != profile["kata_config_sha256"]:
        raise ValueError("Captured runtime configuration differs from the installed snapshot")
    if not ("initrd" in actual["artifacts"] or "image" in actual["artifacts"]):
        raise ValueError("Actual launch lacks an initrd or rootfs image")
    for key in ("path", "firmware", "kernel", "initrd", "image"):
        observed = actual["artifacts"].get("qemu_executable" if key == "path" else key)
        if key in ("initrd", "image") and observed is None:
            continue
        if not observed or observed["sha256"] != profile["artifacts"].get(key, {}).get("sha256"):
            raise ValueError("Captured launch artifact differs from the installed snapshot: " + key)
    fingerprint = hashlib.sha256(json.dumps(actual["launch_inputs"], sort_keys=True).encode()).hexdigest()
    if fingerprint != actual["launch_inputs_sha256"]:
        raise ValueError("Captured launch fingerprint is inconsistent")
    vcpus, memory = profile["runtime_default_vcpus"], profile["runtime_default_memory_mib"]
    if type(vcpus) is not int or vcpus <= 0 or type(memory) is not int or memory <= 0:
        raise ValueError("Invalid installed VM defaults")
    if actual["launch_inputs"]["smp"].split(",")[0] != str(vcpus) or actual["launch_inputs"]["memory"] != f"{memory}M":
        raise ValueError("Actual VM sizing differs from the installed runtime defaults")
    version = runtime_version or profile.get("kata_version")
    if version != "3.29.0":
        raise ValueError("The existing runtime must match supported Kata/genpolicy 3.29.0")
    return {
        "schema": api["PROFILE_SCHEMA_V4"],
        "cpu_tee": target["cpu_tee"],
        "gpu": target["gpu"],
        "workload_security_context": profile["workload_security_context"],
        "guest_token_api": api["CAPABILITY"],
        "profile_id": profile_id,
        "runtime_class": runtime,
        "kata_version": version,
        "kata_config_sha256": profile["kata_config_sha256"],
        "launch_inputs_sha256": fingerprint,
        "vm_defaults": {"vcpus": vcpus, "memory_mib": memory},
        "pod_constraints": {
            "container_count": 1,
            "gpu_resource": "nvidia.com/pgpu" if target["gpu_count"] else None,
            "gpu_count": target["gpu_count"],
            "cpu_memory_resources": "omitted",
            "host_namespaces": False,
            "allowed_annotations": ["io.katacontainers.config.hypervisor.cc_init_data"],
        },
    }


def snp_floors(env):
    result = {}
    for component in ("bootloader", "tee", "snp", "microcode"):
        key = "SNP_MIN_REPORTED_TCB_" + component.upper()
        value = env.get(key, "")
        if not isinstance(value, str) or not re.fullmatch(r"0|[1-9][0-9]{0,2}", value) or int(value) > 255:
            raise ValueError("Supply independently approved uint8 " + key)
        result["snp_min_reported_tcb_" + component] = int(value)
    return result


def make_collector(env, snapshot, profile, directory, namespace):
    target = runtime_api()["runtime_target"](env["RUNTIME_CLASS"])
    ca = Path(snapshot["registry_ca"]).read_text()
    cdh = (
        '[kbc]\nname = "offline_fs_kbc"\nurl = ""\n[image]\nextra_root_certificates = ['
        + json.dumps(ca)
        + ']\n[image.registry_config]\nunqualified-search-registries = ["docker.io"]\n'
        "[[image.registry_config.registry]]\nlocation = "
        + json.dumps(env["COLLECTOR_IMAGE"].split("/", 1)[0])
        + "\ninsecure = false\n"
    )
    initdata = 'version = "0.1.0"\nalgorithm = "sha256"\n[data]\n"cdh.toml" = ' + json.dumps(cdh) + "\n"
    (directory / "initdata.toml").write_text(initdata)
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": "collector",
            "namespace": namespace,
            "annotations": {
                "io.katacontainers.config.hypervisor.cc_init_data": base64.b64encode(
                    gzip.compress(initdata.encode(), mtime=0)
                ).decode()
            },
        },
        "spec": {
            "runtimeClassName": env["RUNTIME_CLASS"],
            "nodeName": snapshot["node"]["name"],
            "restartPolicy": "Never",
            "automountServiceAccountToken": False,
            "enableServiceLinks": False,
            "containers": [
                {
                    "name": "collector",
                    "image": env["COLLECTOR_IMAGE"],
                    "imagePullPolicy": "Never",
                    "stdin": False,
                    "tty": False,
                    "securityContext": {"privileged": target["cpu_tee"] == "snp", "runAsUser": 0, "runAsGroup": 0},
                    "resources": profile["pod_resources"],
                    "volumeMounts": [{"name": "challenge", "mountPath": "/challenge", "readOnly": True}],
                }
            ],
            "volumes": [{"name": "challenge", "configMap": {"name": "trusted-challenge", "defaultMode": 256}}],
        },
    }


def collect(env, output, snapshot, profile):
    helper = helpers()
    directory = output / "evidence"
    directory.mkdir(mode=0o700)
    helper["cache_collector"](env["COLLECTOR_IMAGE"], directory, env, profile=output)
    challenge = secrets.token_hex(32).encode("ascii")
    (directory / "request-data.bin").write_bytes(challenge)
    namespace = "coco-measurement-" + secrets.token_hex(8)
    pod = make_collector(env, snapshot, profile, directory, namespace)
    pod_path = directory / "collector-pod.yaml"
    pod_path.write_text(yaml.safe_dump(pod, sort_keys=False))
    kube = helper["kube_command"](snapshot)
    created = False
    try:
        subprocess.run(kube + ["create", "namespace", namespace], check=True)
        created = True
        subprocess.run(
            kube
            + [
                "-n",
                namespace,
                "create",
                "configmap",
                "trusted-challenge",
                "--from-file=request-data.bin=" + str(directory / "request-data.bin"),
            ],
            check=True,
        )
        subprocess.run(kube + ["apply", "-f", str(pod_path)], check=True)
        captured = False
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            obj = json.loads(
                subprocess.check_output(kube + ["-n", namespace, "get", "pod", "collector", "-o", "json"], text=True)
            )
            helper["check_pod_node"](obj, snapshot)
            phase = obj.get("status", {}).get("phase")
            if phase == "Running" and not captured:
                command = [
                    "sudo",
                    "python3",
                    str(HERE.parent / "capture-running-launch.py"),
                    namespace,
                    "collector",
                    snapshot["installed_config"],
                    str(directory / "actual-launch.json"),
                    snapshot["kubeconfig"],
                    "--platform",
                    str(output),
                ]
                result = subprocess.run(command, timeout=60)
                if result.returncode not in (0, 75):
                    raise ValueError("Cannot capture the collector's actual runtime launch")
                if result.returncode == 0:
                    subprocess.run(
                        ["sudo", "chown", f"{os.getuid()}:{os.getgid()}", str(directory / "actual-launch.json")],
                        check=True,
                    )
                    captured = True
            if phase == "Succeeded":
                break
            if phase == "Failed":
                raise ValueError("Diagnostic collector failed; retain this profile for diagnosis")
            time.sleep(2)
        else:
            raise ValueError("Diagnostic collector did not finish within 600 seconds")
        if not captured:
            raise ValueError("The collector finished without an actual runtime launch capture")
        write_json(directory / "pod-result.json", obj)
        logs = subprocess.check_output(kube + ["-n", namespace, "logs", "collector"], text=True)
        (directory / "collector.log").write_text(logs)
        crypto = evidence_api()
        if profile["cpu_tee"] == "snp":
            source = crypto["decode_snp"](logs, directory)
            crypto["verify_snp"](source, snapshot["verifier"], snp_floors(env), fetch_certificates=True)
        else:
            write_json(directory / "evidence.json", crypto["decode_evidence"](logs))
        return verify_evidence(output, env, snapshot)
    finally:
        if created:
            subprocess.run(kube + ["delete", "namespace", namespace, "--wait=false"], check=False, timeout=35)


def verify_evidence(output, env, snapshot):
    crypto = evidence_api()
    directory = output / "evidence"
    if runtime_api()["runtime_target"](env["RUNTIME_CLASS"])["cpu_tee"] == "snp":
        values, summary = crypto["verify_snp"](directory / "evidence-input", snapshot["verifier"], snp_floors(env))
    else:
        measured, claims = crypto["verify_tdx"](directory, snapshot["verifier"], output.name)
        values = {"schema": "coco-platform-reference-values/v2", "tee": "tdx", "profiles": [measured]}
        summary = {"profile": measured}
    runpy.run_path(str(HERE / "platform-reference-schema.py"))["validate_values"](values)
    return values, summary


def require_capture_tools():
    # Match capture-running-launch.py's effective sudo Python environment:
    # checking the caller's PATH would miss sudoers secure_path differences.
    command = [
        "sudo",
        "python3",
        "-c",
        "import runpy,shutil,sys\n"
        "if sys.version_info < (3, 11): raise SystemExit('Python 3.11+ is required')\n"
        "runpy.run_path(sys.argv[1])\n"
        "if not shutil.which('crictl'): raise SystemExit('crictl is required')\n",
        str(HERE.parent / "capture-running-launch.py"),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError(
            "Capture requires Python 3.11+ with the capture helper's dependencies and crictl in the PATH used by "
            "sudo python3; check the privileged interpreter and command access before retrying"
        ) from error


def capture(config, output):
    helper = helpers()
    env = helper["read_config"](Path(config).resolve(strict=True), ENV_KEYS)
    target = runtime_api()["runtime_target"](env["RUNTIME_CLASS"])
    if target["cpu_tee"] == "snp":
        snp_floors(env)
    output = Path(output).absolute()
    name_pattern = r"[a-z0-9][a-z0-9_.-]{0,63}" if target["cpu_tee"] == "tdx" else r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}"
    if not re.fullmatch(name_pattern, output.name):
        if target["cpu_tee"] == "tdx":
            raise ValueError(
                "TDX output directory name must be 1..64 lowercase alphanumeric, dot, underscore or hyphen characters, "
                "starting with a lowercase letter or digit"
            )
        raise ValueError("Choose a simple profile directory name")
    if output.exists() or output.is_symlink():
        raise ValueError("Use a new output directory; existing evidence is never overwritten")
    require_capture_tools()
    os.umask(0o077)
    output.mkdir(parents=True, mode=0o700)
    write_json(output / "capture-inputs.json", env)
    snapshot = helper["inspect_platform"](output, env)
    profile = prepare_profile(snapshot, env["RUNTIME_CLASS"])
    write_json(output / "workload-profile.json", profile)
    values, summary = collect(env, output, snapshot, profile)
    helper["verify_snapshot"](output, env)
    actual = json.loads((output / "evidence/actual-launch.json").read_text())
    helper["verify_launch"](actual, snapshot, output)
    build_launch_profile(profile, actual, output.name)
    write_json(
        output / "capture-summary.json",
        {
            "node": snapshot["node"],
            "runtime_class": env["RUNTIME_CLASS"],
            "kata_version": snapshot["runtime_version"],
            "workload_security_context": profile["workload_security_context"],
            "resources": profile["pod_resources"],
            "verified_platform_references": values,
            "evidence_summary": summary,
        },
    )
    retained = {
        str(path.relative_to(output)): helper["digest"](path)
        for path in output.rglob("*")
        if path.is_file() and path.suffix not in (".tar", ".gz")
    }
    write_json(output / "capture-record.json", retained)
    _export_verified_capture(output)
    print("Verified capture summary:", output / "capture-summary.json")
    return output


def _export_verified_capture(output):
    output = Path(output).resolve(strict=True)
    helper = helpers()
    record = json.loads((output / "capture-record.json").read_text())
    for relative, expected in record.items():
        path = output / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts or helper["digest"](path) != expected:
            raise ValueError("Captured evidence changed before export: " + relative)
    env = json.loads((output / "capture-inputs.json").read_text())
    snapshot = helper["verify_snapshot"](output, env)
    profile = json.loads((output / "workload-profile.json").read_text())
    actual = json.loads((output / "evidence/actual-launch.json").read_text())
    helper["verify_launch"](actual, snapshot, output)
    helper["check_pod_node"](json.loads((output / "evidence/pod-result.json").read_text()), snapshot)
    values, summary = verify_evidence(output, env, snapshot)
    if values != json.loads((output / "capture-summary.json").read_text())["verified_platform_references"]:
        raise ValueError("Reverified references differ from the captured values")
    contract = build_launch_profile(profile, actual, output.name)
    values_path, launch_path = (
        output / "platform-reference-values.json",
        output / "approved-workload-launch-profile.json",
    )
    if values_path.exists() or values_path.is_symlink() or launch_path.exists() or launch_path.is_symlink():
        raise ValueError("Verified outputs already exist; preserve this capture")
    write_json(values_path, values)
    try:
        write_json(launch_path, contract)
    except BaseException:
        values_path.unlink()
        raise
    print("Verified Trustee references:", values_path)
    print("Verified workload launch profile:", launch_path)
    return values, contract
