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

"""Trusted-host TDX rehearsal. Cryptography is delegated to the pinned Trustee verifier.

Only trusted local configuration files may be supplied. Collecting valid evidence
does not approve a new software configuration: stage 09 additionally requires the
reviewed candidate's exact SHA-256 in platform-approval.env.
"""

import argparse
import base64
import gzip
import hashlib
import json
import os
import re
import runpy
import secrets
import shlex
import shutil
import subprocess
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
RUNTIMES = {
    "kata-qemu-tdx": "configuration-qemu-tdx.toml",
    "kata-qemu-nvidia-gpu-tdx": "configuration-qemu-nvidia-gpu-tdx.toml",
}
REGISTRY = "docker.io/library/registry@sha256:a3d8aaa63ed8681a604f1dea0aa03f100d5895b6a58ace528858a7b332415373"
FRAME = "COCO_TDX_EVIDENCE_V1="
ENV_KEYS = (
    "PLATFORM_PROFILE PLATFORM_WORK_ROOT KATA_VERSION RUNTIME_CLASS KATA_CHART_OCI "
    "KATA_CHART_OCI_DIGEST KATA_CHART_TGZ_SHA256 KATA_DEPLOY_INDEX KATA_DEPLOY_AMD64 "
    "REHEARSAL_WORKLOAD_YAML KUBECONFIG_PATH TDX_VERIFIER TDX_VERIFIER_SHA256 "
    "TDX_SECURITY_BASELINE_APPROVED APPROVED_TDX_PROFILE_SHA256 "
    "APPROVED_WORKLOAD_PROFILE_SHA256 APPROVED_ACTUAL_LAUNCH_SHA256 "
    "TDX_REFERENCE_VALUES_SHA256"
).split()


def run(args, **kwargs):
    return subprocess.run([str(x) for x in args], check=True, **kwargs)


def output(args, timeout=300):
    return run(args, capture_output=True, text=True, timeout=timeout).stdout.strip()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    """Never replace prior evidence or approved artifacts."""
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(data)


def write_json(path, value):
    write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_env(path):
    # Files here are trusted local shell configuration, never received handoffs.
    # Approval must be in this reviewed file, not accidentally inherited from a
    # shell that approved an earlier run. Keep HOME/PATH for normal config use.
    script = "set -ae; unset " + " ".join(ENV_KEYS) + '; source "$1"; python3 -c "$2"'
    code = "import json,os; print(json.dumps({k:os.environ[k] for k in " + repr(ENV_KEYS) + " if k in os.environ}))"
    return json.loads(output(["bash", "-c", script, "tdx-reference", Path(path).resolve(strict=True), code]))


def config(path):
    env = read_env(path)
    if env.get("RUNTIME_CLASS") not in RUNTIMES:
        raise ValueError("Expected an explicitly selected TDX runtime")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", env.get("PLATFORM_PROFILE", "")):
        raise ValueError("Invalid PLATFORM_PROFILE")
    root = Path(env["PLATFORM_WORK_ROOT"])
    if not root.is_absolute():
        raise ValueError("PLATFORM_WORK_ROOT must be absolute")
    profile = root / env["PLATFORM_PROFILE"]
    if not profile.is_dir():
        raise ValueError("Run stages 03-05 first")
    return env, profile


def approval(profile, path):
    path = Path(path).resolve(strict=True)
    if path.parent != profile.resolve(strict=True):
        raise ValueError("Approval environment must be in selected profile")
    values = read_env(path)
    if values.get("TDX_SECURITY_BASELINE_APPROVED") != "1":
        raise ValueError(
            "Review the strict TDX security baseline; set TDX_SECURITY_BASELINE_APPROVED=1 before rehearsal"
        )
    return values


def prepare(config_path):
    env, profile = config(config_path)
    if not (profile / "approved-kata-config.toml").is_file():
        raise ValueError("Run stage 03 first")
    for name in ("platform-derived.env", "platform-approval.env"):
        if (profile / name).exists():
            raise ValueError("Refusing to overwrite " + name)
    tools = profile / "tdx-verifier-build"
    run(["bash", HERE / "tdx-verifier/build.sh", tools])
    verifier = tools / "tdx-evidence-verify"
    if not verifier.is_file() or not os.access(verifier, os.X_OK):
        raise ValueError("Pinned TDX verifier build did not produce its executable")
    write(
        profile / "platform-derived.env",
        f"TDX_VERIFIER={shlex.quote(str(verifier))}\nTDX_VERIFIER_SHA256={sha(verifier)}\n",
    )
    write(
        profile / "platform-approval.env",
        "# Trusted authority reviews this fixed baseline before stage 07:\n"
        "# Intel DCAP UpToDate; collateral not expired; debug=false;\n"
        "# fresh REPORTDATA; InitData MRCONFIGID; all four RTMRs replayed.\n"
        "TDX_SECURITY_BASELINE_APPROVED=\n"
        "# After stage 08, review candidate-tdx-profile.json AND runtime artifacts.\n"
        "# Set SHA256 of that exact file to authorize stage 09 (not automatic).\n"
        "APPROVED_TDX_PROFILE_SHA256=\n",
    )
    print("Prepared TDX verifier and platform-approval.env; review baseline before stage 07.")


def verifier(profile):
    values = read_env(profile / "platform-derived.env")
    path = Path(values["TDX_VERIFIER"])
    if not path.is_file() or sha(path) != values.get("TDX_VERIFIER_SHA256"):
        raise ValueError("Pinned verifier changed after preparation")
    return path


def preflight(env, profile):
    """Read-only local checks before any registry, namespace or collector starts."""
    missing = [
        name
        for name in ("docker", "kubectl", "curl", "openssl", "skopeo", "ip", "sudo", "sha256sum", "ctr", "crictl")
        if not shutil.which(name)
    ]
    if missing:
        raise ValueError("Missing rehearsal commands: " + ", ".join(missing))
    workload_path = Path(env.get("REHEARSAL_WORKLOAD_YAML", ""))
    if not workload_path.is_file():
        raise ValueError("REHEARSAL_WORKLOAD_YAML must identify the reviewed source Pod")
    workload = yaml.safe_load(workload_path.read_text())
    approved = json.loads((profile / "approved-launch-profile.json").read_text())
    if approved.get("workload_yaml_sha256") != sha(workload_path):
        raise ValueError("Source Pod changed after stage 05 approval")
    if workload.get("spec", {}).get("runtimeClassName") != env["RUNTIME_CLASS"]:
        raise ValueError("Source Pod runtime differs from approved runtime")
    runpy.run_path(str(HERE / "lib/workload-security-context.py"))["validate_pod_context"](
        workload, approved.get("workload_security_context")
    )
    pinned = profile / (profile / "kata-config-relative-path.txt").read_text().strip()
    run(
        [
            "python3",
            HERE / "lib/kata-runtime-profile.py",
            "verify",
            pinned,
            profile / "approved-kata-config.toml",
            profile / "kata-runtime-profile.json",
            "--installed",
            "/opt/kata/share/defaults/kata-containers/" + RUNTIMES[env["RUNTIME_CLASS"]],
        ]
    )


def extract_profile(claims, profile_id):
    """Extract only already verified Trustee claims, never decoded quote bytes."""
    body = claims["quote"]["body"]
    result = {"id": profile_id}
    for field in ("mr_td", "rtmr_1", "rtmr_2", "xfam"):
        value = body.get(field)
        length = 16 if field == "xfam" else 96
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value):
            raise ValueError("Invalid verified TDX field: " + field)
        result[field] = value
    for name in ("tdvfkernel", "tdvfkernelparams"):
        matches = []
        event_count = 0
        for event in claims.get("uefi_event_logs", []):
            details = event.get("details", {})
            kernel = event.get("type_name") == "EV_EFI_BOOT_SERVICES_APPLICATION" and "File(kernel)" in details.get(
                "device_paths", []
            )
            params = event.get("type_name") == "EV_EVENT_TAG" and details.get("string") == "LOADED_IMAGE::LoadOptions"
            if not (kernel if name == "tdvfkernel" else params):
                continue
            event_count += 1
            matches.extend(d.get("digest") for d in event.get("digests", []) if d.get("alg") == "SHA-384")
        if (
            event_count != 1
            or len(matches) != 1
            or not isinstance(matches[0], str)
            or not re.fullmatch(r"[0-9a-f]{96}", matches[0])
        ):
            raise ValueError("Require exactly one verified TDVF " + name + " SHA-384 event")
        result[name] = matches[0]
    return result


def verify_run(profile, directory, profile_id):
    evidence = directory / "evidence.json"
    challenge = directory / "request-data.bin"
    initdata = directory / "initdata.toml"
    if len(challenge.read_bytes()) != 64:
        raise ValueError("Rehearsal nonce must have 64 bytes")
    # Verifier exits unsuccessfully on signature/collateral/RTMR/challenge/initdata failure.
    claims = json.loads(output([verifier(profile), evidence, challenge, initdata]))
    return extract_profile(claims, profile_id), claims


def read_launch(path):
    launch = json.loads(Path(path).read_text())
    expected = hashlib.sha256(json.dumps(launch["launch_inputs"], sort_keys=True).encode()).hexdigest()
    if launch.get("launch_inputs_sha256") != expected:
        raise ValueError("Captured launch fingerprint is stale or inconsistent")
    return launch


def decode_evidence(logs):
    lines = [line[len(FRAME) :] for line in logs.splitlines() if line.startswith(FRAME)]
    if len(lines) != 1 or len(lines[0]) > 32 * 1024 * 1024:
        raise ValueError("Expected exactly one bounded TDX evidence frame")
    evidence = json.loads(base64.b64decode(lines[0], validate=True))
    if not isinstance(evidence, dict) or not evidence.get("quote") or not evidence.get("cc_eventlog"):
        raise ValueError("TDX evidence requires quote and nonempty CCEL")
    return evidence


def tls_registry(directory, address):
    tls = directory / "registry-tls"
    tls.mkdir(mode=0o700)
    (directory / "registry-data").mkdir(mode=0o700)
    run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:3072",
            "-sha256",
            "-nodes",
            "-days",
            "2",
            "-keyout",
            tls / "ca.key",
            "-out",
            tls / "ca.crt",
            "-subj",
            "/CN=CoCo trusted TDX rehearsal CA",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    run(
        [
            "openssl",
            "req",
            "-new",
            "-newkey",
            "rsa:3072",
            "-nodes",
            "-sha256",
            "-keyout",
            tls / "server.key",
            "-out",
            tls / "server.csr",
            "-subj",
            "/CN=" + address,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    write(tls / "server.ext", f"subjectAltName=IP:{address}\nextendedKeyUsage=serverAuth\n")
    run(
        [
            "openssl",
            "x509",
            "-req",
            "-sha256",
            "-days",
            "2",
            "-in",
            tls / "server.csr",
            "-CA",
            tls / "ca.crt",
            "-CAkey",
            tls / "ca.key",
            "-CAcreateserial",
            "-extfile",
            tls / "server.ext",
            "-out",
            tls / "server.crt",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for name in ("ca.crt", "server.crt"):
        (tls / name).chmod(0o644)
    # Skopeo interprets every *.key in its trust directory as a client key.
    # Keep signing/server keys private and supply only the registry trust root.
    certificates = directory / "skopeo-certs"
    certificates.mkdir(mode=0o700)
    shutil.copyfile(tls / "ca.crt", certificates / "ca.crt")
    (certificates / "ca.crt").chmod(0o644)


def start_registry(first, address, name):
    run(
        [
            "docker",
            "run",
            "--detach",
            "--rm",
            "--name",
            name,
            "--network",
            "host",
            "--env",
            "REGISTRY_HTTP_ADDR=" + address,
            "--env",
            "REGISTRY_HTTP_TLS_CERTIFICATE=/tls/server.crt",
            "--env",
            "REGISTRY_HTTP_TLS_KEY=/tls/server.key",
            "--volume",
            str(first / "registry-tls") + ":/tls:ro",
            "--volume",
            str(first / "registry-data") + ":/var/lib/registry",
            REGISTRY,
        ],
        stdout=subprocess.DEVNULL,
    )
    for _ in range(30):
        result = subprocess.run(
            [
                "curl",
                "--fail",
                "--silent",
                "--connect-timeout",
                "5",
                "--max-time",
                "10",
                "--cacert",
                str(first / "registry-tls/ca.crt"),
                "https://" + address + "/v2/",
            ],
            stdout=subprocess.DEVNULL,
            timeout=15,
        )
        if result.returncode == 0:
            return
        time.sleep(1)
    raise RuntimeError("Temporary TLS registry did not become ready")


def make_pod(env, first, namespace, address, image):
    ca = (first / "registry-tls/ca.crt").read_text()
    initdata = (
        'version = "0.1.0"\nalgorithm = "sha256"\n\n[data]\n"cdh.toml" = \'\'\'\n'
        '[kbc]\nname = "offline_fs_kbc"\nurl = ""\n[image]\nextra_root_certificates = ["""'
        + ca
        + '"""]\n[image.registry_config]\nunqualified-search-registries = ["docker.io"]\n'
        '[[image.registry_config.registry]]\nlocation = "' + address + "\"\ninsecure = false\n'''\n"
    )
    write(first / "initdata.toml", initdata)
    workload = yaml.safe_load(Path(env["REHEARSAL_WORKLOAD_YAML"]).read_text())
    spec = workload["spec"]
    if spec.get("runtimeClassName") != env["RUNTIME_CLASS"] or len(spec.get("containers", [])) != 1:
        raise ValueError("Rehearsal needs exactly one container and the selected runtime")
    if spec.get("initContainers") or any(spec.get(x, False) for x in ("hostNetwork", "hostPID", "hostIPC")):
        raise ValueError("Unsupported workload launch setting")
    for key in workload.get("metadata", {}).get("annotations", {}):
        if key.startswith("io.katacontainers.config.") and key != "io.katacontainers.config.hypervisor.cc_init_data":
            raise ValueError("Unreviewed Kata override in workload")
    resources = spec["containers"][0].get("resources", {})
    for key, value in resources.get("limits", {}).items():
        resources.setdefault("requests", {}).setdefault(key, value)
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": "tdx-collector",
            "namespace": namespace,
            "annotations": {
                "io.katacontainers.config.hypervisor.cc_init_data": base64.b64encode(
                    gzip.compress(initdata.encode(), mtime=0)
                ).decode()
            },
        },
        "spec": {
            "runtimeClassName": env["RUNTIME_CLASS"],
            "automountServiceAccountToken": False,
            "enableServiceLinks": False,
            "restartPolicy": "Never",
            "containers": [
                {
                    "name": "collector",
                    "image": image,
                    "imagePullPolicy": "Never",
                    "stdin": False,
                    "tty": False,
                    "securityContext": {"runAsUser": 0, "runAsGroup": 0},
                    "resources": resources,
                    "volumeMounts": [{"name": "challenge", "mountPath": "/challenge", "readOnly": True}],
                }
            ],
            "volumes": [{"name": "challenge", "configMap": {"name": "trusted-challenge", "defaultMode": 256}}],
        },
    }


def pin_collector_cache(tagged_image, digest):
    """Guest-pull only the immutable manifest copied from our local build.

    Kubelet's Never policy needs a local cache name matching that same digest
    reference. The host cache is metadata only; CDH independently fetches and
    checks the digest in the guest through the measured private registry CA.
    """
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError("Invalid immutable collector image digest")
    repository, tag = tagged_image.rsplit(":", 1)
    if not tag or "/" in tag:
        raise ValueError("Expected an explicitly tagged collector image")
    pinned = repository + "@" + digest
    run(["sudo", "ctr", "--namespace", "k8s.io", "images", "tag", tagged_image, pinned])
    return pinned


def collect(config_path, approval_path, repeat=False):
    env, profile = config(config_path)
    approval(profile, approval_path)
    verifier(profile)
    preflight(env, profile)
    first = profile / "rehearsal-collector-build"
    directory = profile / "repeat-rehearsal" if repeat else first
    directory.mkdir(mode=0o700)
    if not repeat:
        write_json(
            profile / "tdx-rehearsal-config.json",
            {
                "config": str(Path(config_path).resolve(strict=True)),
                "approval": str(Path(approval_path).resolve(strict=True)),
            },
        )
    namespace = "coco-tdx-rehearsal-" + secrets.token_hex(8)
    registry = namespace + "-registry"
    kubeconfig = env.get("KUBECONFIG_PATH", "/etc/kubernetes/admin.conf")
    kube = ["kubectl", "--kubeconfig", kubeconfig, "--request-timeout=30s"]
    if not os.access(kubeconfig, os.R_OK):
        kube.insert(0, "sudo")
    run(kube + ["get", "runtimeclass", env["RUNTIME_CLASS"]], stdout=subprocess.DEVNULL)
    # Guest AA accepts runtime_data as UTF-8 text, not base64-decoded bytes.
    # 32 random bytes encoded as hex supply 256 bits of freshness in 64 ABI bytes.
    challenge = secrets.token_hex(32).encode("ascii")
    (directory / "request-data.bin").write_bytes(challenge)
    created_namespace = False
    try:
        if repeat:
            pod = yaml.safe_load((first / "collector-pod.yaml").read_text())
            pod["metadata"]["namespace"] = namespace
            address = pod["spec"]["containers"][0]["image"].split("/")[0]
            shutil.copyfile(first / "initdata.toml", directory / "initdata.toml")
        else:
            address = json.loads(output(["ip", "-j", "-4", "route", "get", "1.1.1.1"]))[0]["prefsrc"]
            tls_registry(first, address)
            address += ":5443"
        start_registry(first, address, registry)
        if not repeat:
            # This directory also holds registry private keys and evidence.
            # Never send those files to the Docker builder or its build cache.
            write(first / ".dockerignore", "*\n!Dockerfile\n!collect-tdx-evidence.py\n")
            for source, target in (
                ("Dockerfile.tdx", "Dockerfile"),
                ("collect-tdx-evidence.py", "collect-tdx-evidence.py"),
            ):
                shutil.copyfile(HERE / "rehearsal-collector" / source, first / target)
            local_image = "localhost/coco-tdx-rehearsal:" + namespace
            run(["docker", "build", "--pull", "--tag", local_image, first])
            image = address + "/coco-tdx-rehearsal:" + namespace
            run(
                [
                    "skopeo",
                    "copy",
                    "--digestfile",
                    first / "collector-manifest.digest",
                    "--dest-cert-dir",
                    first / "skopeo-certs",
                    "docker-daemon:" + local_image,
                    "docker://" + image,
                ]
            )
            digest = (first / "collector-manifest.digest").read_text().strip()
            published_digest = output(
                [
                    "skopeo",
                    "inspect",
                    "--cert-dir",
                    first / "skopeo-certs",
                    "--format",
                    "{{.Digest}}",
                    "docker://" + image,
                ]
            )
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest) or published_digest != digest:
                raise ValueError("Registry image differs from the immutable manifest copied from our local build")
            write_json(
                first / "collector-image.json",
                {
                    "image": image,
                    "digest": digest,
                    "image_id": output(["docker", "image", "inspect", "--format", "{{.Id}}", local_image]),
                },
            )
            run(["docker", "tag", local_image, image])
            run(["docker", "save", "--output", first / "collector-host-cache.tar", image])
            run(["sudo", "ctr", "--namespace", "k8s.io", "images", "import", first / "collector-host-cache.tar"])
            pinned_image = pin_collector_cache(image, digest)
            pod = make_pod(env, first, namespace, address, pinned_image)
        pod_path = directory / "collector-pod.yaml"
        write(pod_path, yaml.safe_dump(pod, sort_keys=False))
        run(kube + ["create", "namespace", namespace])
        created_namespace = True
        run(
            kube
            + [
                "-n",
                namespace,
                "create",
                "configmap",
                "trusted-challenge",
                "--from-file=request-data.bin=" + str(directory / "request-data.bin"),
            ]
        )
        run(kube + ["apply", "-f", pod_path])
        deadline = time.monotonic() + 600
        captured = False
        while time.monotonic() < deadline:
            obj = json.loads(output(kube + ["-n", namespace, "get", "pod", "tdx-collector", "-o", "json"]))
            phase = obj["status"].get("phase")
            if phase == "Running" and not captured:
                result = subprocess.run(
                    [
                        "sudo",
                        "python3",
                        str(HERE / "capture-running-launch.py"),
                        namespace,
                        "tdx-collector",
                        "/opt/kata/share/defaults/kata-containers/" + RUNTIMES[env["RUNTIME_CLASS"]],
                        str(directory / "actual-launch.json"),
                        kubeconfig,
                    ],
                    timeout=60,
                )
                if result.returncode not in (0, 75):
                    raise RuntimeError("TDX actual launch capture failed")
                if result.returncode == 0:
                    run(["sudo", "chown", f"{os.getuid()}:{os.getgid()}", directory / "actual-launch.json"])
                    captured = True
            if phase == "Succeeded":
                break
            if phase == "Failed":
                raise RuntimeError("TDX collector Pod failed; retaining local diagnostics before cleanup")
            time.sleep(2)
        else:
            raise RuntimeError("Timed out waiting for TDX collector")
        if not captured:
            raise RuntimeError("Actual TDX launch was not captured")
        logs = output(kube + ["-n", namespace, "logs", "tdx-collector"])
        write(directory / "collector.log", logs + "\n")
        write_json(directory / "evidence.json", decode_evidence(logs))
        candidate, claims = verify_run(profile, directory, env["PLATFORM_PROFILE"])
        write_json(directory / "verified-claims.json", claims)
        if repeat:
            initial, _ = verify_run(profile, first, env["PLATFORM_PROFILE"])
            before = read_launch(first / "actual-launch.json")
            after = read_launch(directory / "actual-launch.json")
            if (
                initial != candidate
                or before["launch_inputs"] != after["launch_inputs"]
                or before["pod_resources"] != after["pod_resources"]
            ):
                raise ValueError("TDX measurements or launch profile did not reproduce")
            if challenge == (first / "request-data.bin").read_bytes():
                raise ValueError("Repeat nonce must differ")
            write_json(profile / "candidate-tdx-profile.json", candidate)
            print("Verified repeated candidate SHA256: " + sha(profile / "candidate-tdx-profile.json"))
            print("Review runtime artifacts and candidate; set APPROVED_TDX_PROFILE_SHA256 before stage 09.")
        else:
            print("TDX quote/CCEL verified; run stage 08 with a fresh nonce. Nothing approved yet.")
    except BaseException:
        # Keep useful failure diagnostics on the trusted machine before cleanup,
        # without leaving an unprotected collector Pod running on error.
        if created_namespace:
            diagnostics = (
                ("failed-pod.json", ["get", "pod", "tdx-collector", "-o", "json"]),
                ("failed-collector.log", ["logs", "tdx-collector", "--limit-bytes=1048576"]),
                ("failed-events.txt", ["get", "events", "--sort-by=.lastTimestamp"]),
            )
            for name, command in diagnostics:
                try:
                    result = subprocess.run(
                        kube + ["-n", namespace] + command, capture_output=True, text=True, timeout=35
                    )
                    write(directory / name, result.stdout + result.stderr)
                except (OSError, subprocess.SubprocessError):
                    pass
        raise
    finally:
        if created_namespace:
            try:
                subprocess.run(kube + ["delete", "namespace", namespace, "--wait=false"], check=False, timeout=35)
            except subprocess.SubprocessError:
                print("Warning: namespace cleanup timed out: " + namespace)
        try:
            subprocess.run(["docker", "rm", "--force", registry], check=False, stdout=subprocess.DEVNULL, timeout=35)
        except subprocess.SubprocessError:
            print("Warning: registry cleanup timed out: " + registry)


def finalize(config_path, approval_path):
    env, profile = config(config_path)
    approved = approval(profile, approval_path)
    candidate_path = profile / "candidate-tdx-profile.json"
    if approved.get("APPROVED_TDX_PROFILE_SHA256") != sha(candidate_path):
        raise ValueError("Trusted authority must explicitly approve this candidate's exact SHA256")
    initial, _ = verify_run(profile, profile / "rehearsal-collector-build", env["PLATFORM_PROFILE"])
    repeated, _ = verify_run(profile, profile / "repeat-rehearsal", env["PLATFORM_PROFILE"])
    if initial != repeated or initial != json.loads(candidate_path.read_text()):
        raise ValueError("Approved candidate does not match both newly verified runs")
    first = profile / "rehearsal-collector-build"
    repeat = profile / "repeat-rehearsal"
    if (first / "request-data.bin").read_bytes() == (repeat / "request-data.bin").read_bytes():
        raise ValueError("Repeated nonce is not fresh")
    pinned = profile / (profile / "kata-config-relative-path.txt").read_text().strip()
    installed = "/opt/kata/share/defaults/kata-containers/" + RUNTIMES[env["RUNTIME_CLASS"]]
    run(["sha256sum", "--check", "--strict", profile / "kata-artifacts.sha256"], stdout=subprocess.DEVNULL)
    for directory in (first, repeat):
        run(
            [
                "python3",
                HERE / "lib/kata-runtime-profile.py",
                "verify",
                pinned,
                profile / "approved-kata-config.toml",
                profile / "kata-runtime-profile.json",
                "--installed",
                installed,
                "--launch",
                directory / "actual-launch.json",
            ]
        )
        launch = read_launch(directory / "actual-launch.json")
        approved_launch = json.loads((profile / "approved-launch-profile.json").read_text())
        runpy.run_path(str(HERE / "lib/workload-security-context.py"))["validate_actual_launch"](
            approved_launch, launch, Path(env["REHEARSAL_WORKLOAD_YAML"])
        )
    before = read_launch(first / "actual-launch.json")
    after = read_launch(repeat / "actual-launch.json")
    if before["launch_inputs"] != after["launch_inputs"] or before["pod_resources"] != after["pod_resources"]:
        raise ValueError("Actual launch profile did not reproduce")
    values = {"schema": "coco-platform-reference-values/v2", "tee": "tdx", "profiles": [initial]}
    runpy.run_path(str(HERE / "lib/platform-reference-schema.py"))["validate_values"](values)
    write_json(profile / "approved-tdx-reference-values.json", values)
    env["TDX_REFERENCE_VALUES_SHA256"] = sha(profile / "approved-tdx-reference-values.json")
    env["APPROVED_WORKLOAD_PROFILE_SHA256"] = sha(profile / "approved-launch-profile.json")
    env["APPROVED_ACTUAL_LAUNCH_SHA256"] = sha(first / "actual-launch.json")
    write(
        profile / "platform-reference.final.env",
        "# Verified and explicitly approved TDX profile\n" + "".join(f"{k}={shlex.quote(v)}\n" for k, v in env.items()),
    )
    print("TDX references finalized; stage 10 exports only the public approved profile.")


def export(config_path, values_path, launch_path=None):
    env, profile = config(config_path)
    if Path(config_path).resolve() != (profile / "platform-reference.final.env").resolve():
        raise ValueError("Use stage 09 final environment")
    source = profile / "approved-tdx-reference-values.json"
    if sha(source) != env.get("TDX_REFERENCE_VALUES_SHA256"):
        raise ValueError("Approved TDX references changed after finalization")
    payloads = [(Path(values_path), source.read_text())]
    if launch_path:
        launch = output(
            [
                "python3",
                HERE / "export-workload-launch-profile.py",
                profile,
                env["PLATFORM_PROFILE"],
                env["KATA_VERSION"],
                env["RUNTIME_CLASS"],
                env["KATA_DEPLOY_AMD64"],
                env["APPROVED_WORKLOAD_PROFILE_SHA256"],
                env["APPROVED_ACTUAL_LAUNCH_SHA256"],
            ]
        )
        payloads.append((Path(launch_path), launch + "\n"))
    if len({str(p.resolve()) for p, _ in payloads}) != len(payloads):
        raise ValueError("Secure-services and provisioning outputs must differ")
    for path, _ in payloads:
        if path.exists() or path.is_symlink() or not path.parent.is_dir():
            raise ValueError("Output exists or parent missing: " + str(path))
    created = []
    try:
        for path, data in payloads:
            with path.open("x", encoding="utf-8") as stream:
                created.append(path)
                stream.write(data)
    except BaseException:
        for path in created:
            path.unlink()
        raise
    print("Exported approved TDX profile to " + str(values_path))


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "collect", "repeat", "finalize", "export"))
    parser.add_argument("config")
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    if args.operation == "prepare":
        prepare(args.config)
    elif args.operation in ("collect", "repeat"):
        collect(args.config, *args.arguments, repeat=args.operation == "repeat")
    elif args.operation == "finalize":
        finalize(args.config, *args.arguments)
    else:
        export(args.config, *args.arguments)


if __name__ == "__main__":
    main()
