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

"""Repeat the trusted one-GPU launch with a fresh nonce; require equal signed measurement."""

import base64
import hashlib
import io
import json
import os
import secrets
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import yaml

os.umask(0o077)
profile = Path(sys.argv[1]).resolve(strict=True)
first = profile / "rehearsal-collector-build"
out = profile / "repeat-rehearsal"
out.mkdir(mode=0o700)
pod = yaml.safe_load((first / "collector-pod.yaml").read_text())
run_id = secrets.token_hex(8)
namespace = "coco-repeat-" + run_id
pod["metadata"]["namespace"] = namespace
pod["metadata"]["name"] = "collector-" + run_id
pod_path = out / "pod.yaml"
pod_path.write_text(yaml.safe_dump(pod, sort_keys=False))
challenge = secrets.token_bytes(64)
(out / "request-data.bin").write_bytes(challenge)
registry_host = pod["spec"]["containers"][0]["image"].split("/")[0]
registry = "coco-repeat-registry-" + run_id
kube = ["kubectl", "--kubeconfig", "/etc/kubernetes/admin.conf"]
if not os.access("/etc/kubernetes/admin.conf", os.R_OK):
    kube.insert(0, "sudo")


def run(command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


created_registry = created_namespace = False
try:
    run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            registry,
            "--network",
            "host",
            "-e",
            "REGISTRY_HTTP_ADDR=" + registry_host,
            "-e",
            "REGISTRY_HTTP_TLS_CERTIFICATE=/tls/server.crt",
            "-e",
            "REGISTRY_HTTP_TLS_KEY=/tls/server.key",
            "-v",
            str(first / "registry-tls") + ":/tls:ro",
            "-v",
            str(first / "registry-data") + ":/var/lib/registry",
            "docker.io/library/registry@sha256:a3d8aaa63ed8681a604f1dea0aa03f100d5895b6a58ace528858a7b332415373",
        ],
        stdout=subprocess.DEVNULL,
    )
    created_registry = True
    for attempt in range(30):
        check = subprocess.run(
            [
                "curl",
                "--fail",
                "--silent",
                "--cacert",
                str(first / "registry-tls/ca.crt"),
                "https://" + registry_host + "/v2/",
            ],
            stdout=subprocess.DEVNULL,
        )
        if check.returncode == 0:
            break
        time.sleep(1)
    else:
        raise RuntimeError("Repeat TLS registry did not start")
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
            "--from-file=request-data.bin=" + str(out / "request-data.bin"),
        ]
    )
    run(kube + ["apply", "-f", str(pod_path)])
    deadline = time.monotonic() + 600
    captured = False
    while time.monotonic() < deadline:
        obj = json.loads(
            subprocess.check_output(kube + ["-n", namespace, "get", "pod", pod["metadata"]["name"], "-o", "json"])
        )
        phase = obj["status"].get("phase")
        if phase == "Running" and not captured:
            result = subprocess.run(
                [
                    "sudo",
                    "python3",
                    str(Path(__file__).with_name("capture-running-launch.py")),
                    namespace,
                    pod["metadata"]["name"],
                    "/opt/kata/share/defaults/kata-containers/configuration-qemu-nvidia-gpu-snp.toml",
                    str(out / "actual-launch.json"),
                ]
            )
            if result.returncode not in (0, 75):
                raise RuntimeError("Repeat actual-launch capture failed")
            if result.returncode == 0:
                run(["sudo", "chown", f"{os.getuid()}:{os.getgid()}", str(out / "actual-launch.json")])
                captured = True
        if phase == "Succeeded":
            break
        if phase == "Failed":
            raise RuntimeError("Repeat collector failed")
        time.sleep(2)
    else:
        raise RuntimeError("Repeat collector timed out")
    if not captured:
        raise RuntimeError("Missing actual repeat launch capture")
    logs = subprocess.check_output(kube + ["-n", namespace, "logs", pod["metadata"]["name"]], text=True)
    (out / "collector.log").write_text(logs)
    payloads = [line.split("=", 1)[1] for line in logs.splitlines() if line.startswith("COCO_SNP_EVIDENCE_V1=")]
    if len(payloads) != 1:
        raise RuntimeError("Expected one repeat evidence frame")
    archive = base64.b64decode(payloads[0], validate=True)
    expected = {"SHA256SUMS", "attestation-report.bin", "attestation-report.txt", "request-data.bin"}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        members = tar.getmembers()
        if len(members) != 4 or {m.name for m in members} != expected or any(not m.isfile() for m in members):
            raise RuntimeError("Unexpected repeat evidence archive contents")
        for member in members:
            data = tar.extractfile(member).read()
            if member.name == "request-data.bin":
                if data != challenge:
                    raise RuntimeError("Repeat evidence challenge mismatch")
            else:
                (out / member.name).write_bytes(data)
    run(["sha256sum", "--check", "--strict", "SHA256SUMS"], cwd=out)
    snpguest = first / "snpguest"
    certs = profile / "reported-tcb-evidence/certs"
    report_path = out / "attestation-report.bin"
    run([str(snpguest), "verify", "certs", str(certs)])
    run([str(snpguest), "verify", "attestation", str(certs), str(report_path)])
    run([str(snpguest), "verify", "attestation", str(certs), str(report_path), "--tcb"])
    report = report_path.read_bytes()
    initial = (first / "evidence-input/attestation-report.bin").read_bytes()
    if len(report) < 0x188 or report[0x50:0x90] != challenge:
        raise RuntimeError("Signed repeat challenge mismatch")
    if report[0x90:0xC0] != initial[0x90:0xC0] or report[0x180:0x188] != initial[0x180:0x188]:
        raise RuntimeError("Fresh signed measurement or reported TCB did not reproduce")
    before = json.loads((first / "actual-launch.json").read_text())
    after = json.loads((out / "actual-launch.json").read_text())
    if before["launch_inputs"] != after["launch_inputs"] or before["pod_resources"] != after["pod_resources"]:
        raise RuntimeError("Captured launch profile did not reproduce")
    summary = {
        "measurement": report[0x90:0xC0].hex(),
        "reported_tcb_hex": report[0x180:0x188].hex(),
        "launch_inputs_sha256": after["launch_inputs_sha256"],
        "nonce_sha256": hashlib.sha256(challenge).hexdigest(),
        "first_report_sha256": hashlib.sha256(initial).hexdigest(),
        "repeat_report_sha256": hashlib.sha256(report).hexdigest(),
        "result": "PASS: fresh signed reports and captured launch profile match",
    }
    (out / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
finally:
    if created_namespace:
        subprocess.run(kube + ["delete", "namespace", namespace, "--wait=false"], check=False)
    if created_registry:
        subprocess.run(["docker", "rm", "-f", registry], check=False, stdout=subprocess.DEVNULL)
