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

"""Capture the actual QEMU launch for one trusted rehearsal Pod. Run as root."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rootfs_image(argv):
    """Resolve Kata's rootfs drive, never a configured-but-unused image path.

    This profile supports Kata's file-backed -drive id=image[-...]. Other
    rootfs encodings fail closed when no actual initrd is present.
    """
    images = []
    for index, arg in enumerate(argv):
        if arg != "-drive":
            continue
        options = argv[index + 1].split(",")
        fields = {}
        for option in options:
            key, separator, value = option.partition("=")
            if not separator or key in fields:
                raise ValueError("Unsupported or ambiguous QEMU drive encoding")
            fields[key] = value
        drive_id = fields.get("id", "")
        if drive_id == "image" or drive_id.startswith("image-"):
            if not fields.get("file"):
                raise ValueError("Rootfs drive has no explicit file")
            images.append(fields["file"])
    if len(images) > 1:
        raise ValueError("Ambiguous QEMU rootfs drives")
    return images[0] if images else None


def capture(namespace, pod, config):
    kube = ["kubectl", "--kubeconfig", "/etc/kubernetes/admin.conf"]
    obj = json.loads(subprocess.check_output(kube + ["-n", namespace, "get", "pod", pod, "-o", "json"]))
    uid = obj["metadata"]["uid"]
    cri = ["crictl", "--runtime-endpoint=unix:///run/containerd/containerd.sock"]
    sandboxes = json.loads(subprocess.check_output(cri + ["pods", "-o", "json"]))
    ids = [s["id"] for s in sandboxes.get("items", []) if s.get("labels", {}).get("io.kubernetes.pod.uid") == uid]
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
        except (OSError, UnicodeError):
            continue
        if argv and Path(argv[0]).name.startswith("qemu-system-") and any(s in "\0".join(argv) for s in ids):
            matches.append((entry, argv))
    if not matches:
        sys.exit(75)  # Retry while the Pod is starting.
    if len(matches) != 1:
        raise ValueError("Expected exactly one QEMU for the target Pod")
    process, argv = matches[0]

    def option(name):
        indexes = [i for i, value in enumerate(argv) if value == name]
        if len(indexes) > 1:
            raise ValueError(f"Ambiguous QEMU argument: {name}")
        return argv[indexes[0] + 1] if indexes else None

    with Path(config).open("rb") as stream:
        settings = tomllib.load(stream)
    qemu = settings["hypervisor"]["qemu"]
    files = {"qemu_executable": str((process / "exe").resolve()), "kata_config": str(Path(config).resolve())}
    for key, flag in [("kernel", "-kernel"), ("initrd", "-initrd"), ("firmware", "-bios")]:
        actual = option(flag)
        if actual:
            files[key] = actual
    image = rootfs_image(argv)
    if image:
        files["image"] = image
    if "initrd" not in files and "image" not in files:
        raise ValueError("No actual initrd or supported Kata rootfs drive was captured")
    for key in ["firmware", "kernel", "initrd", "image"]:
        if qemu.get(key):
            files["configured_" + key] = qemu[key]
    hashes = {key: {"path": value, "sha256": digest(value)} for key, value in files.items()}
    objects = [argv[i + 1] for i, value in enumerate(argv) if value == "-object" and i + 1 < len(argv)]
    if not any("sev-snp-guest" in value for value in objects):
        raise ValueError("Actual QEMU launch is not SEV-SNP")
    stable = {
        "artifacts": {k: v["sha256"] for k, v in hashes.items()},
        "cpu": option("-cpu"),
        "smp": option("-smp"),
        "memory": option("-m"),
        "machine": option("-machine"),
        "kernel_command_line": option("-append"),
    }
    if not stable["smp"] or not stable["kernel_command_line"]:
        raise ValueError("Actual launch lacks explicit CPU topology or kernel command line")
    return {
        "schema": 1,
        "pod_uid": uid,
        "sandbox_ids": ids,
        "qemu_pid": int(process.name),
        "qemu_argv": argv,
        "artifacts": hashes,
        "launch_inputs": stable,
        "launch_inputs_sha256": hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest(),
        "pod_resources": [c.get("resources", {}) for c in obj["spec"]["containers"]],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("namespace")
    parser.add_argument("pod")
    parser.add_argument("config")
    parser.add_argument("output")
    args = parser.parse_args()
    os.umask(0o077)
    result = capture(args.namespace, args.pod, args.config)
    with Path(args.output).open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print("Captured actual QEMU launch:", result["launch_inputs_sha256"])
