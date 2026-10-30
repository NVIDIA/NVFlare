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
import runpy
import subprocess
import sys
import tomllib
from pathlib import Path

RUNTIME = runpy.run_path(str(Path(__file__).resolve().parent / "lib/kata-runtime-profile.py"))
SECURITY = runpy.run_path(str(Path(__file__).resolve().parent / "lib/workload-security-context.py"))


def confidential_guest(argv, runtime):
    """Validate both pinned QEMU encodings, without substring TEE detection."""
    target = RUNTIME["runtime_target"](runtime)
    guests = []
    for index, value in enumerate(argv):
        if value != "-object":
            continue
        if index + 1 >= len(argv):
            raise ValueError("Missing QEMU object value")
        raw = argv[index + 1]
        if raw.startswith("{"):

            def unique(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError("Duplicate QEMU object key")
                    result[key] = value
                return result

            obj = json.loads(raw, object_pairs_hook=unique)
            if not isinstance(obj, dict):
                raise ValueError("Malformed QEMU object")
        else:
            fields = raw.split(",")
            obj = {"qom-type": fields[0]}
            for field in fields[1:]:
                key, separator, value = field.partition("=")
                if not separator or key in obj:
                    raise ValueError("Unsupported or ambiguous QEMU object encoding")
                obj[key] = value
        if obj.get("qom-type") in ("sev-guest", "sev-snp-guest", "tdx-guest"):
            guests.append(obj)
    expected = "sev-snp-guest" if target["cpu_tee"] == "snp" else "tdx-guest"
    if len(guests) != 1 or guests[0].get("qom-type") != expected:
        raise ValueError("Actual QEMU confidential guest does not match approved RuntimeClass")
    guest = guests[0]
    machines = [argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == "-machine"]
    if len(machines) != 1 or not guest.get("id"):
        raise ValueError("Missing explicit confidential machine/guest identifier")
    machine_fields = {}
    for field in machines[0].split(",")[1:]:
        key, separator, value = field.partition("=")
        if not separator or key in machine_fields:
            raise ValueError("Unsupported or duplicate QEMU machine field")
        machine_fields[key] = value
    if machine_fields.get("confidential-guest-support") != guest["id"]:
        raise ValueError("QEMU machine does not select the approved confidential guest")
    if target["cpu_tee"] == "tdx":
        debug = guest.get("debug", False)
        if not (debug is False or (type(debug) is str and debug in ("off", "false"))):
            raise ValueError("TDX debug is not approved")
    # InitData differs per workload and is independently authenticated in
    # attestation. QGS is an untrusted transport, not a platform measurement.
    # Preserve the exact original arguments separately for the private audit.
    stable = {
        key: value for key, value in guest.items() if key not in ("mrconfigid", "host-data", "quote-generation-socket")
    }
    return target, stable


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


def capture(namespace, pod, config, kubeconfig="/etc/kubernetes/admin.conf"):
    kube = ["kubectl", "--kubeconfig", kubeconfig]
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
    target, guest = confidential_guest(argv, obj["spec"].get("runtimeClassName"))
    RUNTIME["require_confidential_config"](qemu, obj["spec"]["runtimeClassName"])
    stable = {
        "artifacts": {k: v["sha256"] for k, v in hashes.items()},
        "cpu": option("-cpu"),
        "smp": option("-smp"),
        "memory": option("-m"),
        "machine": option("-machine"),
        "kernel_command_line": option("-append"),
        "confidential_guest": guest,
    }
    if not stable["smp"] or not stable["kernel_command_line"]:
        raise ValueError("Actual launch lacks explicit CPU topology or kernel command line")
    return {
        "schema": 1,
        "cpu_tee": target["cpu_tee"],
        "gpu": target["gpu"],
        "pod_uid": uid,
        "sandbox_ids": ids,
        "qemu_pid": int(process.name),
        "qemu_argv": argv,
        "artifacts": hashes,
        "launch_inputs": stable,
        "launch_inputs_sha256": hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest(),
        "pod_resources": [
            {kind: values for kind, values in SECURITY["normalize_resources"](c.get("resources", {})).items() if values}
            for c in obj["spec"]["containers"]
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("namespace")
    parser.add_argument("pod")
    parser.add_argument("config")
    parser.add_argument("output")
    parser.add_argument("kubeconfig", nargs="?", default="/etc/kubernetes/admin.conf")
    args = parser.parse_args()
    os.umask(0o077)
    result = capture(args.namespace, args.pod, args.config, args.kubeconfig)
    with Path(args.output).open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print("Captured actual QEMU launch:", result["launch_inputs_sha256"])
