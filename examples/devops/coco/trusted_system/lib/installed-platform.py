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

"""Inspect an existing worker during a trusted platform-owner reference ceremony.

The worker must be under the platform owner's control during capture and
verification. Machine identity and file hashes are diagnostics, not approval
of a host actively controlled by an adversary.
"""

import hashlib
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG_KEYS = {
    "NODE_NAME": ("node", "name"),
    "RUNTIME_CLASS": ("runtime_class",),
    "INSTALLED_KATA_CONFIG": ("installed_config",),
    "KATA_RUNTIME": ("kata_runtime",),
    "KUBECONFIG_PATH": ("kubeconfig",),
    "COLLECTOR_IMAGE": ("collector_image",),
    "COLLECTOR_METADATA_FILE": ("collector_metadata_file",),
    "COLLECTOR_METADATA_SHA256": ("collector_metadata_sha256",),
    "COLLECTOR_REGISTRY_CA": ("registry_ca",),
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def kube_command(snapshot):
    command = ["kubectl", "--kubeconfig", snapshot["kubeconfig"], "--request-timeout=30s"]
    if not os.access(snapshot["kubeconfig"], os.R_OK):
        command.insert(0, "sudo")
    return command


def verify_worker(snapshot, kube=None):
    """Require the inspected worker, its Kubernetes UID, and the local machine identity."""
    node = json.loads(
        subprocess.check_output(
            (kube or kube_command(snapshot)) + ["get", "node", snapshot["node"]["name"], "-o", "json"], text=True
        )
    )
    actual = {
        "name": node["metadata"]["name"],
        "uid": node["metadata"]["uid"],
        "machine_id": node["status"]["nodeInfo"]["machineID"],
        "addresses": sorted(node["status"].get("addresses", []), key=lambda value: (value["type"], value["address"])),
    }
    if actual != snapshot["node"]:
        raise ValueError("Selected worker identity or addresses changed after runtime inspection")
    machine_id = Path("/etc/machine-id").read_text().strip()
    if not machine_id or actual["machine_id"] != machine_id:
        raise ValueError("Run baseline capture on the platform-owner-controlled selected worker")
    addresses = json.loads(subprocess.check_output(["ip", "-j", "address", "show"], text=True))
    local = {entry["local"] for iface in addresses for entry in iface.get("addr_info", [])}
    expected = {entry["address"] for entry in actual["addresses"] if entry["type"] in ("InternalIP", "ExternalIP")}
    if not expected.intersection(local):
        raise ValueError("Selected worker has no address on this local host")
    return actual


def check_pod_node(pod, snapshot):
    if pod.get("spec", {}).get("nodeName") != snapshot["node"]["name"]:
        raise ValueError("Collector Pod was not assigned to the inspected worker")


def verify_launch(launch, snapshot, profile):
    if (
        launch.get("node_name") != snapshot["node"]["name"]
        or launch.get("node_uid") != snapshot["node"]["uid"]
        or launch.get("platform_snapshot_sha256") != digest(Path(profile) / "installed-platform.json")
    ):
        raise ValueError("Captured launch does not match the inspected worker and runtime snapshot")


def read_config(path, keys):
    script = "set -ae; unset " + " ".join(keys) + '; source "$1"; python3 -c "$2"'
    code = "import json,os; print(json.dumps({k:os.environ[k] for k in " + repr(keys) + " if k in os.environ}))"
    return json.loads(subprocess.check_output(["bash", "-c", script, "installed-platform", str(path), code], text=True))


def runtime_settings(raw, runtime):
    config = tomllib.loads(raw.decode())
    qemu = config["hypervisor"]["qemu"]
    helper = runpy.run_path(str(HERE / "kata-runtime-profile.py"))
    helper["require_confidential_config"](qemu, runtime)
    helper["require_token_api"](qemu.get("kernel_params"))
    for key in ("default_vcpus", "default_memory"):
        if type(qemu.get(key)) is not int or qemu[key] <= 0:
            raise ValueError("Installed runtime needs an explicit positive " + key)
    for key in ("path", "firmware", "kernel"):
        if not qemu.get(key):
            raise ValueError("Installed runtime has no explicit " + key + " path")
    if not qemu.get("initrd") and not qemu.get("image"):
        raise ValueError("Installed runtime needs an initrd or rootfs image")
    return config


def verify_snapshot(profile, env=None):
    profile = Path(profile)
    snapshot = json.loads((profile / "installed-platform.json").read_text())
    if env is not None:
        for key, fields in CONFIG_KEYS.items():
            expected = snapshot
            for field in fields:
                expected = expected[field]
            supplied = env.get(key)
            if key == "COLLECTOR_METADATA_SHA256" and supplied in (None, ""):
                continue
            if supplied != expected:
                raise ValueError("Configuration changed after runtime inspection: " + key)
    verify_worker(snapshot)
    for path, expected in snapshot["files"].items():
        if digest(path) != expected:
            raise ValueError("Installed platform input changed after inspection: " + path)
    with Path(snapshot["installed_config"]).open("rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024 or hashlib.sha256(raw).hexdigest() != snapshot["files"][snapshot["installed_config"]]:
        raise ValueError("Installed runtime configuration changed during verification")
    runtime_settings(raw, snapshot["runtime_class"])
    return snapshot


def inspect_platform(profile, env):
    """Snapshot the ready worker's installed configuration and actual local files."""
    profile = Path(profile)
    worker = env["NODE_NAME"]
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,252}", worker):
        raise ValueError("NODE_NAME must be an explicit Kubernetes worker name")
    values = {
        "installed_config": env["INSTALLED_KATA_CONFIG"],
        "kata_runtime": env["KATA_RUNTIME"],
        "kubeconfig": env["KUBECONFIG_PATH"],
        "runtime_class": env["RUNTIME_CLASS"],
        "collector_image": env["COLLECTOR_IMAGE"],
        "collector_metadata_file": env["COLLECTOR_METADATA_FILE"],
        "registry_ca": env["COLLECTOR_REGISTRY_CA"],
    }
    for name in ("installed_config", "kata_runtime", "kubeconfig", "registry_ca"):
        if not Path(values[name]).is_absolute() or not Path(values[name]).is_file():
            raise ValueError(name + " must name an existing absolute file")
    with Path(values["installed_config"]).open("rb") as stream:
        config_raw = stream.read(1024 * 1024 + 1)
    if len(config_raw) > 1024 * 1024:
        raise ValueError("Installed runtime configuration exceeds the size limit")
    config = runtime_settings(config_raw, env["RUNTIME_CLASS"])
    qemu = config["hypervisor"]["qemu"]
    target = runpy.run_path(str(HERE / "kata-runtime-profile.py"))["runtime_target"](env["RUNTIME_CLASS"])
    verifier = Path(env["SNP_VERIFIER" if target["cpu_tee"] == "amd_sev_snp" else "TDX_VERIFIER"])
    if not verifier.is_absolute() or not verifier.is_file() or not os.access(verifier, os.X_OK):
        raise ValueError("Configure an existing reviewed executable CPU evidence verifier")
    metadata, checksum, metadata_raw = load_collector(env, with_bytes=True)
    node = json.loads(subprocess.check_output(kube_command(values) + ["get", "node", worker, "-o", "json"], text=True))
    values["node"] = {
        "name": node["metadata"]["name"],
        "uid": node["metadata"]["uid"],
        "machine_id": node["status"]["nodeInfo"]["machineID"],
        "addresses": sorted(node["status"].get("addresses", []), key=lambda item: (item["type"], item["address"])),
    }
    verify_worker(values)
    helper = runpy.run_path(str(HERE / "kata-runtime-profile.py"))
    actual_env = json.loads(
        subprocess.check_output(
            [values["kata_runtime"], "--config", values["installed_config"], "kata-env", "--json"], text=True
        )
    )
    effective = actual_env.get("Kernel", {}).get("Parameters", "")
    helper["require_token_api"](effective)
    if any(token not in effective.split() for token in qemu["kernel_params"].split()):
        raise ValueError("Effective runtime parameters omit a configured parameter")
    files = {
        values["installed_config"]: hashlib.sha256(config_raw).hexdigest(),
        values["kata_runtime"]: digest(values["kata_runtime"]),
        values["registry_ca"]: digest(values["registry_ca"]),
        env["COLLECTOR_METADATA_FILE"]: checksum,
        str(verifier): digest(verifier),
    }
    for key in ("path", "firmware", "firmware_volume", "kernel", "initrd", "image"):
        path = qemu.get(key)
        if path:
            if not Path(path).is_absolute() or not Path(path).is_file():
                raise ValueError("Installed runtime artifact must be an existing absolute file: " + key)
            files[path] = digest(path)
    subprocess.run(["openssl", "x509", "-in", values["registry_ca"], "-noout"], check=True)
    copy = profile / "collector-metadata.json"
    with copy.open("xb") as stream:
        stream.write(metadata_raw)
    copy.chmod(0o600)
    files[str(copy)] = checksum
    values.update(
        files=files,
        collector_metadata_sha256=checksum,
        collector_metadata_snapshot=str(copy),
        verifier=str(verifier),
        config=config,
    )
    version_text = subprocess.check_output([values["kata_runtime"], "--version"], text=True)
    version = re.search(
        r"(?im)^kata-runtime\s*:?\s*(?:version\s*)?(\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?)(?:\s|$)", version_text
    )
    if version is None or version.group(1) != "3.29.0":
        raise ValueError(
            "Existing runtime must be Kata 3.29.0, compatible with this example's genpolicy tool; ask the platform owner"
        )
    values["runtime_version"] = version.group(1)
    with (profile / "installed-platform.json").open("x") as stream:
        json.dump(values, stream, indent=2)
        stream.write("\n")
    return values


def publish_collector(local_image, tagged_image, directory, env):
    """Publish with operator credentials, then pin an anonymously readable immutable manifest."""
    directory = Path(directory)
    certs = directory / "skopeo-certs"
    certs.mkdir(mode=0o700, exist_ok=True)
    shutil.copyfile(env["COLLECTOR_REGISTRY_CA"], certs / "ca.crt")
    auth = Path(env["COLLECTOR_REGISTRY_AUTH_FILE"])
    if not auth.is_absolute() or not auth.is_file() or auth.stat().st_mode & 0o077:
        raise ValueError("COLLECTOR_REGISTRY_AUTH_FILE must be a private existing absolute authfile")
    anonymous = directory / "anonymous-auth.json"
    anonymous.write_text('{"auths":{}}\n')
    digestfile = directory / "collector-manifest.digest"
    subprocess.run(
        [
            "skopeo",
            "copy",
            "--digestfile",
            str(digestfile),
            "--dest-tls-verify=true",
            "--dest-cert-dir",
            str(certs),
            "--dest-authfile",
            str(auth),
            "docker-daemon:" + local_image,
            "docker://" + tagged_image,
        ],
        check=True,
        stdout=sys.stderr,
    )
    expected = digestfile.read_text().strip()
    actual = subprocess.check_output(
        [
            "skopeo",
            "inspect",
            "--tls-verify=true",
            "--authfile",
            str(anonymous),
            "--cert-dir",
            str(certs),
            "--format",
            "{{.Digest}}",
            "docker://" + tagged_image,
        ],
        text=True,
    ).strip()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", expected) or actual != expected:
        raise ValueError("Published collector manifest differs from the immutable local publication")
    pinned = tagged_image.rsplit(":", 1)[0] + "@" + expected
    return pinned


def load_collector(env, with_checksum=False, with_bytes=False, profile=None):
    supplied = env.get("COLLECTOR_METADATA_SHA256", "")
    if supplied not in (None, ""):
        if not isinstance(supplied, str) or not re.fullmatch(r"[0-9a-f]{64}", supplied):
            raise ValueError("COLLECTOR_METADATA_SHA256 must be a lowercase SHA256 when supplied")
    path = Path(env["COLLECTOR_METADATA_FILE"])
    if profile is not None:
        # Runtime consumers use the exact validated stage-03 bytes, never a
        # second reading of the transferred source to establish expected state.
        recorded = json.loads((Path(profile) / "installed-platform.json").read_text())
        if str(path) != recorded["collector_metadata_file"]:
            raise ValueError("Collector metadata source differs from the inspected profile")
        expected = recorded["collector_metadata_sha256"]
        if supplied not in (None, "") and supplied != expected:
            raise ValueError("Collector metadata differs from its supplied SHA256")
        supplied = expected
        path = Path(recorded["collector_metadata_snapshot"])
    if not path.is_absolute():
        raise ValueError("Collector metadata must be a bounded absolute file")
    with path.open("rb") as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError("Collector metadata exceeds the size limit")
    checksum = hashlib.sha256(raw).hexdigest()
    if supplied not in (None, "") and checksum != supplied:
        raise ValueError("Collector metadata differs from its supplied SHA256")
    metadata = json.loads(raw)
    image = env.get("COLLECTOR_IMAGE", "")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9.:-]*/[a-z0-9._/-]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("COLLECTOR_IMAGE must be an immutable registry digest")
    if metadata.get("schema") != 1 or metadata.get("image") != image:
        raise ValueError("Collector reference differs from publication metadata")
    if "RUNTIME_CLASS" in env:
        target = runpy.run_path(str(HERE / "kata-runtime-profile.py"))["runtime_target"](env["RUNTIME_CLASS"])
        declared = metadata.get("cpu_tee")
        if declared != target["cpu_tee"]:
            raise ValueError("Published collector target differs from the selected runtime")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", metadata.get("image_id", "")):
        raise ValueError("Collector metadata lacks the reviewed image configuration digest")
    if with_bytes:
        return metadata, checksum, raw
    return (metadata, checksum) if with_checksum else metadata


def cache_collector(image, directory, env, profile=None):
    """Populate only the worker's metadata cache from an anonymously readable pinned image."""
    metadata = load_collector(env, profile=profile)
    if image != metadata["image"]:
        raise ValueError("Unexpected collector image")
    directory = Path(directory)
    certs = directory / "skopeo-certs"
    certs.mkdir(mode=0o700, exist_ok=True)
    shutil.copyfile(env["COLLECTOR_REGISTRY_CA"], certs / "ca.crt")
    anonymous = directory / "anonymous-auth.json"
    anonymous.write_text('{"auths":{}}\n')
    inspected = json.loads(
        subprocess.check_output(
            [
                "skopeo",
                "inspect",
                "--tls-verify=true",
                "--authfile",
                str(anonymous),
                "--cert-dir",
                str(certs),
                "docker://" + image,
            ],
            text=True,
        )
    )
    if inspected.get("Digest") != image.rsplit("@", 1)[1]:
        raise ValueError("Collector registry returned a different immutable manifest")
    manifest = json.loads(
        subprocess.check_output(
            [
                "skopeo",
                "inspect",
                "--tls-verify=true",
                "--raw",
                "--authfile",
                str(anonymous),
                "--cert-dir",
                str(certs),
                "docker://" + image,
            ],
            text=True,
        )
    )
    if manifest.get("config", {}).get("digest") != metadata["image_id"]:
        raise ValueError("Published collector image configuration differs from trusted publication")
    archive = directory / "collector-host-cache.tar"
    tag = image.split("@", 1)[0] + ":collector-" + image.rsplit(":", 1)[1][:16]
    subprocess.run(
        [
            "skopeo",
            "copy",
            "--src-authfile",
            str(anonymous),
            "--src-tls-verify=true",
            "--src-cert-dir",
            str(certs),
            "docker://" + image,
            "docker-archive:" + str(archive) + ":" + tag,
        ],
        check=True,
        stdout=sys.stderr,
    )
    subprocess.run(
        ["sudo", "ctr", "--namespace", "k8s.io", "images", "import", str(archive)], check=True, stdout=sys.stderr
    )
    subprocess.run(
        ["sudo", "ctr", "--namespace", "k8s.io", "images", "tag", "--force", tag, image], check=True, stdout=sys.stderr
    )
