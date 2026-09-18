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

"""Validate build inputs and resolve project configuration."""

import copy
import ipaddress
import os
import re
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import yaml

from ..common.contracts import HEADER_BYTES, PLATFORMS, STORAGE_PROFILE, identifier
from ..common.errors import BuildError, require
from ..common.io import canonical, digest_file, read_json
from ..common.validation import ports, validate_nfs_mount
from ..common.versions import NVAT_COMMIT, TRUSTEE_COMMIT
from .provisioning import validate_apt_repositories

PRIVATE_KEY_MARKERS = (
    b"-----BEGIN PRIVATE KEY-----",
    b"-----BEGIN ENCRYPTED PRIVATE KEY-----",
    b"-----BEGIN RSA PRIVATE KEY-----",
    b"-----BEGIN EC PRIVATE KEY-----",
    b"-----BEGIN DSA PRIVATE KEY-----",
    b"-----BEGIN OPENSSH PRIVATE KEY-----",
)


SOURCE = Path(__file__).resolve().parents[2]


INPUTS = SOURCE / "inputs"


# Minimum NVAT 3.0 claims. `secboot` and `dbgstat` are the confidential
# computing gate. The remaining claims bind the result to the fresh nonce and
# verified report, driver RIM, vBIOS RIM, certificate chains, and measurements.
GPU_CERTIFICATE_RULES = {
    "x-nvidia-cert-status": "valid",
    "x-nvidia-cert-ocsp-status": "good",
    "x-nvidia-cert-ocsp-nonce-matches": True,
    "x-nvidia-cert-ocsp-response-valid": True,
}


GPU_REQUIRED_CLAIMS = {
    "secboot": True,
    "dbgstat": "disabled",
    "measres": "success",
    "x-nvidia-gpu-arch-check": True,
    "x-nvidia-gpu-attestation-report-cert-chain": GPU_CERTIFICATE_RULES,
    "x-nvidia-gpu-attestation-report-cert-chain-fwid-match": True,
    "x-nvidia-gpu-attestation-report-parsed": True,
    "x-nvidia-gpu-attestation-report-nonce-match": True,
    "x-nvidia-gpu-attestation-report-signature-verified": True,
    "x-nvidia-gpu-driver-rim-fetched": True,
    "x-nvidia-gpu-driver-rim-cert-chain": GPU_CERTIFICATE_RULES,
    "x-nvidia-gpu-driver-rim-signature-verified": True,
    "x-nvidia-gpu-driver-rim-measurements-available": True,
    "x-nvidia-gpu-driver-rim-version-match": True,
    "x-nvidia-gpu-vbios-rim-fetched": True,
    "x-nvidia-gpu-vbios-rim-cert-chain": GPU_CERTIFICATE_RULES,
    "x-nvidia-gpu-vbios-rim-signature-verified": True,
    "x-nvidia-gpu-vbios-rim-measurements-available": True,
    "x-nvidia-gpu-vbios-rim-version-match": True,
    "x-nvidia-gpu-vbios-index-no-conflict": True,
}


# Current NRAS v3 responses can omit these descriptive schema claims. RIM
# signatures, chains, versions and measurements above remain mandatory.
GPU_CLAIMS_IF_PRESENT = {
    "x-nvidia-device-type": "gpu",
    "x-nvidia-ver": "3.0",
    "x-nvidia-gpu-driver-rim-schema-validated": True,
    "x-nvidia-gpu-vbios-rim-schema-validated": True,
}


DEFAULT_PACKAGES = [
    "python3=3.14.3-0ubuntu2",
    "python3-yaml=6.0.3-1build1",
    "python3-cryptography=46.0.5-1ubuntu2.2",
    "initramfs-tools=0.151ubuntu1",
    "cryptsetup-bin=2:2.8.4-1ubuntu4",
    "nfs-common=1:2.8.5-1ubuntu1",
    "nftables=1.1.6-1",
    "e2fsprogs=1.47.2-3ubuntu4",
    "dmsetup=2:1.02.205-2ubuntu3",
    "iproute2=6.19.0-1ubuntu1.1",
    "chrony=4.8-2ubuntu1",
    "docker.io=29.1.3-0ubuntu4.1",
    "containerd=2.2.2-0ubuntu1.1",
    "linux-image-7.0.0-31-generic=7.0.0-31.31",
    "linux-modules-7.0.0-31-generic=7.0.0-31.31",
]


PROFILE_DEFAULTS = {
    "profile_version": "cpu-2026.09-r4",
    "guest_release": "26.04",
    "gpu": "none",
    "gpu_count": 1,
    "base_image": str(INPUTS / "ubuntu-26.04-server-cloudimg-amd64.img"),
    "build_firmware": "/usr/share/ovmf/OVMF.fd",
    "build_user": "ubuntu",
    "root_drive_size": 8,
    "vcpus": 4,
    "memory_gib": 8,
    "vault_header_bytes": HEADER_BYTES,
    "vault_storage_profile": STORAGE_PROFILE,
    "kernel_version": "7.0.0-31-generic",
    "python_version": "3.14.3-0ubuntu2",
    "docker_version": "29.1.3-0ubuntu4.1",
    "containerd_version": "2.2.2-0ubuntu1.1",
    "cryptsetup_version": "2:2.8.4-1ubuntu4",
    "gpu_attestation_library": str(INPUTS / "libnvat.so.1"),
    "gpu_attestation_provenance": str(INPUTS / "nvat_build.json"),
    "required_system_packages": DEFAULT_PACKAGES,
    # Unmodified CoCo v0.23.0 / Trustee v0.22.0.
    "trustee_commit": TRUSTEE_COMMIT,
    "kbs_url": "https://kbs.example.org:8443",
    "kbs_cert": str(INPUTS / "kbs-ca.pem"),
    "as_public_key": str(INPUTS / "as-public.pem"),
    "token_algorithm": "ES256",
    "token_issuer": "CoCo-Attestation-Service",
    # The upstream kbs-client CLI uses the default AS policy selector.
    "attestation_policy_id": "default",
    "attestation_policy": str(SOURCE / "config/attestation_policy.rego"),
    "reference_values": str(INPUTS / "approved-tcb-references.json"),
    "bootstrap_egress": [443, 8443],
    "platforms": {
        "amd_sev_snp": {
            "enabled": True,
            "attester": "snp",
            "firmware": "/usr/share/ovmf/OVMF.amdsev.fd",
            "kbs_client": str(INPUTS / "kbs-client"),
            "cpu_model": "EPYC-v4",
        },
        "intel_tdx": {
            "enabled": True,
            "attester": "tdx",
            "firmware": str(INPUTS / "OVMF.inteltdx.fd"),
            "kbs_client": str(INPUTS / "kbs-client"),
            "cpu_model": "host",
            "quote_generation": {"type": "vsock", "cid": 2, "port": 4050},
        },
    },
}


def contains_private_key(path):
    overlap = b""
    keep = max(map(len, PRIVATE_KEY_MARKERS)) - 1
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value = overlap + block
            if any(marker in value for marker in PRIVATE_KEY_MARKERS):
                return True
            overlap = value[-keep:]
    return False


class UniqueLoader(yaml.SafeLoader):
    pass


def mapping(loader, node, deep=False):
    result = {}
    for key, value in node.value:
        key = loader.construct_object(key, deep=deep)
        require(isinstance(key, str) and key not in result, "Duplicate or non-string YAML key")
        result[key] = loader.construct_object(value, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)


def load_yaml(path):
    path = Path(path).resolve()
    with path.open() as stream:
        value = yaml.load(stream, Loader=UniqueLoader)
    require(isinstance(value, dict), "Configuration must be a mapping")
    return value


def local_path(config_path, value, directory=False):
    require(isinstance(value, str) and value, "Expected input path")
    path = Path(value)
    if not path.is_absolute():
        path = Path(config_path).resolve().parent / path
    path = path.resolve()
    require(path.is_dir() if directory else path.is_file(), f"Input does not exist: {path}")
    return str(path)


def public_sidecar(path):
    """Reject private-key material before copying an unencrypted sidecar tree."""
    root = Path(path)
    for item in root.rglob("*"):
        require(not item.is_symlink(), "Clear-text sidecar inputs cannot contain symbolic links")
        if not item.is_file():
            continue
        name = item.name.lower()
        require(
            not name.endswith((".key", ".p12", ".pfx", ".jks"))
            and name not in ("id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"),
            "Private-key files must be placed in application_files (the encrypted vault)",
        )
        require(
            not contains_private_key(item),
            "Private-key material must be placed in application_files (the encrypted vault)",
        )
    return str(root)


def resolve_root_overlay_max_mib(profile):
    """Resolve the writable root capacity, defaulting to half of guest RAM."""
    memory_gib = profile.get("memory_gib")
    require(type(memory_gib) is int and memory_gib > 0, "Invalid memory_gib")
    value = profile.setdefault("root_overlay_max_mib", memory_gib * 512)
    require(
        type(value) is int and 0 < value <= memory_gib * 1024,
        "root_overlay_max_mib must be a positive integer no larger than guest RAM",
    )
    return value


def validate_gpu_policy(path):
    """Refuse a GPU policy that would accept a non-confidential or debug GPU.

    The AS compares the verified NRAS version 3.0 claims against every
    rule stated here. Require the full safe rule set at Stage 1 so a profile
    cannot weaken appraisal by omitting a claim.
    See config/gpu_policy.json for a conforming policy.
    """
    import json

    try:
        policy = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        raise BuildError("GPU policy is unreadable or not valid JSON") from None
    require(policy.get("version") == "3.0", "GPU policy must declare version 3.0")
    rules = policy.get("required-claims")
    require(isinstance(rules, dict), "GPU policy must state required-claims")

    def safe_subset(required, stated, prefix=""):
        require(isinstance(stated, dict), f"GPU policy does not constrain {prefix.rstrip('.')}")
        for claim, expected in required.items():
            name = prefix + claim
            require(claim in stated, f"GPU policy does not constrain {name}")
            if isinstance(expected, dict):
                safe_subset(expected, stated[claim], name + ".")
            else:
                require(
                    type(stated[claim]) is type(expected) and stated[claim] == expected,
                    f"GPU policy accepts an unsafe value for {name}",
                )

    safe_subset(GPU_REQUIRED_CLAIMS, rules)
    safe_subset(GPU_CLAIMS_IF_PRESENT, policy.get("claims-if-present"), "claims-if-present.")
    return path


def gpu_inputs(path, value):
    """Bind GPU repository trust roots and NVAT source provenance to Stage 1."""
    repositories = value.get("gpu_apt_repositories")
    require(isinstance(repositories, list) and repositories, "Supply authenticated gpu_apt_repositories")
    normalized = []
    for repository in repositories:
        require(isinstance(repository, dict) and "keyring" in repository, "GPU repository requires a keyring")
        metadata = {key: item for key, item in repository.items() if key != "keyring"}
        try:
            validate_apt_repositories([metadata])
        except ValueError as error:
            raise BuildError(str(error)) from None
        keyring = local_path(path, repository["keyring"])
        require(digest_file(keyring) == metadata["keyring_sha256"], "GPU repository keyring digest mismatch")
        normalized.append(dict(metadata, keyring=keyring))
    value["gpu_apt_repositories"] = normalized
    provenance = read_json(value["gpu_attestation_provenance"])
    require(isinstance(provenance, dict), "Invalid NVAT provenance")
    require(
        provenance.get("source_repository") == "https://github.com/NVIDIA/attestation-sdk.git"
        and provenance.get("source_commit") == NVAT_COMMIT,
        "NVAT provenance must match Trustee v0.22.0's pinned SDK source",
    )
    require(
        provenance.get("patch_sha256") == digest_file(SOURCE / "cvm/build/nvat_libxml2_const.patch"),
        "NVAT provenance must record the reviewed libxml2 compatibility patch",
    )
    require(
        provenance.get("library_sha256") == digest_file(value["gpu_attestation_library"]),
        "NVAT library digest differs from its provenance",
    )
    require(provenance.get("build_environment") == "ubuntu-26.04-x86_64", "Build NVAT for the guest environment")


def profile(path):
    value = load_yaml(path)
    supplied_platforms = value.pop("platforms", None)
    defaults = copy.deepcopy(PROFILE_DEFAULTS)
    defaults.update(value)
    value = defaults
    if supplied_platforms is not None:
        require(isinstance(supplied_platforms, dict) and supplied_platforms, "No platforms configured")
        value["platforms"] = {}
        for name, settings in supplied_platforms.items():
            require(isinstance(settings, dict), "Platform settings must be a mapping")
            merged = copy.deepcopy(PROFILE_DEFAULTS["platforms"].get(name, {}))
            merged.update(settings)
            value["platforms"][name] = merged
    require(re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,63}", value.get("profile_version", "")), "Invalid profile version")
    require(value.get("gpu") in ("none", "nvidia_cc"), "gpu must be none or nvidia_cc")
    require(
        type(value.get("gpu_count")) is int and 1 <= value["gpu_count"] <= 8,
        "gpu_count must be an integer from 1 through 8",
    )
    require(value.get("vault_header_bytes") == HEADER_BYTES, "Unsupported header range")
    require(value.get("vault_storage_profile") == STORAGE_PROFILE, "Unsupported authenticated storage profile")
    require(value.get("guest_release") == "26.04", "This implementation targets an Ubuntu 26.04 guest")
    require(re.fullmatch(r"[a-f0-9]{40}", value.get("trustee_commit", "")), "Pin trustee_commit to a full revision")
    require("trustee_patch_digest" not in value, "Use unmodified CoCo Trustee; remove trustee_patch_digest")
    require("gpu_attestation_binary" not in value, "Upstream NVIDIA attestation uses libnvat, not a custom collector")
    identifier(value.get("attestation_policy_id"))
    require(value["attestation_policy_id"] == "default", "The upstream kbs-client uses the default AS policy")
    require(urlparse(value.get("kbs_url", "")).scheme == "https", "KBS requires HTTPS")
    require(value.get("token_algorithm") in ("RS256", "ES256", "EdDSA"), "Pin the AS token algorithm")
    require(isinstance(value.get("token_issuer"), str) and value["token_issuer"], "Pin the AS token issuer")
    for key in ("kernel_version", "python_version", "docker_version", "containerd_version", "cryptsetup_version"):
        require(isinstance(value.get(key), str) and value[key] and "<" not in value[key], f"Pin {key}")
    packages = value.get("required_system_packages")
    require(
        isinstance(packages, list) and packages and all(isinstance(p, str) and "=" in p for p in packages),
        "Every system package must have an exact apt version pin",
    )
    pins = dict(p.split("=", 1) for p in packages)
    require(len(pins) == len(packages) and all(pins.values()), "Duplicate or empty package pin")
    for package, field in (
        ("python3", "python_version"),
        ("docker.io", "docker_version"),
        ("containerd", "containerd_version"),
        ("cryptsetup-bin", "cryptsetup_version"),
    ):
        require(pins.get(package) == value[field], f"Package pin differs from {field}")
    require("chrony" in pins, "Pin chrony for the pre-attestation clock gate")
    require(
        "linux-image-" + value["kernel_version"] in pins and "linux-modules-" + value["kernel_version"] in pins,
        "Pin both the selected kernel image and its modules",
    )
    for key in ("vcpus", "memory_gib", "root_drive_size"):
        require(type(value.get(key)) is int and value[key] > 0, f"Invalid {key}")
    resolve_root_overlay_max_mib(value)
    ports(value.get("bootstrap_egress", []))
    require(isinstance(value.get("platforms"), dict) and value["platforms"], "No platforms configured")
    for name, settings in value["platforms"].items():
        require(name in PLATFORMS and isinstance(settings, dict), "Unsupported platform")
        require(settings.get("attester") == ("snp" if name == PLATFORMS[0] else "tdx"), "Attester mismatch")
        require(type(settings.get("enabled", True)) is bool, "enabled must be boolean")
        for key in ("firmware", "kbs_client"):
            settings[key] = local_path(path, settings.get(key))
        require(
            not str(settings["firmware"]).endswith(".ms.fd") or settings.get("shim"),
            "Secure Boot firmware requires a reviewed signed shim/kernel path; use inputs/OVMF.inteltdx.fd for measured direct boot",
        )
        if settings.get("shim"):
            require(name == "intel_tdx", "Shim direct boot currently requires the TDX/QEMU 10 profile")
            settings["shim"] = local_path(path, settings["shim"])
        require(settings.get("cpu_model") and re.fullmatch(r"[A-Za-z0-9_.-]+", settings["cpu_model"]), "Pin CPU model")
    for key in ("base_image", "build_firmware", "kbs_cert", "as_public_key", "attestation_policy", "reference_values"):
        value[key] = local_path(path, value.get(key))
    from ..common.references import validate_references

    validate_references(
        read_json(value["reference_values"]), [p for p, v in value["platforms"].items() if v.get("enabled", True)]
    )
    if value["gpu"] == "nvidia_cc":
        for key in ("gpu_policy", "gpu_attestation_library", "gpu_attestation_provenance"):
            value[key] = local_path(path, value.get(key))
        gpu_inputs(path, value)
        validate_gpu_policy(value["gpu_policy"])
        validate_references(read_json(value["reference_values"]), gpu=True)
        require(urlparse(value.get("gpu_attestation_url", "")).scheme == "https", "GPU attestation requires HTTPS")
        gpu_packages = value.get("gpu_packages")
        require(isinstance(gpu_packages, list) and gpu_packages, "Pin GPU driver/toolkit and NVAT packages")
        require(all(isinstance(p, str) and "=" in p for p in gpu_packages), "Pin every GPU package to an exact version")
        gpu_pins = dict(p.split("=", 1) for p in gpu_packages)
        require(len(gpu_pins) == len(gpu_packages) and all(gpu_pins.values()), "Duplicate or empty GPU package pin")
        require(not set(gpu_pins) & set(pins), "GPU package pins duplicate required_system_packages")
        for package in ("nvidia-container-toolkit",):
            require(package in gpu_pins, f"GPU profile must pin {package}")
        driver_meta = any(name.startswith("nvidia-driver-") for name in gpu_pins)
        precompiled_modules = any(name.startswith("linux-modules-nvidia-") for name in gpu_pins)
        require(
            driver_meta or (precompiled_modules and any(name.startswith("libnvidia-compute-") for name in gpu_pins)),
            "GPU profile must pin an NVIDIA driver or precompiled modules and compute library",
        )
    if value.get("acceptance_runner"):
        require(isinstance(value["acceptance_runner"], str), "acceptance_runner must be a command")
        if "/" in value["acceptance_runner"]:
            value["acceptance_runner"] = local_path(path, value["acceptance_runner"])
            require(os.access(value["acceptance_runner"], os.X_OK), "acceptance_runner must be executable")
    value.setdefault("build_user", "ubuntu")
    require(re.fullmatch(r"[a-z_][a-z0-9_-]*", value["build_user"]), "Invalid build user")
    return value


def cvm_image(config_path, value):
    """Resolve a profile-set folder or a pinned OCI registry reference."""
    require(isinstance(value, str) and value, "cvm_image must be a folder or registry digest reference")
    registry_url = value.startswith(("oci://", "https://"))
    reference = value.split("://", 1)[1] if registry_url else value
    if registry_url or "@sha256:" in reference:
        require(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.:-]*/[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}", reference),
            "cvm_image registry reference must be registry/repository@sha256:<64 lowercase hex characters>",
        )
        return reference
    require("://" not in value, "cvm_image supports HTTPS or oci registry URLs and local folders")
    return local_path(config_path, str(Path(value).expanduser()), directory=True)


def project(build_config, project_config=None):
    """Load shared builder settings; credentials are relative to this file."""
    if project_config is not None:
        path = Path(project_config).expanduser().resolve()
        require(path.is_file(), f"Project configuration does not exist: {path}")
    else:
        directory = Path(build_config).resolve().parent
        path = next(
            (
                parent / "cvm_project.yml"
                for parent in (directory, *directory.parents)
                if (parent / "cvm_project.yml").exists() or (parent / "cvm_project.yml").is_symlink()
            ),
            None,
        )
        require(path is not None, "No cvm_project.yml found; create it or pass --project-config")
        require(path.is_file(), f"Project configuration is not a file: {path}")
    value = load_yaml(path)
    require(set(value) == {"trustee"}, "Project configuration must contain only trustee")
    service = value["trustee"]
    require(
        isinstance(service, dict) and set(service) == {"url", "ca", "admin_token_file"},
        "Project trustee requires url, ca and admin_token_file",
    )
    endpoint = service["url"]
    require(isinstance(endpoint, str) and endpoint, "Trustee requires an HTTPS URL")
    parsed = urlparse(endpoint)
    require(
        parsed.scheme == "https"
        and parsed.hostname
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
        and not any(char.isspace() for char in endpoint),
        "Trustee requires an HTTPS URL without credentials, query or fragment",
    )
    for key in ("ca", "admin_token_file"):
        service[key] = local_path(path, service[key])
    return value


def application(path):
    value = load_yaml(path)
    require("trustee" not in value, "Move trustee from vault_build.yml to cvm_project.yml")
    allowed = {
        "cvm_image",
        "docker_archive",
        "image_id",
        "platforms",
        "application_files",
        "user_config",
        "user_data",
        "hosts_entries",
        "container",
        "applog_drive_size",
        "user_config_drive_size",
        "user_data_drive_size",
        "vault_drive_size",
        "allowed_ports",
        "allowed_out_ports",
        "requires_gpu",
        "services",
        "nfs_mount",
    }
    require(not set(value) - allowed, "Unknown application input; provisioning schemas are external to the builder")
    for key, default in {
        "applog_drive_size": 1,
        "user_config_drive_size": 1,
        "user_data_drive_size": 1,
        "vault_drive_size": 8,
    }.items():
        value.setdefault(key, default)
    require(
        re.fullmatch(r"sha256:[a-f0-9]{64}", value.get("image_id", "")), "image_id must be an immutable Docker image ID"
    )
    value["cvm_image"] = cvm_image(path, value.get("cvm_image"))
    value["docker_archive"] = local_path(path, value.get("docker_archive"))
    if "platforms" in value:
        platforms = value["platforms"]
        require(
            isinstance(platforms, list)
            and platforms
            and all(isinstance(platform, str) and platform in PLATFORMS for platform in platforms)
            and len(set(platforms)) == len(platforms),
            "platforms must be a non-empty list of distinct supported platforms, or omitted",
        )
    for key in ("application_files", "user_config", "user_data"):
        if key in value:
            value[key] = local_path(path, value[key], directory=True)
    for key in ("user_config", "user_data"):
        if key in value:
            public_sidecar(value[key])
    for key in ("applog_drive_size", "user_config_drive_size", "user_data_drive_size", "vault_drive_size"):
        require(type(value.get(key)) is int and value[key] > 0, f"{key} must be a positive integer GiB size")
    require(type(value.get("requires_gpu", False)) is bool, "requires_gpu must be boolean")
    value.setdefault("requires_gpu", False)
    for key in ("allowed_ports", "allowed_out_ports"):
        ports(value.setdefault(key, []))
    container = value.setdefault("container", {})
    require(isinstance(container, dict), "container must be a mapping")
    require(
        not set(container) - {"entrypoint", "command", "env", "volumes", "ports", "tee_device"},
        "Unknown container option",
    )
    for key in ("entrypoint", "command"):
        if key in container:
            require(
                isinstance(container[key], list)
                and all(isinstance(x, str) and "\x00" not in x for x in container[key]),
                f"container.{key} must be an argument array",
            )
    require(type(container.get("tee_device", False)) is bool, "tee_device must be boolean")
    env = container.setdefault("env", {})
    require(
        isinstance(env, dict)
        and all(
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k) and isinstance(v, str) and "\x00" not in v
            for k, v in env.items()
        ),
        "Invalid environment mapping",
    )
    require(isinstance(container.setdefault("volumes", []), list), "container.volumes must be a list")
    require(isinstance(container.setdefault("ports", []), list), "container.ports must be a list")
    for volume in container["volumes"]:
        require(isinstance(volume, dict) and set(volume) == {"source", "target", "read_only"}, "Invalid volume mapping")
        require(type(volume["read_only"]) is bool, "read_only must be boolean")
        source, target = PurePosixPath(volume["source"]), PurePosixPath(volume["target"])
        require(
            source.is_absolute()
            and target.is_absolute()
            and ".." not in source.parts + target.parts
            and "," not in str(source) + str(target),
            "Volume paths must be absolute without traversal",
        )
        require(
            source.parts[1:2] in (("vault",), ("applog",), ("user_config",), ("user_data",)),
            "Unsupported volume source",
        )
        require(
            source.parts[1] not in ("user_config", "user_data") or volume["read_only"],
            "Sidecar inputs must remain read-only",
        )
        if source.parts[1] == "vault":
            require(source.is_relative_to("/vault/application"), "Container cannot mount vault control files")
            require(
                volume["read_only"]
                or any(source.is_relative_to(p) for p in ("/vault/application/runtime", "/vault/application/data")),
                "Only application runtime/data may be writable",
            )
        require(
            str(target) not in ("/", "/vault", "/applog", "/user_config", "/user_data", "/host/bin"),
            "Cannot replace mandatory mounts",
        )
    for port in container["ports"]:
        require(isinstance(port, dict) and set(port) == {"host", "container"}, "Port mappings need host/container")
        ports([port["host"]])
        ports([port["container"]])
        require(port["host"] in value["allowed_ports"], "Container port is not allowed by firewall")
    hosts = value.setdefault("hosts_entries", {})
    require(isinstance(hosts, dict), "hosts_entries must be a mapping")
    for hostname, address in hosts.items():
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", hostname), "Invalid hostname")
        try:
            ipaddress.ip_address(address)
        except ValueError:
            require(False, "hosts_entries requires literal IP addresses")
    text = canonical(container).decode()
    require(
        not any(
            item in text for item in ("/dev/sev", "/dev/tdx", "snpguest", "cpu_attestation_snp", "cpu_attestation_tdx")
        ),
        "Launch configuration must obtain TEE settings from the generic runtime",
    )
    from ..common.services import validate_service

    require(isinstance(value.setdefault("services", []), list), "services must be a list of unit file paths")
    value["services"] = [local_path(path, item) for item in value["services"]]
    for item in value["services"]:
        validate_service(Path(item).name, Path(item).read_text())
    if value.get("nfs_mount") is not None:
        validate_nfs_mount(value["nfs_mount"])
    return value
