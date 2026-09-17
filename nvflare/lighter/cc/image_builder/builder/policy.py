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

"""Reusable bundle policy. No vault identity is ever embedded in a rule."""

import base64
import hashlib
import json
import re
from pathlib import Path

from .common import PLATFORMS, digest_file, identifier, read_json, require, write_json

PRELUDE = """package policy
default allow = false
cpu := input["submods"]["cpu0"]
ev := cpu["ear.veraison.annotated-evidence"]
tv := cpu["ear.trustworthiness-vector"]
approved_cpu(policy_id) {
    cpu["ear.appraisal-policy-id"] == policy_id
    cpu["ear.status"] == "affirming"
    tv["executables"] == 3
    tv["hardware"] == 2
    tv["configuration"] == 2
}
"""


def validate_measurements(platform, values):
    require(platform in PLATFORMS and isinstance(values, dict), "Invalid reference measurements")
    if platform == "amd_sev_snp":
        require(set(values) == {"snp.measurement"}, "SNP requires exactly its launch measurement")
        value = values["snp.measurement"]
        require(
            isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9+/]{64}", value), "Invalid SNP measurement encoding"
        )
        require(len(base64.b64decode(value, validate=True)) == 48, "Invalid SNP measurement size")
    else:
        require(
            set(values) == {"mr_td", "rtmr_0", "rtmr_1", "rtmr_2"},
            "TDX requires MRTD and RTMR0/1/2",
        )
        require(
            all(isinstance(x, str) and re.fullmatch(r"[0-9a-f]{96}", x) for x in values.values()),
            "Invalid TDX measurement encoding",
        )


def bundle_rule(manifest):
    platform = manifest["platform"]
    values = manifest["measurements"]
    validate_measurements(platform, values)
    build = json.dumps(identifier(manifest["build_id"]))
    policy = json.dumps(identifier(manifest["attestation_policy_id"]))
    lines = ["allow {", f"    approved_cpu({policy})", '    is_string(ev["init_data"])']
    contract = manifest.get("contract", {})
    if contract.get("gpu") == "nvidia_cc":
        from .gpu_policy import resource_conditions

        lines += resource_conditions(contract["gpu_count"], policy)
    if platform == "amd_sev_snp":
        lines += [
            f'    ev["snp"]["measurement"] == {json.dumps(values["snp.measurement"])}',
            '    ev["snp"]["policy_debug_allowed"] == false',
            '    ev["snp"]["policy_migrate_ma"] == false',
            '    regex.match("^[A-Za-z0-9+/]{42}[AEIMQUYcgkosw048]=$", ev["init_data"])',
            '    binding_id := replace(replace(trim_suffix(ev["init_data"], "="), "+", "-"), "/", "_")',
        ]
    else:
        lines += [
            f'    ev["tdx"]["quote"]["body"]["{key}"] == {json.dumps(values[key])}'
            for key in ("mr_td", "rtmr_0", "rtmr_1", "rtmr_2")
        ]
        lines += [
            '    ev["tdx"]["td_attributes"]["debug"] == false',
            '    regex.match("^[0-9a-f]{64}0{32}$", ev["init_data"])',
            '    binding_id := ev["init_data"]',
        ]
    lines += [f'    data["resource-path"] == sprintf("resource/keys/%s/%s", [{build}, binding_id])', "}", ""]
    return "\n".join(lines)


def compose(manifests):
    ids = [m["build_id"] for m in manifests]
    require(len(ids) == len(set(ids)), "Duplicate bundle identity")
    return PRELUDE + "\n" + "\n".join(bundle_rule(m) for m in sorted(manifests, key=lambda m: m["build_id"]))


def verify_bundle(directory):
    directory = Path(directory).resolve()
    manifest = read_json(directory / "cvm_manifest.json")
    identifier(manifest["build_id"])
    require(re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,63}", manifest["profile_version"]), "Invalid profile version")
    if manifest.get("dev_mode"):
        require(manifest["profile_version"].startswith("dev-") and manifest["measurements"] == {}, "Invalid dev bundle")
    else:
        validate_measurements(manifest["platform"], manifest["measurements"])
    required_artifacts = {
        "verity_root.qcow2",
        "OVMF.fd",
        "vmlinuz",
        "initrd.img",
        "attestation_policy.rego",
        "reference_values.json",
        "launch_cvm.sh.tmpl",
        "shutdown_cvm.sh.tmpl",
    }
    if manifest["contract"].get("gpu") == "nvidia_cc":
        required_artifacts.add("gpu_attestation_policy.rego")
    if manifest["launch_shape"].get("shim"):
        required_artifacts.add("shim.efi")
    require(set(manifest["sha256"]) == required_artifacts, "Incomplete bundle artifact hashes")
    for name, expected in manifest["sha256"].items():
        require(Path(name).name == name, "Invalid bundle filename")
        require(digest_file(directory / name) == expected, f"Bundle artifact hash mismatch: {name}")
    require(
        hashlib.sha256(manifest["cmdline"].encode()).hexdigest() == manifest["cmdline_sha256"],
        "Command-line digest mismatch",
    )
    overlay_max = manifest["contract"].get("root_overlay_max_mib")
    require(type(overlay_max) is int and overlay_max > 0, "Invalid root overlay capacity")
    require(
        manifest["cmdline"].split().count(f"cvm.root_overlay_max_mib={overlay_max}") == 1,
        "Measured command line does not contain the profile's root overlay capacity",
    )
    require(
        (directory / "resource_policy.rego").read_text() == compose([] if manifest.get("dev_mode") else [manifest]),
        "Bundle resource policy mismatch",
    )
    return manifest


# A receipt is per generic bundle, never per vault. Hardware evidence remains
# a required administrative input, not a boolean flag that the builder invents.
ACCEPTANCE_CHECKS = {
    "boot_measurements",
    "local_binding",
    "wrong_binding",
    "header_snapshot",
    "payload_corruption",
    "reboot_after_writes",
    "interrupted_journal",
    "interrupted_docker_load",
    "integrity_monitor_failure",
    "exclusive_attachment",
    "cross_vault_key_denial",
    "negative_appraisal",
    "policy_selection",
    "unauthorized_administration",
    "durable_key_retry",
    "key_revocation",
    "rollback_retirement",
    "generic_container",
    "initramfs_no_kbs",
    "policy_readback",
    "clear_sidecar_scan",
    "read_only_input_disks",
    "writable_applog",
    "root_disk_corruption",
    "ssh_service_and_socket_disabled",
    "attestation_drop_deadline",
    "clock_synchronized_before_attestation",
    "root_overlay_capacity",
}

SNP_ACCEPTANCE_CHECKS = {"snp_vcek_cache"}
GPU_ACCEPTANCE_CHECKS = {
    "gpu_negative_key_denial",
    "gpu_positive_key_release",
    "gpu_policy_selection",
    "cross_class_denial",
    "periodic_gpu_denial",
}


def required_acceptance_checks(manifest):
    checks = set(ACCEPTANCE_CHECKS)
    if manifest.get("platform") == "amd_sev_snp":
        checks |= SNP_ACCEPTANCE_CHECKS
    if manifest.get("contract", {}).get("gpu") == "nvidia_cc":
        checks |= GPU_ACCEPTANCE_CHECKS
    return checks


def verify_approval(directory):
    directory = Path(directory)
    manifest = verify_bundle(directory)
    require(not manifest.get("dev_mode"), "Development roots can never receive production approval")
    receipt = read_json(directory / "approval.json")
    require(
        receipt.get("manifest_sha256") == digest_file(directory / "cvm_manifest.json"),
        "Approval does not cover this bundle",
    )
    require(receipt.get("build_id") == manifest["build_id"], "Approval bundle mismatch")
    require(receipt.get("status") == "approved", "Bundle has not passed production acceptance")
    evidence = receipt.get("checks", {})
    checks = required_acceptance_checks(manifest)
    require(checks <= set(evidence), "Incomplete production acceptance evidence")
    for name in checks:
        item = evidence[name]
        require(
            isinstance(item, dict)
            and item.get("passed") is True
            and re.fullmatch(r"[0-9a-f]{64}", item.get("evidence_sha256", "")),
            f"Missing successful evidence: {name}",
        )
    return manifest


def load_profile_set(path, approved=True):
    path = Path(path).resolve()
    result = read_json(path)
    require(re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,63}", result["profile_version"]), "Invalid profile version")
    require(result.get("schema_version") == 2 and result.get("bundles"), "Invalid profile set")
    for platform, entry in result["bundles"].items():
        require(platform in PLATFORMS, "Invalid platform bundle name")
        directory = path.parent / platform
        manifest = verify_approval(directory) if approved else verify_bundle(directory)
        require(
            manifest["platform"] == platform and manifest["profile_version"] == result["profile_version"],
            "Profile identity mismatch",
        )
        require(manifest["contract"] == result["contract"], "Bundles do not share the vault-facing contract")
        require(
            manifest["build_id"] == entry["build_id"]
            and digest_file(directory / "cvm_manifest.json") == entry["manifest_sha256"],
            "Profile set no longer matches its bundle",
        )
        entry["manifest"] = manifest
        entry["directory"] = str(directory)
    return result


def approve_bundle(directory, report):
    directory = Path(directory)
    manifest = verify_bundle(directory)
    require(
        report.get("manifest_sha256") == digest_file(directory / "cvm_manifest.json"),
        "Acceptance report covers another bundle",
    )
    receipt = dict(report, status="approved", build_id=manifest["build_id"])
    # Validate with the same rules before publishing an approval file.
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        for item in directory.iterdir():
            if item.is_file() and item.name != "approval.json":
                (temporary / item.name).symlink_to(item.resolve())
        write_json(temporary / "approval.json", receipt)
        verify_approval(temporary)
    write_json(directory / "approval.json", receipt, mode=0o644)
