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

"""Validate reusable bundles, profile sets and signed acceptance receipts."""

import base64
import hashlib
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from ..common.contracts import PLATFORMS, identifier
from ..common.errors import require
from ..common.io import canonical, digest_file, read_json, write_json
from ..common.measurements import validate_measurements
from ..common.policy import compose

APPROVAL_SIGNATURE_ALGORITHM = "ed25519"


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
    require(
        manifest.get("production_ready") == manifest["contract"].get("production_ready"),
        "Production eligibility differs from the shared profile contract",
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
    "attestation_quarantine_recovery",
    "clock_synchronized_before_attestation",
    "root_overlay_capacity",
}


SNP_ACCEPTANCE_CHECKS = {"snp_collateral_availability"}


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


def _ed25519():
    # Imported lazily: the delivered launcher verifies bundles without needing
    # the cryptography package on the runtime host.
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519

    return InvalidSignature, serialization, ed25519


def key_id(public_key):
    """Stable identifier: SHA-256 of the DER SubjectPublicKeyInfo encoding."""
    _, serialization, _ = _ed25519()
    der = public_key.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return hashlib.sha256(der).hexdigest()


def load_public_keys(values):
    """Load trusted acceptance authorities from PEM paths; keys already loaded pass through."""
    _, serialization, ed25519 = _ed25519()
    keys = []
    for value in values:
        if isinstance(value, ed25519.Ed25519PublicKey):
            keys.append(value)
            continue
        try:
            key = serialization.load_pem_public_key(Path(value).read_bytes())
        except (OSError, ValueError, TypeError):
            require(False, "Cannot load an approval public key; supply Ed25519 PEM files")
        require(isinstance(key, ed25519.Ed25519PublicKey), "Approval signing keys must be Ed25519")
        keys.append(key)
    require(keys, "No approval signing keys are trusted")
    return keys


def load_signing_key(path):
    _, serialization, ed25519 = _ed25519()
    try:
        key = serialization.load_pem_private_key(Path(path).read_bytes(), password=None)
    except (OSError, ValueError, TypeError):
        require(False, "Cannot load the approval signing key; supply an unencrypted Ed25519 PEM file")
    require(isinstance(key, ed25519.Ed25519PrivateKey), "Approval signing key must be Ed25519")
    return key


def _signed_body(receipt):
    return canonical({key: value for key, value in receipt.items() if key != "signature"})


def sign_receipt(receipt, signing_key):
    """Return the receipt with a detached Ed25519 signature over its canonical body."""
    key = load_signing_key(signing_key)
    body = {name: value for name, value in receipt.items() if name != "signature"}
    signature = key.sign(_signed_body(body))
    return dict(
        body,
        signature={
            "algorithm": APPROVAL_SIGNATURE_ALGORITHM,
            "key_id": key_id(key.public_key()),
            "value": base64.b64encode(signature).decode(),
        },
    )


def verify_receipt_signature(receipt, trusted_keys):
    """Require a signature by one of the trusted acceptance authorities."""
    InvalidSignature, _, _ = _ed25519()
    keys = load_public_keys(trusted_keys)
    signature = receipt.get("signature")
    require(
        isinstance(signature, dict)
        and signature.get("algorithm") == APPROVAL_SIGNATURE_ALGORITHM
        and isinstance(signature.get("key_id"), str)
        and isinstance(signature.get("value"), str),
        "Approval receipt is unsigned or uses an unsupported signature",
    )
    try:
        value = base64.b64decode(signature["value"], validate=True)
    except ValueError:
        require(False, "Approval signature encoding is invalid")
    signer = next((key for key in keys if key_id(key) == signature["key_id"]), None)
    require(signer is not None, "Approval is not signed by a trusted acceptance authority")
    try:
        signer.verify(value, _signed_body(receipt))
    except InvalidSignature:
        require(False, "Approval signature does not match the receipt")


def verify_approval(directory, trusted_keys):
    directory = Path(directory)
    manifest = verify_bundle(directory)
    require(not manifest.get("dev_mode"), "Development roots can never receive production approval")
    require(manifest.get("production_ready") is True, "This profile is not eligible for production approval")
    receipt = read_json(directory / "approval.json")
    verify_receipt_signature(receipt, trusted_keys)
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


def load_profile_set(path, approved=True, trusted_keys=()):
    path = Path(path).resolve()
    result = read_json(path)
    require(re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,63}", result["profile_version"]), "Invalid profile version")
    require(result.get("schema_version") == 2 and result.get("bundles"), "Invalid profile set")
    for platform, entry in result["bundles"].items():
        require(platform in PLATFORMS, "Invalid platform bundle name")
        directory = path.parent / platform
        manifest = verify_approval(directory, trusted_keys) if approved else verify_bundle(directory)
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


def approve_bundle(directory, report, signing_key):
    """Sign and publish an acceptance receipt for the exact finalized bundle."""
    directory = Path(directory)
    manifest = verify_bundle(directory)
    require(manifest.get("production_ready") is True, "This profile is not eligible for production approval")
    require(
        report.get("manifest_sha256") == digest_file(directory / "cvm_manifest.json"),
        "Acceptance report covers another bundle",
    )
    receipt = sign_receipt(dict(report, status="approved", build_id=manifest["build_id"]), signing_key)
    # Validate with the same rules before publishing an approval file.
    signer = load_signing_key(signing_key).public_key()
    with TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        for item in directory.iterdir():
            if item.is_file() and item.name != "approval.json":
                (temporary / item.name).symlink_to(item.resolve())
        write_json(temporary / "approval.json", receipt)
        verify_approval(temporary, [signer])
    write_json(directory / "approval.json", receipt, mode=0o644)
