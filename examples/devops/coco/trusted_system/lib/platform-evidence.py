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

"""Verify raw platform evidence without choosing a trusted baseline or approving it."""

import base64
import gzip
import hashlib
import hmac
import io
import json
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path

FRAME = "COCO_TDX_EVIDENCE_V1="
SNP_FRAME = "COCO_SNP_EVIDENCE_V1="
MAX_EVIDENCE_BYTES = 32 * 1024 * 1024
TCB_FIELDS = ("bootloader", "tee", "snp", "microcode")
FLOOR_FIELDS = tuple("snp_min_reported_tcb_" + field for field in TCB_FIELDS)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate evidence JSON key: " + key)
        result[key] = value
    return result


def framed_payload(logs, prefix):
    lines = [line[len(prefix) :] for line in logs.splitlines() if line.startswith(prefix)]
    if len(lines) != 1 or len(lines[0]) > MAX_EVIDENCE_BYTES:
        raise ValueError("Expected exactly one bounded evidence frame")
    return base64.b64decode(lines[0], validate=True)


def decode_evidence(logs):
    evidence = json.loads(framed_payload(logs, FRAME), object_pairs_hook=unique_object)
    if not isinstance(evidence, dict) or not evidence.get("quote") or not evidence.get("cc_eventlog"):
        raise ValueError("TDX evidence requires quote and nonempty CCEL")
    return evidence


def extract_profile(claims, profile_id):
    """Extract only already verified Trustee claims, never decoded quote bytes."""
    body = claims["quote"]["body"]
    result = {"id": profile_id}
    for field in ("mr_td", "rtmr_0", "rtmr_1", "rtmr_2", "rtmr_3", "xfam"):
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


def verify_tdx(directory, verifier_path, profile_id):
    """Only the reviewed verifier may authenticate quote, collateral, RTMRs and InitData."""
    directory = Path(directory)
    challenge = (directory / "request-data.bin").read_bytes()
    if len(challenge) != 64:
        raise ValueError("Rehearsal nonce must have 64 bytes")
    inputs = {"request-data.bin": challenge}
    inputs.update({name: (directory / name).read_bytes() for name in ("evidence.json", "initdata.toml")})
    with tempfile.TemporaryDirectory(prefix=".tdx-verify-", dir=directory) as temporary:
        snapshot = Path(temporary)
        for name, content in inputs.items():
            (snapshot / name).write_bytes(content)
            (snapshot / name).chmod(0o600)
        result = subprocess.run(
            [
                str(verifier_path),
                str(snapshot / "evidence.json"),
                str(snapshot / "request-data.bin"),
                str(snapshot / "initdata.toml"),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )
    claims = json.loads(result.stdout, object_pairs_hook=unique_object)
    return extract_profile(claims, profile_id), claims


def decode_snp(logs, directory):
    """Extract the exact SNP frame only after every member, checksum and nonce passes."""
    directory = Path(directory)
    expected_challenge = (directory / "request-data.bin").read_bytes()
    if len(expected_challenge) != 64:
        raise ValueError("Rehearsal nonce must have 64 bytes")
    payload = framed_payload(logs, SNP_FRAME)
    with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as stream:
        archive = stream.read(MAX_EVIDENCE_BYTES + 1)
    if len(archive) > MAX_EVIDENCE_BYTES:
        raise ValueError("SNP evidence archive is too large")
    expected = {"SHA256SUMS", "attestation-report.bin", "attestation-report.txt", "request-data.bin"}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source:
        members = source.getmembers()
        if len(members) != len(expected) or {member.name for member in members} != expected:
            raise ValueError("Unexpected SNP evidence archive contents")
        if any(not member.isfile() or member.size > MAX_EVIDENCE_BYTES for member in members):
            raise ValueError("SNP evidence members must be bounded regular files")
        files = {member.name: source.extractfile(member).read() for member in members}
    if not hmac.compare_digest(files["request-data.bin"], expected_challenge):
        raise ValueError("SNP evidence challenge mismatch")
    checked = set()
    for line in files["SHA256SUMS"].decode("ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](?:\./)?([a-zA-Z0-9.-]+)", line)
        if not match or match[2] not in expected - {"SHA256SUMS"} or match[2] in checked:
            raise ValueError("Malformed SNP evidence checksum manifest")
        name = match[2]
        if not hmac.compare_digest(hashlib.sha256(files[name]).hexdigest(), match[1]):
            raise ValueError("SNP evidence checksum mismatch: " + name)
        checked.add(name)
    if checked != expected - {"SHA256SUMS"}:
        raise ValueError("SNP evidence checksum manifest is incomplete")
    output = directory / "evidence-input"
    output.mkdir(mode=0o700)
    for name, content in files.items():
        with (output / name).open("xb") as stream:
            stream.write(content)
        (output / name).chmod(0o600)
    return output


def verify_snp(directory, verifier_path, floors, fetch_certificates=False):
    """Verify AMD signatures and fresh REPORT_DATA against independently chosen TCB floors."""
    directory = Path(directory)
    if set(floors) != set(FLOOR_FIELDS) or any(
        type(floors[key]) is not int or not 0 <= floors[key] <= 255 for key in FLOOR_FIELDS
    ):
        raise ValueError("Provide all four independently approved uint8 SNP TCB floors")
    challenge = (directory / "request-data.bin").read_bytes()
    report_path = directory / "attestation-report.bin"
    report = report_path.read_bytes()
    if len(challenge) != 64:
        raise ValueError("Rehearsal nonce must have 64 bytes")
    if len(report) < 0x188:
        raise ValueError("SNP attestation report is truncated")
    if not hmac.compare_digest(report[0x50:0x90], challenge):
        raise ValueError("Signed SNP REPORT_DATA does not match the fresh challenge")
    certs = directory / "certs"
    if fetch_certificates:
        certs.mkdir(mode=0o700, exist_ok=True)
        subprocess.run(
            [str(verifier_path), "fetch", "ca", "pem", str(certs), "--report", str(report_path), "--endorser", "vcek"],
            check=True,
            timeout=300,
        )
        subprocess.run(
            [str(verifier_path), "fetch", "vcek", "pem", str(certs), str(report_path)],
            check=True,
            timeout=300,
        )
    if not certs.is_dir():
        raise ValueError("SNP certificate directory is missing")
    certificates = {}
    for certificate in certs.iterdir():
        if certificate.is_symlink() or not certificate.is_file():
            raise ValueError("SNP certificate inputs must be regular files")
        certificates[certificate.name] = certificate.read_bytes()
    with tempfile.TemporaryDirectory(prefix=".snp-verify-", dir=directory) as temporary:
        snapshot = Path(temporary)
        snapshot_report = snapshot / "attestation-report.bin"
        snapshot_report.write_bytes(report)
        snapshot_report.chmod(0o600)
        snapshot_challenge = snapshot / "request-data.bin"
        snapshot_challenge.write_bytes(challenge)
        snapshot_challenge.chmod(0o600)
        snapshot_certs = snapshot / "certs"
        snapshot_certs.mkdir(mode=0o700)
        for name, content in certificates.items():
            (snapshot_certs / name).write_bytes(content)
            (snapshot_certs / name).chmod(0o600)
        for command in (
            [str(verifier_path), "verify", "certs", str(snapshot_certs)],
            [str(verifier_path), "verify", "attestation", str(snapshot_certs), str(snapshot_report)],
            [str(verifier_path), "verify", "attestation", str(snapshot_certs), str(snapshot_report), "--tcb"],
        ):
            subprocess.run(command, check=True, timeout=300)
    tcb = report[0x180:0x188]
    if any(tcb[2:6]):
        raise ValueError("Reserved signed REPORTED_TCB bytes are nonzero")
    reported = dict(zip(TCB_FIELDS, (tcb[0], tcb[1], tcb[6], tcb[7])))
    if any(reported[field] < floors["snp_min_reported_tcb_" + field] for field in TCB_FIELDS):
        raise ValueError("Verified reported TCB is below an independently approved minimum")
    measurement = report[0x90:0xC0].hex()
    values = {"snp_launch_measurement": measurement, **floors}
    summary = {
        "measurement": measurement,
        "reported_tcb": reported,
        "report_sha256": hashlib.sha256(report).hexdigest(),
        "challenge_sha256": hashlib.sha256(challenge).hexdigest(),
    }
    return values, summary
