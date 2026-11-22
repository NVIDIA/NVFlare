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

"""Fail-closed SNP envelope and signed-report verification checks."""

import base64
import copy
import gzip
import hashlib
import importlib.util
import io
import subprocess
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[5]
HELPER = ROOT / "examples/devops/coco/trusted_system/lib/platform-evidence.py"
SPEC = importlib.util.spec_from_file_location("platform_evidence", HELPER)
evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence)


def snp_report(challenge=b"a" * 64):
    report = bytearray(1184)
    report[0x50:0x90] = challenge
    report[0x90:0xC0] = b"m" * 48
    report[0x180:0x188] = bytes([1, 2, 0, 0, 0, 0, 3, 4])
    return report


def prepare_report(tmp_path):
    (tmp_path / "request-data.bin").write_bytes(b"a" * 64)
    (tmp_path / "attestation-report.bin").write_bytes(snp_report())
    (tmp_path / "certs").mkdir()
    return {field: 0 for field in evidence.FLOOR_FIELDS}


def archive_frame(files, members=None):
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as output:
        for name, content, kind in members or [(name, content, tarfile.REGTYPE) for name, content in files.items()]:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            member.type = kind
            if kind == tarfile.SYMTYPE:
                member.linkname = "../../outside"
            output.addfile(member, io.BytesIO(content))
    return evidence.SNP_FRAME + base64.b64encode(gzip.compress(archive.getvalue())).decode()


def archive_files():
    files = {
        "attestation-report.bin": bytes(snp_report()),
        "attestation-report.txt": b"diagnostic text is not authoritative\n",
        "request-data.bin": b"a" * 64,
    }
    files["SHA256SUMS"] = "".join(
        hashlib.sha256(content).hexdigest() + "  ./" + name + "\n" for name, content in sorted(files.items())
    ).encode()
    return files


def test_snp_floors_remain_independent_of_verified_report(tmp_path):
    floors = prepare_report(tmp_path)
    original = copy.deepcopy(floors)
    with patch.object(evidence.subprocess, "run") as verifier:
        values, summary = evidence.verify_snp(tmp_path, "reviewed-snpguest", floors)
    assert floors == original
    assert values == {"snp_launch_measurement": (b"m" * 48).hex(), **original}
    assert summary["reported_tcb"] == {"bootloader": 1, "tee": 2, "snp": 3, "microcode": 4}
    assert summary["report_sha256"] == hashlib.sha256(snp_report()).hexdigest()
    commands = [call.args[0] for call in verifier.call_args_list]
    certificates = Path(commands[0][-1])
    report = certificates.parent / "attestation-report.bin"
    assert certificates.parent.parent == tmp_path
    assert certificates.parent.name.startswith(".snp-verify-")
    assert [command[1:] for command in commands] == [
        ["verify", "certs", str(certificates)],
        ["verify", "attestation", str(certificates), str(report)],
        ["verify", "attestation", str(certificates), str(report), "--tcb"],
    ]
    assert not certificates.parent.exists()
    assert all(call.kwargs["check"] is True for call in verifier.call_args_list)


@pytest.mark.parametrize("field", evidence.FLOOR_FIELDS)
@pytest.mark.parametrize("value", [None, True, False, -1, 256, "0", 0.0])
def test_snp_invalid_floors_fail_before_verification(tmp_path, field, value):
    floors = prepare_report(tmp_path)
    floors[field] = value
    with patch.object(evidence.subprocess, "run") as verifier:
        with pytest.raises(ValueError, match="independently approved"):
            evidence.verify_snp(tmp_path, "reviewed-snpguest", floors)
    verifier.assert_not_called()


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_snp_requires_exact_floor_fields(tmp_path, mutation):
    floors = prepare_report(tmp_path)
    if mutation == "missing":
        del floors[evidence.FLOOR_FIELDS[0]]
    else:
        floors["unexpected_floor"] = 0
    with patch.object(evidence.subprocess, "run") as verifier:
        with pytest.raises(ValueError, match="independently approved"):
            evidence.verify_snp(tmp_path, "reviewed-snpguest", floors)
    verifier.assert_not_called()


@pytest.mark.parametrize("field", evidence.FLOOR_FIELDS)
def test_snp_tcb_below_independent_floor_is_rejected(tmp_path, field):
    floors = prepare_report(tmp_path)
    floors[field] = 5
    with patch.object(evidence.subprocess, "run"):
        with pytest.raises(ValueError, match="below an independently"):
            evidence.verify_snp(tmp_path, "reviewed-snpguest", floors)
    assert floors[field] == 5


@pytest.mark.parametrize("failed_step", range(3))
def test_snp_certificate_signature_or_tcb_failure_cannot_return_values(tmp_path, failed_step):
    floors = prepare_report(tmp_path)
    results = [subprocess.CompletedProcess([], 0)] * failed_step
    results.append(subprocess.CalledProcessError(1, ["reviewed-snpguest"]))
    with patch.object(evidence.subprocess, "run", side_effect=results):
        with pytest.raises(subprocess.CalledProcessError):
            evidence.verify_snp(tmp_path, "reviewed-snpguest", floors)


@pytest.mark.parametrize("mutation", ["challenge", "truncated", "reserved"])
def test_snp_signed_fields_fail_closed(tmp_path, mutation):
    floors = prepare_report(tmp_path)
    report = snp_report()
    if mutation == "challenge":
        report[0x50] ^= 1
    elif mutation == "truncated":
        report = report[:0x187]
    else:
        report[0x182] = 1
    (tmp_path / "attestation-report.bin").write_bytes(report)
    with patch.object(evidence.subprocess, "run"):
        with pytest.raises(ValueError):
            evidence.verify_snp(tmp_path, "reviewed-snpguest", floors)


@pytest.mark.parametrize("permissions", [1 << 18, 1 << 19, (1 << 18) | (1 << 19)])
def test_snp_signed_policy_rejects_debug_and_migration_permissions(tmp_path, permissions):
    floors = prepare_report(tmp_path)
    report = snp_report()
    report[0x08:0x10] = permissions.to_bytes(8, "little")
    (tmp_path / "attestation-report.bin").write_bytes(report)
    with patch.object(evidence.subprocess, "run") as verifier:
        with pytest.raises(ValueError, match="disable debug and migration-agent"):
            evidence.verify_snp(tmp_path, "reviewed-snpguest", floors)
    assert verifier.call_count == 3


def test_snp_optional_certificate_fetch_still_verifies_chain_and_report(tmp_path):
    floors = prepare_report(tmp_path)
    with patch.object(evidence.subprocess, "run") as verifier:
        evidence.verify_snp(tmp_path, "reviewed-snpguest", floors, fetch_certificates=True)
    commands = [call.args[0] for call in verifier.call_args_list]
    assert len(commands) == 5
    assert commands[0][1:4] == ["fetch", "ca", "pem"]
    assert commands[1][1:4] == ["fetch", "vcek", "pem"]
    assert commands[2][1:3] == ["verify", "certs"]
    assert commands[-1][-1] == "--tcb"


def test_snp_decode_checks_exact_envelope_and_retains_original_bytes(tmp_path):
    files = archive_files()
    (tmp_path / "request-data.bin").write_bytes(files["request-data.bin"])
    result = evidence.decode_snp(archive_frame(files), tmp_path)
    assert result == tmp_path / "evidence-input"
    assert {path.name: path.read_bytes() for path in result.iterdir()} == files
    with pytest.raises(FileExistsError):
        evidence.decode_snp(archive_frame(files), tmp_path)


@pytest.mark.parametrize("mutation", ["extra", "duplicate", "path", "symlink", "checksum", "missing_checksum", "nonce"])
def test_snp_decode_rejects_archive_tampering_before_extracting(tmp_path, mutation):
    files = archive_files()
    (tmp_path / "request-data.bin").write_bytes(files["request-data.bin"])
    members = [(name, content, tarfile.REGTYPE) for name, content in files.items()]
    if mutation == "extra":
        members.append(("extra", b"unexpected", tarfile.REGTYPE))
    elif mutation == "duplicate":
        members.append(members[0])
    elif mutation == "path":
        members[0] = ("../attestation-report.bin", members[0][1], tarfile.REGTYPE)
    elif mutation == "symlink":
        members[0] = (members[0][0], b"", tarfile.SYMTYPE)
    elif mutation == "checksum":
        members[0] = (members[0][0], b"changed report", tarfile.REGTYPE)
    elif mutation == "missing_checksum":
        members[-1] = ("SHA256SUMS", b"", tarfile.REGTYPE)
    else:
        (tmp_path / "request-data.bin").write_bytes(b"b" * 64)
    with pytest.raises(ValueError):
        evidence.decode_snp(archive_frame(files, members), tmp_path)
    assert not (tmp_path / "evidence-input").exists()


@pytest.mark.parametrize(
    "frame", ["", "COCO_SNP_EVIDENCE_V1=notbase64", "COCO_SNP_EVIDENCE_V1=AA==\nCOCO_SNP_EVIDENCE_V1=AA=="]
)
def test_snp_decode_rejects_invalid_framing(tmp_path, frame):
    (tmp_path / "request-data.bin").write_bytes(b"a" * 64)
    with pytest.raises(ValueError):
        evidence.decode_snp(frame, tmp_path)


def test_snp_parses_and_verifies_same_private_report_when_source_is_replaced(tmp_path):
    floors = prepare_report(tmp_path)
    original_report = bytes(snp_report())
    certificate = tmp_path / "certs/vcek.pem"
    certificate.write_bytes(b"reviewed certificate")
    snapshots = []

    def replace_original_inputs(command, **kwargs):
        certs = Path(command[3])
        snapshot = certs.parent
        snapshots.append(snapshot)
        altered = snp_report()
        altered[0x90:0xC0] = b"x" * 48
        (tmp_path / "attestation-report.bin").write_bytes(altered)
        (tmp_path / "request-data.bin").write_bytes(b"replaced original challenge")
        certificate.write_bytes(b"replaced original certificate")
        assert (snapshot / "attestation-report.bin").read_bytes() == original_report
        assert (snapshot / "request-data.bin").read_bytes() == b"a" * 64
        assert (certs / "vcek.pem").read_bytes() == b"reviewed certificate"
        assert snapshot.stat().st_mode & 0o777 == 0o700
        assert certs.stat().st_mode & 0o777 == 0o700
        assert (snapshot / "attestation-report.bin").stat().st_mode & 0o777 == 0o600
        return subprocess.CompletedProcess(command, 0)

    with patch.object(evidence.subprocess, "run", side_effect=replace_original_inputs):
        values, summary = evidence.verify_snp(tmp_path, "reviewed-snpguest", floors)
    assert values["snp_launch_measurement"] == (b"m" * 48).hex()
    assert summary["report_sha256"] == hashlib.sha256(original_report).hexdigest()
    assert len(snapshots) == 3
    assert all(not snapshot.exists() for snapshot in snapshots)
