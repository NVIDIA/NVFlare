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

"""Offline ledger checks cannot certify hardware acceptance."""

import json
import runpy
import stat
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[5]
EVIDENCE = runpy.run_path(str(ROOT / "examples/devops/coco/acceptance/evidence.py"))


@pytest.fixture
def ledger(tmp_path):
    manifest = tmp_path / "manifest.json"
    assert EVIDENCE["main"](["init", "--manifest", str(manifest), "--run-id", "test-1", "--git-commit", "a" * 40]) == 0
    proof = tmp_path / "proof.json"
    proof.write_text('{"sanitized": true}\n')
    return manifest, proof


def record(manifest, proof, topology="A", case="P1", scope="hardware", status="PASS"):
    args = [
        "record",
        "--manifest",
        str(manifest),
        "--topology",
        topology,
        "--scope",
        scope,
        "--case",
        case,
        "--status",
        status,
    ]
    if proof:
        args.extend(["--evidence", str(proof)])
    return EVIDENCE["main"](args)


def fill(manifest, proof, cases, scope="hardware", topologies=("A", "B")):
    for topology in topologies:
        for case in cases:
            assert record(manifest, proof, topology, case, scope) == 0


def report(manifest):
    return EVIDENCE["assess"](EVIDENCE["load"](manifest))


def test_new_private_manifest_preserves_default_open_gap(ledger):
    manifest, _ = ledger
    assert stat.S_IMODE(manifest.stat().st_mode) == 0o600
    result = report(manifest)
    assert not result["functional_acceptance"]
    assert not result["full_security_qualification"]
    assert len(result["unresolved"]) == 2 * len(EVIDENCE["CASES"])
    assert {r["status"] for r in result["unresolved"] if r["case"] == "image-binding"} == {"OPEN"}


def test_offline_passes_cannot_satisfy_hardware(ledger):
    manifest, proof = ledger
    fill(manifest, proof, EVIDENCE["CASES"], scope="offline")
    assert not report(manifest)["functional_acceptance"]
    assert not report(manifest)["full_security_qualification"]


def test_requires_both_topologies_and_all_positive_cases(ledger):
    manifest, proof = ledger
    fill(manifest, proof, EVIDENCE["POSITIVES"], topologies=("A",))
    assert not report(manifest)["functional_acceptance"]
    fill(manifest, proof, EVIDENCE["POSITIVES"], topologies=("B",))
    assert report(manifest)["functional_acceptance"]
    assert not report(manifest)["full_security_qualification"]
    assert all(item["case"] in EVIDENCE["NEGATIVES"] for item in report(manifest)["unresolved"])


def test_security_requires_denials_and_gap_resolution(ledger):
    manifest, proof = ledger
    fill(manifest, proof, EVIDENCE["CASES"])
    assert report(manifest)["full_security_qualification"]
    assert EVIDENCE["main"](["assess", "--manifest", str(manifest)]) == 0
    record(manifest, None, "B", "image-binding", status="OPEN")
    assert report(manifest)["functional_acceptance"]
    assert not report(manifest)["full_security_qualification"]


def test_latest_failed_hardware_result_overrides_prior_pass_but_not_by_offline(ledger):
    manifest, proof = ledger
    fill(manifest, proof, EVIDENCE["CASES"])
    record(manifest, None, "A", "F3", status="FAIL")
    record(manifest, proof, "A", "F3", scope="offline")
    result = report(manifest)
    assert not result["functional_acceptance"]
    assert result["unresolved"] == [{"topology": "A", "case": "F3", "status": "FAIL"}]
    assert len(EVIDENCE["load"](manifest)["records"]) == 2 * len(EVIDENCE["CASES"]) + 2


@pytest.mark.parametrize("missing", (False, True))
def test_modified_or_missing_evidence_invalidates_claim(ledger, missing):
    manifest, proof = ledger
    fill(manifest, proof, EVIDENCE["CASES"])
    if missing:
        proof.unlink()
    else:
        proof.write_text("tampered")
    result = report(manifest)
    assert not result["functional_acceptance"]
    assert not result["full_security_qualification"]
    expected = "EVIDENCE_UNAVAILABLE" if missing else "EVIDENCE_CHANGED"
    assert {item["status"] for item in result["unresolved"]} == {expected}


def test_pass_without_evidence_rejected_without_changing_manifest(ledger):
    manifest, _ = ledger
    original = manifest.read_bytes()
    with pytest.raises(SystemExit) as error:
        record(manifest, None)
    assert error.value.code == 1
    assert manifest.read_bytes() == original


def test_manifest_records_only_file_hashes_and_no_file_contents(ledger):
    manifest, proof = ledger
    proof.write_text("DO-NOT-EMBED-THIS-CONTENT")
    record(manifest, proof)
    assert "DO-NOT-EMBED-THIS-CONTENT" not in manifest.read_text()
    item = EVIDENCE["load"](manifest)["records"][0]["evidence"][0]
    assert item == EVIDENCE["fingerprint"](proof)
    assert stat.S_IMODE(manifest.stat().st_mode) == 0o600


def test_init_never_overwrites_existing_manifest(ledger):
    manifest, _ = ledger
    original = manifest.read_bytes()
    with pytest.raises(SystemExit):
        EVIDENCE["main"](["init", "--manifest", str(manifest), "--run-id", "other", "--git-commit", "b" * 40])
    assert manifest.read_bytes() == original


@pytest.mark.parametrize("change", ({"schema_version": 2}, {"records": {}}, {"git_commit": "2.9"}))
def test_malformed_manifest_rejected(ledger, change):
    manifest, _ = ledger
    raw = json.loads(manifest.read_text())
    raw.update(change)
    manifest.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        EVIDENCE["load"](manifest)


def test_metadata_kept_private_and_assessment_does_not_echo_it(ledger, capsys):
    manifest, _ = ledger
    raw = EVIDENCE["load"](manifest)
    raw["hosts"] = {"private": "INTERNAL-HOST-IDENTITY"}
    EVIDENCE["save"](manifest, raw)
    assert EVIDENCE["main"](["assess", "--manifest", str(manifest)]) == 2
    assert "INTERNAL-HOST-IDENTITY" not in capsys.readouterr().out
