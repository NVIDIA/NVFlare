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

import json
import os
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[5]
API = runpy.run_path(str(ROOT / "examples/devops/coco/acceptance/trusted_proof_audit.py"))
FIXTURE = runpy.run_path(str(ROOT / "tests/unit_test/app_opt/confidential_computing/coco_authorizer_test.py"))


@pytest.fixture
def audit_path(tmp_path):
    tmp_path.chmod(0o700)
    return tmp_path / "proofs.jsonl"


@pytest.fixture
def signed_material():
    return FIXTURE["material"].__wrapped__(SimpleNamespace(param=("ec", "tdx")))


def test_original_signature_binding_and_replay_results_are_preserved(audit_path, signed_material):
    _, _, verifier, generate, _ = signed_material
    token = generate()
    session = API["install"](audit_path, ["site-1"])
    try:
        assert verifier.verify_for_site(token, "wrong-site") is False
        assert verifier.verify_for_site("not-a-jwt", "site-1") is False
        assert verifier.verify_for_site(token, "site-1") is True
        assert verifier.verify_for_site(token, "site-1") is False
    finally:
        session.close()
    _, records, footer = API["read_audit"](audit_path)
    assert len(records) == footer["record_count"] == 1
    assert set(records[0]) == API["PROOF_FIELDS"]
    saved = audit_path.read_text()
    assert token not in saved
    assert "jti" not in saved and "PRIVATE KEY" not in saved and "annotated-evidence" not in saved
    assert os.stat(audit_path).st_mode & 0o777 == 0o600


@pytest.mark.parametrize("failure", ["privacy", "schema", "issuer"])
def test_audit_failure_latches_without_changing_success(audit_path, signed_material, failure, capsys):
    _, client, verifier, generate, _ = signed_material
    session = API["install"](audit_path, ["site-1"] if failure != "schema" else ["site-2"])
    try:
        if failure == "privacy":
            audit_path.chmod(0o644)
        target = client if failure == "issuer" else verifier
        assert target.verify_for_site(generate(), "site-1") is True
        assert session.failed
        audit_path.chmod(0o600)
        assert verifier.verify_for_site(generate(), "site-1") is True
        assert session.count == 0
    finally:
        session.close()
    assert "renewal evidence is incomplete" in capsys.readouterr().err
    with pytest.raises(API["AuditError"]):
        API["read_audit"](audit_path)


def record(site, issued, observed, ear_issued, ear_exp, fingerprint):
    return dict(
        kind="verified_proof",
        site=site,
        observed_at=observed,
        proof_iat=issued,
        proof_exp=issued + 300,
        ear_iat=ear_issued,
        ear_exp=ear_exp,
        proof_sha256=f"{issued:064x}",
        ear_sha256=f"{fingerprint:064x}",
    )


def write_session(path, rows, completed=1000):
    session = API["AuditSession"](path, ["site-1", "site-2"], clock=lambda: 100)
    for row in sorted(rows, key=lambda r: r["observed_at"]):
        session._append(row)
        session.count += 1
    session.clock = lambda: completed
    session.close()


def renewal_rows():
    return [
        row
        for site in ("site-1", "site-2")
        for row in (
            record(site, 100, 101, 100, 400, 1),
            record(site, 400, 401, 400, 700, 2),
            record(site, 700, 701, 700, 1000, 3),
            record(site, 990, 991, 990, 1290, 4),
        )
    ]


def test_full_renewal_allows_cached_ear_refresh_after_previous_expiry(audit_path):
    write_session(audit_path, renewal_rows())
    result = API["validate_renewal"](audit_path, ["site-1", "site-2"])
    assert result["renewal_evidence_complete"]
    assert result["sites"]["site-1"]["ear_refresh_timing"][0]["seconds_relative_to_previous_expiry"] == 1
    assert not result["fresh_hardware_quote_demonstrated"]


@pytest.mark.parametrize("fault", ["missing-site", "cached-ear", "same-proof", "expired-ear", "short"])
def test_incomplete_renewal_never_qualifies(audit_path, fault):
    rows = renewal_rows()
    if fault == "missing-site":
        rows = [r for r in rows if r["site"] != "site-2"]
    if fault == "cached-ear":
        for r in rows:
            r.update(ear_iat=100, ear_exp=2000, ear_sha256="1" * 64)
    if fault == "same-proof":
        for r in rows:
            r.update(proof_sha256="1" * 64)
    if fault == "expired-ear":
        rows[-1]["ear_exp"] = rows[-1]["observed_at"]
    write_session(audit_path, rows, completed=999 if fault == "short" else 1000)
    assert not API["validate_renewal"](audit_path, ["site-1", "site-2"])["renewal_evidence_complete"]


@pytest.mark.parametrize("fault", ["no-footer", "raw-field", "truncated", "wrong-sites", "unhealthy"])
def test_invalid_session_cannot_be_used(audit_path, fault):
    write_session(audit_path, renewal_rows())
    entries = [json.loads(line) for line in audit_path.read_text().splitlines()]
    if fault == "no-footer":
        entries.pop()
    if fault == "raw-field":
        entries[1]["raw_token"] = "secret"
    if fault == "unhealthy":
        entries[-1]["healthy"] = False
    audit_path.write_text("".join(json.dumps(e) + "\n" for e in entries))
    if fault == "truncated":
        audit_path.write_text(audit_path.read_text().rstrip())
    with pytest.raises(API["AuditError"]):
        API["validate_renewal"](audit_path, ["site-1", "server"] if fault == "wrong-sites" else ["site-1", "site-2"])


def test_original_exception_is_not_swallowed_or_recorded(audit_path, signed_material, monkeypatch):
    from nvflare.app_opt.confidential_computing.coco_authorizer import CoCoAuthorizer

    _, _, verifier, _, _ = signed_material

    def broken_original(self, token, site_name):
        raise RuntimeError("original verifier failure")

    monkeypatch.setattr(CoCoAuthorizer, "verify_for_site", broken_original)
    session = API["install"](audit_path, ["site-1"])
    try:
        with pytest.raises(RuntimeError, match="original verifier failure"):
            verifier.verify_for_site("invalid", "site-1")
        assert not session.failed and session.count == 0
    finally:
        session.close()
    assert CoCoAuthorizer.verify_for_site is broken_original


def test_replaced_output_never_receives_verified_metadata(audit_path, signed_material):
    _, _, verifier, generate, _ = signed_material
    session = API["install"](audit_path, ["site-1"])
    saved = audit_path.with_name("original.jsonl")
    audit_path.rename(saved)
    audit_path.write_text("")
    audit_path.chmod(0o600)
    try:
        assert verifier.verify_for_site(generate(), "site-1") is True
        assert session.failed and session.count == 0
    finally:
        session.close()
    assert audit_path.read_text() == ""
    assert "verified_proof" not in saved.read_text()
    with pytest.raises(API["AuditError"]):
        API["read_audit"](saved)


@pytest.mark.parametrize("fault", ["same-iat", "old-ear-cycle"])
def test_ear_fingerprint_changes_do_not_establish_refresh(audit_path, fault):
    rows = renewal_rows()
    for index, r in enumerate(rows):
        # Long-lived valid EARs isolate renewal from expiration checks.
        if fault == "same-iat":
            r.update(ear_iat=100, ear_exp=2000, ear_sha256=f"{index + 1:064x}")
        else:
            issued = (100, 50, 100, 50)[index % 4]
            r.update(ear_iat=issued, ear_exp=2000, ear_sha256=f"{issued:064x}")
    write_session(audit_path, rows)
    result = API["validate_renewal"](audit_path, ["site-1", "site-2"])
    assert not result["renewal_evidence_complete"]
    for site in result["sites"].values():
        assert site["proof_renewal"]
        assert not site["ear_refresh"]
        assert site["ear_refresh_timing"] == []


def test_ear_timing_only_records_new_issuance_high_water_marks(audit_path):
    rows = renewal_rows()
    for index, r in enumerate(rows):
        issued = (100, 200, 100, 200)[index % 4]
        r.update(ear_iat=issued, ear_exp=2000, ear_sha256=f"{issued:064x}")
    write_session(audit_path, rows)
    result = API["validate_renewal"](audit_path, ["site-1", "site-2"])
    assert result["renewal_evidence_complete"]
    for site in result["sites"].values():
        assert site["ear_refresh_timing"] == [{"observed_at": 401, "seconds_relative_to_previous_expiry": -1599}]
