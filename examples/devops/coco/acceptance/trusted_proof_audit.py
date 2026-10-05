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

"""Opt-in trusted observer proof metadata capture; never a replacement verifier.

Install only in the ordinary issuer-free observer process, before it starts.
Validation requires a graceful session close; audit failure never changes the
original verifier's result. No raw token, jti, key, or annotated evidence is saved.
"""

import argparse
import atexit
import functools
import hashlib
import json
import math
import os
import re
import stat
import sys
import threading
import time
from pathlib import Path

import jwt

SCHEMA = "nvflare-trusted-proof-audit/v1"
PROOF_FIELDS = {
    "kind",
    "site",
    "observed_at",
    "proof_iat",
    "proof_exp",
    "ear_iat",
    "ear_exp",
    "proof_sha256",
    "ear_sha256",
}


class AuditError(ValueError):
    pass


def valid_time(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def check_record(record, sites):
    if not isinstance(record, dict) or set(record) != PROOF_FIELDS or record["kind"] != "verified_proof":
        raise AuditError("Invalid sanitized proof schema")
    if record["site"] not in sites or not valid_time(record["observed_at"]):
        raise AuditError("Invalid observer/site metadata")
    for prefix in ("proof", "ear"):
        issued, expires = record[prefix + "_iat"], record[prefix + "_exp"]
        if type(issued) is not int or type(expires) is not int or issued < 0 or expires <= issued:
            raise AuditError("Invalid verified token time metadata")
        if not isinstance(record[prefix + "_sha256"], str) or not re.fullmatch(
            r"[a-f0-9]{64}", record[prefix + "_sha256"]
        ):
            raise AuditError("Invalid token fingerprint")
    return record


def verified_metadata(token, site, observed_at, sites):
    # The unchanged original verifier has already authenticated both JWTs and
    # independently bound this site. Decode only to select the sanitized fields.
    proof = jwt.decode(token, options={"verify_signature": False})
    ear_token = proof["ear"]
    if proof.get("sub") != site or not isinstance(ear_token, str):
        raise AuditError("Verified metadata did not match authenticated site")
    ear = jwt.decode(ear_token, options={"verify_signature": False})
    return check_record(
        {
            "kind": "verified_proof",
            "site": site,
            "observed_at": observed_at,
            "proof_iat": proof["iat"],
            "proof_exp": proof["exp"],
            "ear_iat": ear["iat"],
            "ear_exp": ear["exp"],
            "proof_sha256": hashlib.sha256(token.encode()).hexdigest(),
            "ear_sha256": hashlib.sha256(ear_token.encode()).hexdigest(),
        },
        sites,
    )


class AuditSession:
    def __init__(self, path, sites, clock=time.time):
        self.path = Path(path)
        self.sites = tuple(sorted(set(sites)))
        if not self.sites or any(not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", s) for s in self.sites):
            raise AuditError("Expected sites must be sanitized authenticated identities")
        parent = self.path.parent.stat()
        if parent.st_uid != os.getuid() or stat.S_IMODE(parent.st_mode) & 0o077:
            raise AuditError("Audit parent must be private and owned by the observer account")
        self.clock = clock
        self.lock = threading.Lock()
        self.failed = False
        self.closed = False
        self.count = 0
        self.restore = None
        self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        try:
            self._append(
                {"kind": "audit_start", "schema": SCHEMA, "expected_sites": list(self.sites), "started_at": clock()}
            )
        except Exception:
            os.close(self.fd)
            raise AuditError("Unable to initialize private audit") from None

    def _append(self, record):
        current, opened = self.path.lstat(), os.fstat(self.fd)
        if current.st_ino != opened.st_ino or current.st_dev != opened.st_dev:
            raise AuditError("Audit file identity changed")
        if not stat.S_ISREG(current.st_mode) or stat.S_IMODE(current.st_mode) != 0o600:
            raise AuditError("Audit file privacy changed")
        payload = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        if os.write(self.fd, payload) != len(payload):
            raise AuditError("Incomplete append")
        os.fsync(self.fd)

    def _fail(self):
        if not self.failed:
            print("Trusted proof audit failed; renewal evidence is incomplete.", file=sys.stderr)
        self.failed = True

    def capture(self, verifier, token, site):
        with self.lock:
            if self.failed or self.closed:
                self._fail()
                return
            try:
                if verifier.site_name is not None:
                    raise AuditError("Hook is restricted to the ordinary verifier-only observer")
                record = verified_metadata(token, site, self.clock(), self.sites)
                self._append(record)
                self.count += 1
            except Exception:
                self._fail()

    def close(self):
        with self.lock:
            if self.closed:
                return
            try:
                self._append(
                    {
                        "kind": "audit_complete",
                        "completed_at": self.clock(),
                        "record_count": self.count,
                        "healthy": not self.failed,
                    }
                )
            except Exception:
                self._fail()
            finally:
                os.close(self.fd)
                self.closed = True
                if self.restore:
                    self.restore()


def install(path, expected_sites):
    """Return a session installed only in this ordinary observer Python process."""
    from nvflare.app_opt.confidential_computing.coco_authorizer import CoCoAuthorizer

    original = CoCoAuthorizer.verify_for_site
    if getattr(original, "_trusted_proof_audit", False):
        raise AuditError("Observer audit is already installed")
    session = AuditSession(path, expected_sites)

    @functools.wraps(original)
    def wrapped(self, token, site_name):
        result = original(self, token, site_name)
        if result is True:
            session.capture(self, token, site_name)
        return result

    wrapped._trusted_proof_audit = True
    CoCoAuthorizer.verify_for_site = wrapped

    def restore():
        if CoCoAuthorizer.verify_for_site is wrapped:
            CoCoAuthorizer.verify_for_site = original

    session.restore = restore
    atexit.register(session.close)
    return session


def read_audit(path):
    path = Path(path)
    info = path.lstat()
    if info.st_uid != os.getuid() or not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
        raise AuditError("Private regular audit file required")
    entries = []
    with path.open() as stream:
        for line in stream:
            if len(line) > 4096 or not line.endswith("\n"):
                raise AuditError("Malformed or incomplete audit line")
            entries.append(json.loads(line))
    if len(entries) < 2:
        raise AuditError("Missing audit boundaries")
    header, footer = entries[0], entries[-1]
    if not isinstance(header, dict) or set(header) != {"kind", "schema", "expected_sites", "started_at"}:
        raise AuditError("Invalid audit header")
    if header["kind"] != "audit_start" or header["schema"] != SCHEMA or not valid_time(header["started_at"]):
        raise AuditError("Invalid audit session")
    sites = header["expected_sites"]
    if (
        not isinstance(sites, list)
        or not sites
        or any(not isinstance(s, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", s) for s in sites)
        or len(set(sites)) != len(sites)
    ):
        raise AuditError("Invalid site set")
    if not isinstance(footer, dict) or set(footer) != {"kind", "completed_at", "record_count", "healthy"}:
        raise AuditError("Missing clean session completion")
    if (
        footer["kind"] != "audit_complete"
        or footer["healthy"] is not True
        or type(footer["record_count"]) is not int
        or footer["record_count"] != len(entries) - 2
        or not valid_time(footer["completed_at"])
    ):
        raise AuditError("Unhealthy or incomplete audit session")
    previous = header["started_at"]
    for record in entries[1:-1]:
        check_record(record, sites)
        if not previous <= record["observed_at"] <= footer["completed_at"]:
            raise AuditError("Audit timestamps do not match append order/session")
        previous = record["observed_at"]
    return header, entries[1:-1], footer


def validate_renewal(path, expected_sites, proof_lifetime=300):
    header, records, footer = read_audit(path)
    if set(expected_sites) != set(header["expected_sites"]) or proof_lifetime != 300:
        raise AuditError("Expected topology sites and 300-second proof lifetime must match")
    results = {}
    for site in sorted(set(expected_sites)):
        observed = [r for r in records if r["site"] == site]
        proof_ok = False
        ear_ok = False
        reasons = []
        if observed:
            first = observed[0]
            proof_ok = (
                all(r["proof_exp"] - r["proof_iat"] == proof_lifetime for r in observed)
                and all(r["proof_iat"] - 180 <= r["observed_at"] < r["proof_exp"] for r in observed)
                and len({r["proof_iat"] for r in observed}) >= 2
                and len({r["proof_sha256"] for r in observed}) >= 2
                and any(
                    r["proof_iat"] >= first["proof_exp"] and r["observed_at"] >= first["proof_exp"] for r in observed
                )
            )
            # A different signature/encoding is not a new attestation result.
            # Compare authenticated issuance times against the running maximum,
            # so cycling older EARs cannot create another refresh transition.
            advances = []
            latest = observed[0]
            seen_hashes = {latest["ear_sha256"]}
            for r in observed[1:]:
                if r["ear_iat"] > latest["ear_iat"]:
                    if r["ear_sha256"] not in seen_hashes:
                        advances.append((latest, r))
                    latest = r
                seen_hashes.add(r["ear_sha256"])
            ear_ok = bool(advances) and all(r["ear_iat"] - 180 <= r["observed_at"] < r["ear_exp"] for r in observed)
        if not proof_ok:
            reasons.append(
                "Fresh proof timestamps/fingerprints did not demonstrate renewal across initial 300s expiration."
            )
        if not ear_ok:
            reasons.append("A distinct valid EAR with a later authenticated issuance time was not observed.")
        timing = (
            [
                {
                    "observed_at": new["observed_at"],
                    "seconds_relative_to_previous_expiry": new["observed_at"] - old["ear_exp"],
                }
                for old, new in advances
            ]
            if observed
            else []
        )
        results[site] = {
            "proof_renewal": proof_ok,
            "ear_refresh": ear_ok,
            "ear_refresh_timing": timing,
            "observations": len(observed),
            "reasons": reasons,
        }
    return {
        "schema": "nvflare-trusted-proof-renewal/v1",
        "proof_lifetime_seconds": proof_lifetime,
        "session_completed_at": footer["completed_at"],
        "sites": results,
        "observation_seconds": footer["completed_at"] - header["started_at"],
        "renewal_evidence_complete": footer["completed_at"] - header["started_at"] >= 900
        and all(r["proof_renewal"] and r["ear_refresh"] for r in results.values()),
        "fresh_hardware_quote_demonstrated": False,
        "scope": "Trusted observer metadata from unchanged successful peer verification; this is only the renewal portion of F4.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--expected-sites", required=True, nargs="+")
    args = parser.parse_args(argv)
    try:
        result = validate_renewal(args.audit, args.expected_sites)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Trusted renewal audit invalid ({type(error).__name__}); inspect metadata privately.\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["renewal_evidence_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
