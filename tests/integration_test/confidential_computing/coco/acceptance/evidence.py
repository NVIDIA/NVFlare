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

"""Private acceptance evidence ledger; recorded PASS is an operator assertion.

This tool hashes evidence without embedding its contents. It does not verify
quotes, logs, or operator claims. Keep evidence and metadata private and sanitized;
never supply tokens, private keys, credentials, or guest token API responses.
"""

import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

POSITIVES = ("P1", "P2", "H1", "H2", "K1", "F1", "F2", "F3", "F4", "C1", "R1")
NEGATIVES = ("platform", "kbs", "guest", "authorizer", "federation", "availability", "image-binding")
CASES = POSITIVES + NEGATIVES
STATUSES = ("PASS", "FAIL", "BLOCKED", "NOT_RUN", "OPEN")


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("evidence must be a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def validate(manifest):
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("unsupported manifest schema")
    if not isinstance(manifest.get("run_id"), str) or not manifest["run_id"]:
        raise ValueError("missing run ID")
    if not re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", manifest.get("git_commit", "")):
        raise ValueError("git commit must be a full hexadecimal commit ID")
    if not isinstance(manifest.get("records"), list):
        raise ValueError("records must be a list")
    for record in manifest["records"]:
        if not isinstance(record, dict):
            raise ValueError("invalid record")
        if record.get("topology") not in ("A", "B") or record.get("scope") not in ("hardware", "offline"):
            raise ValueError("invalid topology or scope")
        if record.get("case") not in CASES or record.get("status") not in STATUSES:
            raise ValueError("invalid case or status")
        evidence = record.get("evidence")
        if not isinstance(evidence, list) or (record["status"] == "PASS" and not evidence):
            raise ValueError("PASS requires evidence")
        for item in evidence:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise ValueError("invalid evidence reference")
            if not Path(item["path"]).is_absolute() or not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")):
                raise ValueError("invalid evidence path or digest")
            if type(item.get("bytes")) is not int or item["bytes"] < 0:
                raise ValueError("invalid evidence size")
    return manifest


def load(path):
    if Path(path).is_symlink():
        raise ValueError("manifest must not be a symbolic link")
    with Path(path).open(encoding="utf-8") as stream:
        return validate(json.load(stream))


def save(path, manifest, create=False):
    validate(manifest)
    path = Path(path)
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if create:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
        return
    if path.is_symlink():
        raise ValueError("manifest must not be a symbolic link")
    descriptor, temporary = tempfile.mkstemp(prefix=".evidence-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def assess(manifest):
    """Require latest hardware assertions and unchanged evidence for each cell."""
    validate(manifest)
    latest = {(r["topology"], r["scope"], r["case"]): r for r in manifest["records"]}
    unresolved = []
    for topology in ("A", "B"):
        for case in CASES:
            record = latest.get((topology, "hardware", case))
            status = record["status"] if record else ("OPEN" if case == "image-binding" else "NOT_RUN")
            if status == "PASS":
                try:
                    if any(fingerprint(item["path"]) != item for item in record["evidence"]):
                        status = "EVIDENCE_CHANGED"
                except (OSError, ValueError):
                    status = "EVIDENCE_UNAVAILABLE"
            if status != "PASS":
                unresolved.append({"topology": topology, "case": case, "status": status})
    functional = not any(item["case"] in POSITIVES for item in unresolved)
    return {
        "run_id": manifest["run_id"],
        "functional_acceptance": functional,
        "full_security_qualification": functional and not unresolved,
        "unresolved": unresolved,
        "assurance": "Operator assertions with file integrity checks; evidence content is not verified by this tool.",
    }


def metadata(path):
    if not path:
        return {}
    with Path(path).open(encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError("metadata must be a JSON object")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--manifest", required=True)
    init.add_argument("--run-id", required=True)
    init.add_argument("--git-commit", required=True)
    init.add_argument("--revisions-file", help="Sanitized JSON object of runtime/service/image revisions")
    init.add_argument("--hosts-file", help="Sanitized JSON object of hardware identities and service endpoints")
    record = commands.add_parser("record")
    record.add_argument("--manifest", required=True)
    record.add_argument("--topology", choices=("A", "B"), required=True)
    record.add_argument("--scope", choices=("hardware", "offline"), required=True)
    record.add_argument("--case", choices=CASES, required=True)
    record.add_argument("--status", choices=STATUSES, required=True)
    record.add_argument("--evidence", action="append", default=[], help="Private sanitized evidence file; repeatable")
    record.add_argument("--note", default="", help="Sanitized operator observation; no secrets")
    report = commands.add_parser("assess")
    report.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            save(
                args.manifest,
                {
                    "schema_version": 1,
                    "run_id": args.run_id,
                    "git_commit": args.git_commit,
                    "created_at": now(),
                    "revisions": metadata(args.revisions_file),
                    "hosts": metadata(args.hosts_file),
                    "records": [],
                },
                create=True,
            )
        elif args.command == "record":
            manifest = load(args.manifest)
            manifest["records"].append(
                {
                    "recorded_at": now(),
                    "topology": args.topology,
                    "scope": args.scope,
                    "case": args.case,
                    "status": args.status,
                    "evidence": [fingerprint(path) for path in args.evidence],
                    "note": args.note,
                }
            )
            save(args.manifest, manifest)
        else:
            result = assess(load(args.manifest))
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["full_security_qualification"] else 2
    except (OSError, ValueError, TypeError) as error:
        # Avoid echoing supplied metadata, file contents, or potentially secret paths.
        parser.exit(1, f"Evidence operation failed ({type(error).__name__}); check inputs privately.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
