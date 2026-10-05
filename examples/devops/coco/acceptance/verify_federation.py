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

"""Trusted admin/observer acceptance check; never run on an untrusted cluster host.

Only reads ordinary observer logs. Does not retrieve protected logs or claim that
periodic EAR/proof validation is fresh hardware quote verification. Secure admin
and observer kits and the observer process must be independently authenticated.
"""

import argparse
import hashlib
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

CLIENTS = {"site-1", "site-2"}
MEMBERS = CLIENTS | {"site-observer"}
VALUES = {"site-1": 3, "site-2": 7}


class AcceptanceError(RuntimeError):
    """A sanitized acceptance failure, without raw logs or server responses."""


def validate_options(timeout, soak):
    if not math.isfinite(timeout) or not 0 < timeout <= 600:
        raise AcceptanceError("Timeout must be finite and in (0, 600] seconds")
    if not math.isfinite(soak) or soak < 0 or 0 < soak < 900:
        raise AcceptanceError("Soak must be zero (partial run) or at least 900 seconds")


class FreshObserverLog:
    """Consume complete new lines, rejecting replaced, truncated, or overwritten logs."""

    def __init__(self, path):
        self.path = Path(path)
        self.stream = self.path.open("rb")
        st = os.fstat(self.stream.fileno())
        self.identity = (st.st_dev, st.st_ino)
        self.offset = st.st_size
        self.pending = b""
        self.periodic = False
        self.rounds = 0
        self.last_success = None
        self.sentinel = self._sentinel()
        self.discard_partial = self.offset > 0 and not self.sentinel.endswith(b"\n")

    def _sentinel(self):
        self.stream.seek(max(0, self.offset - 64))
        return self.stream.read(min(64, self.offset))

    def poll(self, now):
        st = self.path.stat()
        if (st.st_dev, st.st_ino) != self.identity or st.st_size < self.offset:
            raise AcceptanceError("Observer log rotated or truncated; restart observation")
        if self._sentinel() != self.sentinel:
            raise AcceptanceError("Observer log was overwritten; restart observation")
        self.stream.seek(self.offset)
        chunk = self.stream.read()
        self.offset += len(chunk)
        self.sentinel = self._sentinel()
        lines = (self.pending + chunk).split(b"\n")
        self.pending = lines.pop()
        if lines and self.discard_partial:
            lines.pop(0)
            self.discard_partial = False
        if len(self.pending) > 1024 * 1024:
            raise AcceptanceError("Observer log line exceeds safe observation bound")
        for raw in lines:
            line = raw.decode("utf-8", errors="replace")
            if "CCManager" not in line:
                continue
            if any(
                x in line
                for x in (
                    "Cross-site validation failed",
                    "Exception in cross-site validation",
                    "No tokens collected for validation",
                    "Stopping cross-site validation",
                )
            ):
                raise AcceptanceError("Observer CCManager reported failure or shutdown")
            if line.rstrip().endswith("Site site-observer triggering periodic cross-site validation"):
                self.periodic = True
            elif line.rstrip().endswith("Cross-site validation passed") and self.periodic:
                self.rounds += 1
                self.last_success = now
                self.periodic = False
        return self.rounds

    def close(self):
        self.stream.close()


def verify_membership(session):
    names = [c.name for c in session.get_system_info().client_info]
    if len(names) != len(MEMBERS) or set(names) != MEMBERS:
        raise AcceptanceError("Authenticated federation membership differs from expected three clients")


def verify_observer_config(local, topology):
    components = {}
    for name in ("cc_manager", "coco_authorizer"):
        config = json.loads((Path(local) / f"{name}__p_resources.json").read_text())
        for component in config["components"]:
            if component["id"] in components:
                raise AcceptanceError("Duplicate observer CC component")
            components[component["id"]] = component
    manager = components["cc_manager"]
    args = manager["args"]
    required = CLIENTS | ({"server"} if topology == "B" else set())
    if (
        manager["path"] != "nvflare.app_opt.confidential_computing.cc_manager.CCManager"
        or args.get("cc_issuers_conf") != []
        or args.get("cc_verifier_ids") != ["coco_authorizer"]
        or sorted(args.get("cc_enabled_sites", [])) != sorted(required)
        or args.get("required_site_verifier_ids") != {n: ["coco_authorizer"] for n in required}
        or args.get("require_site_binding") is not True
        or args.get("verify_frequency") != 120
        or components["coco_authorizer"]["path"]
        != "nvflare.app_opt.confidential_computing.coco_authorizer.CoCoAuthorizer"
    ):
        raise AcceptanceError("Observer must be verifier-only with complete bound CC coverage")


def verify_job_definition(job, nonce):
    root = Path(job)
    if root.is_symlink() or not root.is_dir():
        raise AcceptanceError("Reviewed job must be a regular directory, not a symbolic link")
    expected = {"meta.json", "app/config/config_fed_server.json", "app/config/config_fed_client.json"}
    files = set()
    for p in root.rglob("*"):
        if p.is_symlink():
            raise AcceptanceError("Symlinks are not permitted in the reviewed job")
        if p.is_file():
            files.add(p.relative_to(root).as_posix())
    if files != expected:
        raise AcceptanceError("Validation job must contain only the reviewed metadata and two JSON configs")
    meta = json.loads((root / "meta.json").read_text())
    if (
        sorted(meta.get("mandatory_clients", [])) != sorted(CLIENTS)
        or meta.get("min_clients") != 2
        or meta.get("deploy_map") != {"app": ["server", "site-1", "site-2"]}
    ):
        raise AcceptanceError("Validation job must require exactly both protected clients")
    server = json.loads((root / "app/config/config_fed_server.json").read_text())
    client = json.loads((root / "app/config/config_fed_client.json").read_text())
    workflows = server.get("workflows", [])
    executors = client.get("executors", [])
    if (
        set(server) != {"format_version", "workflows", "components", "task_data_filters", "task_result_filters"}
        or set(client) != {"format_version", "executors", "components", "task_data_filters", "task_result_filters"}
        or any(config.get("format_version") != 2 for config in (server, client))
        or any(
            config.get(key) != []
            for config in (server, client)
            for key in ("components", "task_data_filters", "task_result_filters")
        )
        or workflows != [{"id": "acceptance", "path": "tdx_acceptance.AcceptanceController", "args": {"nonce": nonce}}]
        or executors
        != [
            {
                "tasks": ["tdx_acceptance"],
                "executor": {"path": "tdx_acceptance.AcceptanceExecutor", "args": {"nonce": nonce}},
            }
        ]
    ):
        raise AcceptanceError("Validation job does not select the baked acceptance components and current nonce")
    return hashlib.sha256(b"".join((root / p).read_bytes() for p in sorted(expected))).hexdigest()


def verify_unchanged_job(job, nonce, expected_hash):
    """Recheck the reviewed source before submission can add API metadata."""
    if verify_job_definition(job, nonce) != expected_hash:
        raise AcceptanceError("Reviewed job changed before submission")


def verify_result(download, nonce, job_id):
    root = Path(download)
    candidates = list(root.rglob("tdx_acceptance_result.json"))
    if len(candidates) != 1 or candidates[0].is_symlink():
        raise AcceptanceError("Expected exactly one downloaded acceptance result")
    result = json.loads(candidates[0].read_text())
    values = result.get("values", {})
    if (
        result.get("schema_version") != 1
        or result.get("status") != "passed"
        or result.get("job_id") != job_id
        or result.get("nonce") != nonce
        or values != VALUES
        or any(type(v) is not int for v in values.values())
        or type(result.get("aggregate")) is not int
        or result.get("aggregate") != 10
        or result.get("errors") != []
    ):
        raise AcceptanceError("Downloaded result fails nonce, identity, value, or aggregate verification")
    return hashlib.sha256(candidates[0].read_bytes()).hexdigest()


def observe(session, log, duration, timeout, min_rounds=2, clock=time.monotonic, sleep=time.sleep):
    start = clock()
    initial_rounds = log.rounds
    while True:
        now = clock()
        verify_membership(session)
        log.poll(now)
        if now - (log.last_success if log.last_success is not None else start) > timeout:
            raise AcceptanceError("Fresh observer validation did not complete within the observation bound")
        if now - start >= duration and log.rounds - initial_rounds >= min_rounds:
            return
        if duration == 0 and now - start >= timeout:
            raise AcceptanceError("Two new periodic validation rounds were not observed within timeout")
        sleep(2)


def run(args, session):
    validate_options(args.timeout, args.soak)
    verify_observer_config(args.observer_local, args.topology)
    job_hash = verify_job_definition(args.job, args.nonce)
    log = FreshObserverLog(args.observer_log)
    try:
        start = time.monotonic()
        while True:
            try:
                verify_membership(session)
                break
            except AcceptanceError:
                if time.monotonic() - start >= args.timeout:
                    raise AcceptanceError("Expected federation did not register within timeout")
                log.poll(time.monotonic())
                time.sleep(2)
        observe(session, log, 0, args.timeout)
        verify_unchanged_job(args.job, args.nonce, job_hash)
        job_id = session.submit_job(args.job)
        start = time.monotonic()
        while True:
            verify_membership(session)
            log.poll(time.monotonic())
            status = session.get_job_meta(job_id).get("status")
            if status == "FINISHED:COMPLETED":
                break
            if status and status.startswith("FINISHED:"):
                raise AcceptanceError("Validation job terminated without successful completion")
            if time.monotonic() - start >= args.timeout:
                raise AcceptanceError("Validation job exceeded completion timeout")
            time.sleep(2)
        result_hash = verify_result(session.download_job_result(job_id, args.download_dir), args.nonce, job_id)
        if args.soak:
            observe(session, log, args.soak, args.timeout)
        return {
            "schema_version": 1,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "topology": args.topology,
            "functional_evidence": {
                "membership": sorted(MEMBERS),
                "new_observer_periodic_rounds": log.rounds,
                "job_id": job_id,
                "nonce": args.nonce,
                "job_sha256": job_hash,
                "result_sha256": result_hash,
                "application_execution": "passed",
                "soak_seconds": args.soak,
            },
            "qualification": "functional_partial" if not args.soak else "functional_observation_passed",
            "independent_evidence_required": [
                "TDX appraisal",
                "resource release",
                "confidentiality",
                "cold relaunch",
                "service restart",
                "mandatory denial tests",
                "exact image binding",
            ],
        }
    finally:
        log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("admin-kit", "username", "observer-local", "observer-log", "job", "nonce", "download-dir", "receipt"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--topology", required=True, choices=("A", "B"))
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--soak", type=float, default=900)
    args = parser.parse_args()
    session = None
    try:
        validate_options(args.timeout, args.soak)
        if not re.fullmatch(r"[A-Za-z0-9_-]{16,128}", args.nonce):
            raise AcceptanceError("Nonce must contain 16-128 ASCII letters, numbers, underscores, or hyphens")
        from nvflare.fuel.flare_api.flare_api import new_secure_session

        session = new_secure_session(
            args.username, args.admin_kit, debug=False, timeout=args.timeout, command_timeout=45, auto_login_max_tries=1
        )
        receipt = run(args, session)
        with open(args.receipt, "x", opener=lambda p, flags: os.open(p, flags, 0o600)) as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
        print("Trusted federation observation completed; hardware and security evidence remain separate.")
    except Exception:
        # SDK exceptions and logs may contain credentials or service responses.
        parser.exit(
            1, "Acceptance check failed; no full qualification claimed. Inspect private trusted-side diagnostics.\n"
        )
    finally:
        if session is not None:
            session.close()


if __name__ == "__main__":
    main()
