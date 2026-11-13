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

"""Register resources and merge a release into an operator-maintained policy export."""

import fcntl
import hashlib
import json
import os
import re
import runpy
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SECURITY = runpy.run_path(str(ROOT / "workload-security-context.py"))
CONTRACT = runpy.run_path(str(ROOT / "workload-release.py"))
TEMPLATES = ROOT.parent / "policies"
MAX_POLICY_BYTES = 8 * 1024 * 1024


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def read_file(path, limit=MAX_POLICY_BYTES):
    with Path(path).open("rb") as stream:
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise ValueError("Local registration file exceeds its size limit")
    return content


def atomic_write(path, content):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class TrusteeClient:
    def __init__(self, executable, url, certificate, credential):
        self.command = [
            executable,
            "--url",
            url,
            "--cert-file",
            certificate,
            "config",
            "--admin-token-file",
            credential,
        ]

    def request(self, operation, *arguments):
        try:
            result = subprocess.run(
                [*self.command, operation, *arguments],
                capture_output=True,
                timeout=90,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            raise ValueError(
                "Trustee administration request failed; no credential or response body was logged"
            ) from None
        if result.returncode:
            raise ValueError("Trustee administration request was rejected or interrupted; check service access")

    def set_resource(self, resource, filename):
        self.request("set-resource", "--path", resource, "--resource-file", str(filename))

    def set_policy(self, filename):
        self.request("set-resource-policy", "--policy-file", str(filename))


def release_inputs(snapshot):
    received = json.loads(
        read_file(snapshot / "release-authorization.json"), object_pairs_hook=CONTRACT["unique_object"]
    )
    authorization = CONTRACT["validate_authorization"](received)
    image = authorization["encrypted_image"]
    repository = image.split("@", 1)[0]
    release = authorization["release_name"]
    resources = {kind: f"default/{kind}/{release}" for kind in ("image-key", "sig-public-key", "security-policy")}
    if set(authorization["kbs_resource_paths"]) != set(resources.values()):
        raise ValueError("Release authorization must name exactly its three release resource paths")
    image_policy = json.loads(
        read_file(snapshot / "image-security-policy.json"), object_pairs_hook=CONTRACT["unique_object"]
    )
    expected = {
        "default": [{"type": "reject"}],
        "transports": {
            "docker": {
                repository: [
                    {
                        "type": "sigstoreSigned",
                        "keyPath": "kbs:///" + resources["sig-public-key"],
                        "signedIdentity": {"type": "matchRepository"},
                    }
                ]
            }
        },
    }
    if image_policy != expected:
        raise ValueError("Image policy must contain the exact default-deny release signature rule")
    if not read_file(snapshot / "cosign.pub").startswith(b"-----BEGIN PUBLIC KEY-----\n"):
        raise ValueError("Signing public key must be PEM encoded")
    fragment = CONTRACT["render_fragment"](
        authorization, (TEMPLATES / "workload-resource-policy.rego.template").read_text()
    )
    expected_received = fragment
    if received["schema"] == CONTRACT["V1"]:
        expected_received = CONTRACT["render_fragment"](
            received,
            (TEMPLATES / "workload-resource-policy-v1.rego.template").read_text(),
            legacy=True,
        )
    if read_file(snapshot / "resource-policy-fragment.rego").decode().strip() != expected_received.strip():
        raise ValueError("Received release fragment differs from the service-owned template")
    return authorization, resources, fragment


def merge_policy(current, fragment, release):
    text = current.decode("utf-8")
    package = re.findall(r"^\s*package\s+policy\s*(?:#.*)?$", text, re.MULTILINE)
    if len(package) != 1:
        raise ValueError("Current resource policy must contain exactly one package policy declaration")
    if not re.search(r"^\s*default\s+allow\s*(?::=|=)\s*false\s*(?:#.*)?$", text, re.MULTILINE):
        raise ValueError("Current resource policy must retain default-deny authorization")
    prefix = "wo_" + re.sub(r"[^a-z0-9_]", "_", release)
    if prefix + "_expected_initdata" in text:
        raise ValueError("Release already exists in the current policy export; use a new release name")
    if re.search(r"^\s*(package|import|default)\b", fragment, re.MULTILINE):
        raise ValueError("Release fragment contains a forbidden top-level declaration")
    if not re.search(r"^\s*import\s+rego\.v1\s*(?:#.*)?$", text, re.MULTILINE):
        raise ValueError(
            "Current resource policy must explicitly import rego.v1; "
            "migrate and validate Rego v0 policies before registration"
        )
    return (text.rstrip() + "\n\n" + fragment.lstrip()).encode()


def recovery_message(current):
    return (
        "Policy registration is pending. Keep the current policy export and its .pending.json/.pending.rego files. "
        "Retry the same handoff to finish the same candidate, or reconcile Trustee's active policy with "
        + str(current)
        + " before removing pending state. Different releases are blocked."
    )


def install_release(source, current_file, state_dir, client, expected_manifest_sha256=None):
    try:
        current = Path(current_file).expanduser().resolve(strict=True)
    except OSError:
        raise ValueError("CURRENT_RESOURCE_POLICY_FILE must name an existing current policy export") from None
    if not current.is_file():
        raise ValueError("CURRENT_RESOURCE_POLICY_FILE must name the operator's current policy export")
    state = Path(state_dir).expanduser()
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix="release-", dir=state) as work:
        snapshot = Path(work) / "handoff"
        SECURITY["authenticate_handoff"](source, snapshot, expected_manifest_sha256=expected_manifest_sha256)
        authorization, resources, fragment = release_inputs(snapshot)
        release = authorization["release_name"]
        handoff_hash = sha256(read_file(snapshot / "SHA256SUMS"))
        lockfile = current.with_name(current.name + ".lock")
        candidate_file = current.with_name(current.name + ".pending.rego")
        pending_file = current.with_name(current.name + ".pending.json")
        descriptor = os.open(lockfile, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "r+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            previous = read_file(current)
            pending = None
            if pending_file.exists():
                try:
                    pending = json.loads(read_file(pending_file, 16384), object_pairs_hook=CONTRACT["unique_object"])
                    fields = {"release_name", "handoff_sha256", "base_policy_sha256", "candidate_sha256"}
                    if (
                        type(pending) is not dict
                        or set(pending) != fields
                        or any(not isinstance(pending[key], str) for key in fields)
                        or any(not re.fullmatch(r"[0-9a-f]{64}", pending[key]) for key in fields - {"release_name"})
                    ):
                        raise ValueError("Malformed pending state")
                    candidate = read_file(candidate_file)
                except (OSError, UnicodeError, ValueError):
                    raise ValueError(recovery_message(current)) from None
                if pending["release_name"] != release or pending["handoff_sha256"] != handoff_hash:
                    raise ValueError(recovery_message(current))
                if sha256(previous) not in (pending["base_policy_sha256"], pending["candidate_sha256"]):
                    raise ValueError(
                        "Current policy export changed during a pending update. " + recovery_message(current)
                    )
                if sha256(candidate) != pending["candidate_sha256"]:
                    raise ValueError("Pending policy candidate changed. " + recovery_message(current))
            else:
                candidate = merge_policy(previous, fragment, release)
            for kind, filename in (
                ("sig-public-key", "cosign.pub"),
                ("security-policy", "image-security-policy.json"),
                ("image-key", "image_key"),
            ):
                client.set_resource(resources[kind], snapshot / filename)
            if pending is None:
                atomic_write(candidate_file, candidate)
                pending = {
                    "release_name": release,
                    "handoff_sha256": handoff_hash,
                    "base_policy_sha256": sha256(previous),
                    "candidate_sha256": sha256(candidate),
                }
                atomic_write(pending_file, json.dumps(pending, sort_keys=True).encode())
            try:
                client.set_policy(candidate_file)
                atomic_write(current, candidate)
                receipt = {
                    "release_name": release,
                    "handoff_sha256": handoff_hash,
                    "resource_policy_sha256": sha256(candidate),
                    "status": "uploads-and-policy-accepted",
                    "accepted_at": datetime.now(timezone.utc).isoformat(),
                    "resource_paths": sorted(resources.values()),
                }
                receipts = state / "receipts"
                receipts.mkdir(exist_ok=True, mode=0o700)
                atomic_write(receipts / (release + ".json"), json.dumps(receipt, indent=2).encode())
            except (OSError, ValueError):
                raise ValueError(recovery_message(current)) from None
            try:
                pending_file.unlink()
                candidate_file.unlink()
                sync_directory(current.parent)
            except OSError:
                raise ValueError(
                    "Policy was accepted and the current export updated, but pending cleanup failed. "
                    + recovery_message(current)
                ) from None
            return receipt


def main():
    if len(sys.argv) not in (8, 9):
        raise ValueError(
            "usage: install-workload-release.py HANDOFF CURRENT_POLICY STATE KBS_CLIENT URL CA CREDENTIAL [MANIFEST_SHA256]"
        )
    source, current, state, executable, url, certificate, credential = sys.argv[1:8]
    client = TrusteeClient(executable, url, certificate, credential)
    receipt = install_release(source, current, state, client, sys.argv[8] if len(sys.argv) == 9 else None)
    print("Trustee accepted resource uploads and the merged release policy for " + receipt["release_name"] + ".")
    print("Updated the local current-policy export. Verify guest startup and NVFlare proofs after launch.")


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, TypeError, UnicodeError, ValueError) as error:
        raise SystemExit(str(error)) from None
