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

"""Serialized administration of reusable CVM bundle policies, never vault records.

AS policies and RVPS reference endorsements are installed by the KBS deployment
operator. This command verifies that installation, publishes the combined bundle
resource policy, reads its actual bytes back, and enables resource creation.
"""

import argparse
import base64
import hashlib
import json
import ssl
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from .common import BuildError, canonical, identifier, lock, read_json, require, write_json
from .key_service import NoRedirect, ResourceStore
from .policy import approve_bundle, compose, verify_approval, verify_bundle


def encode(data):
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def api(config, method, endpoint, data=None):
    require(config["url"].startswith("https://"), "KBS administration requires HTTPS")
    key = serialization.load_pem_private_key(Path(config["admin_private_key"]).read_bytes(), password=None)
    require(isinstance(key, ed25519.Ed25519PrivateKey), "KBS administration requires an Ed25519 key")
    now = int(time.time())
    body = (
        encode(canonical({"alg": "EdDSA", "typ": "JWT"}))
        + "."
        + encode(canonical({"iat": now, "nbf": now - 5, "exp": now + 60}))
    )
    token = body + "." + encode(key.sign(body.encode()))
    context = ssl.create_default_context(cafile=config["ca"])
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context), NoRedirect())
    request = urllib.request.Request(
        config["url"].rstrip("/") + "/kbs/v0/" + endpoint,
        data=data,
        method=method,
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
    )
    try:
        with opener.open(request, timeout=30) as response:
            result = response.read(8 * 1024**2 + 1)
            require(len(result) <= 8 * 1024**2, "Administrative response too large")
            return result
    except urllib.error.HTTPError as exc:
        exc.close()
        raise BuildError("KBS administrative request rejected; no credentials logged") from None
    except (urllib.error.URLError, OSError):
        raise BuildError("KBS administrative request failed; no credentials logged") from None


def verify_readback(expected, returned):
    try:
        text = returned.decode("ascii")
        require(
            text and "=" not in text and encode(base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))) == text,
            "Noncanonical policy readback",
        )
        actual = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
        require(actual == expected, "KBS policy readback differs from published bytes")
    except (UnicodeError, ValueError):
        raise BuildError("KBS did not return URL-safe unpadded policy bytes") from None


def install(config, directory, candidate=False):
    manifest = verify_bundle(directory) if candidate else verify_approval(directory)
    require(
        not candidate or manifest["profile_version"].startswith("test-"),
        "Candidate administration requires a test- profile",
    )
    state = Path(config["state"])
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    store = ResourceStore(config["resources"], config["key_service_state"])
    # Deployment receipts are per policy/revision, not per vault. The operator
    # must have tested selection/admin denial and installed immutable AS files.
    deployment = read_json(config["deployment_receipt"])
    require(
        deployment["trustee_commit"] == manifest["contract"]["trustee_commit"], "Trustee deployment revision mismatch"
    )
    require(
        deployment["trustee_patch_digest"] == manifest["contract"]["trustee_patch_digest"],
        "Untracked Trustee patch set",
    )
    require(
        deployment["policy_selection_tested"] is True and deployment["unauthorized_administration_denied"] is True,
        "Trustee administrative acceptance is incomplete",
    )
    pid = manifest["attestation_policy_id"]
    require(
        deployment["immutable_as_policies"][pid] == manifest["sha256"]["attestation_policy.rego"],
        "AS policy is not installed immutably",
    )
    refs = json.loads(api(config, "GET", "reference-value"))
    expected_refs = read_json(Path(directory) / "reference_values.json")
    require(
        all(
            key in refs
            and (
                set(value) <= set(refs[key])
                if isinstance(value, list) and isinstance(refs[key], list)
                else refs[key] == value
            )
            for key, value in expected_refs.items()
        ),
        "RVPS lacks the approved bundle/TCB reference values",
    )
    with lock(state / "publisher.lock"):
        require(not (store.state / "retired" / manifest["build_id"]).exists(), "Retired bundle cannot be re-enabled")
        bundle_dir = state / "bundles"
        bundle_dir.mkdir(exist_ok=True)
        manifests = {path.stem: read_json(path) for path in bundle_dir.glob("*.json")}
        if manifest["build_id"] in manifests:
            require(manifests[manifest["build_id"]] == manifest, "Bundle ID already names a different manifest")
        manifests[manifest["build_id"]] = manifest
        active = [item for key, item in manifests.items() if not (store.state / "retired" / key).exists()]
        policy = compose(active).encode()
        api(config, "POST", "resource-policy", canonical({"policy": encode(policy)}))
        verify_readback(policy, api(config, "GET", "resource-policy"))
        write_json(bundle_dir / (manifest["build_id"] + ".json"), manifest)
        with lock(store.state / "administration.lock"):
            require(not (store.state / "retired" / manifest["build_id"]).exists(), "Bundle retired during publication")
            write_json(store.state / "approved-bundles.json", {"build_ids": [m["build_id"] for m in active]})
        write_json(
            state / "publication.json",
            {
                "resource_policy_sha256": hashlib.sha256(policy).hexdigest(),
                "bundle_ids": sorted(m["build_id"] for m in active),
            },
        )


def retire(config, build_id):
    identifier(build_id)
    state = Path(config["state"])
    store = ResourceStore(config["resources"], config["key_service_state"])
    with lock(state / "publisher.lock"):
        store.retire(build_id)
        active = [
            read_json(path)
            for path in (state / "bundles").glob("*.json")
            if not (store.state / "retired" / path.stem).exists()
        ]
        policy = compose(active).encode()
        api(config, "POST", "resource-policy", canonical({"policy": encode(policy)}))
        verify_readback(policy, api(config, "GET", "resource-policy"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    approval = sub.add_parser("approve")
    approval.add_argument("bundle")
    approval.add_argument("evidence")
    add = sub.add_parser("install")
    add.add_argument("config")
    add.add_argument("bundle")
    add.add_argument("--candidate", action="store_true")
    remove = sub.add_parser("retire")
    remove.add_argument("config")
    remove.add_argument("build_id")
    args = parser.parse_args()
    try:
        if args.action == "approve":
            approve_bundle(args.bundle, read_json(args.evidence))
            from .builder import package_bundle

            package_bundle(args.bundle)
        elif args.action == "install":
            install(read_json(args.config), args.bundle, args.candidate)
        else:
            retire(read_json(args.config), args.build_id)
    except (BuildError, OSError, ValueError, KeyError, tarfile.TarError) as exc:
        parser.exit(1, f"Bundle administration failed: {exc}\n")


if __name__ == "__main__":
    main()
