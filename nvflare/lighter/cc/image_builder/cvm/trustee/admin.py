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

"""Register, approve and retire bundles; revoke resources through CoCo Trustee."""

import hashlib
import json
import time
from pathlib import Path

from ..artifacts.bundle import verify_approval, verify_bundle
from ..common.contracts import identifier
from ..common.errors import require
from ..common.io import canonical, digest_file, read_json, write_json
from ..common.linux import lock
from ..common.policy import compose
from ..common.references import EXPIRY_REFERENCE, TCB_NAMES
from .client import api, encode
from .references import check_profile, profile_identity


def read_resource_policy(config):
    """v0.22 lists policy IDs over HTTP; read bytes from its local_fs storage.

    Administration runs beside Trustee with read access to the same storage.
    The configured storage directory must be the one mounted into Trustee.
    """
    policies = json.loads(api(config, "GET", "resource-policy"))
    require(isinstance(policies, list) and "resource-policy" in policies, "KBS resource policy is absent")
    return (Path(config["storage_directory"]) / "kbs/resource-policy.rego").read_bytes()


def verify_readback(expected, returned):
    require(returned == expected, "KBS policy readback differs from published bytes")


def check_migration(config):
    # Policy publication must not silently discard the earlier backend's denials.
    require(
        "key_service_state" not in config,
        "Migrate legacy revocations and bundle retirements before removing key_service_state; see TRUSTEE_GUIDE.md",
    )


def install(config, directory, candidate=False):
    check_migration(config)
    manifest = verify_bundle(directory) if candidate else verify_approval(directory)
    require(
        not candidate or manifest["profile_version"].startswith("test-"),
        "Candidate administration requires a test- profile",
    )
    state = Path(config["state"])
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Deployment receipts are per policy/revision, not per vault. The operator
    # must have tested selection/admin denial and installed immutable AS files.
    deployment = read_json(config["deployment_receipt"])
    build = read_json(config["trustee_build"])
    require(
        digest_file(config["trustee_binary"]) == build["binary_sha256"],
        "Installed Trustee binary differs from build provenance",
    )
    require(build.get("source_clean") is True, "Trustee must use unmodified upstream source")
    require(build["trustee_commit"] == manifest["contract"]["trustee_commit"], "Trustee build provenance mismatch")
    require(
        deployment["trustee_commit"] == manifest["contract"]["trustee_commit"], "Trustee deployment revision mismatch"
    )
    require(deployment.get("source_clean") is True, "Trustee deployment must use unmodified upstream source")
    require(
        deployment["policy_selection_tested"] is True and deployment["unauthorized_administration_denied"] is True,
        "Trustee administrative acceptance is incomplete",
    )
    pid = manifest["attestation_policy_id"]
    policies = {pid + "_cpu": "attestation_policy.rego"}
    if manifest["contract"].get("gpu") == "nvidia_cc":
        policies[pid + "_gpu"] = "gpu_attestation_policy.rego"
    for policy_name, artifact in policies.items():
        require(
            deployment["immutable_as_policies"].get(policy_name) == manifest["sha256"][artifact],
            "AS policy is not installed immutably: " + policy_name,
        )
    expected_refs = read_json(Path(directory) / "reference_values.json")
    refs = {name: json.loads(api(config, "GET", "reference-value/" + name)) for name in expected_refs}

    expirations = json.loads(api(config, "GET", "reference-value/" + EXPIRY_REFERENCE))
    require(
        isinstance(expirations, dict)
        and all(
            type(expirations.get(name)) in (int, float) and expirations[name] > time.time() for name in expected_refs
        ),
        "RVPS approvals are expired or lack an expiry",
    )

    for name, expected in expected_refs.items():
        actual = refs[name]
        if name in TCB_NAMES:
            require(actual == expected, "RVPS TCB approval differs from this profile: " + name)
        else:
            require(
                (
                    set(expected) <= set(actual)
                    if isinstance(expected, list) and isinstance(actual, list)
                    else actual == expected
                ),
                "RVPS lacks the approved bundle reference values: " + name,
            )
    with lock(state / "publisher.lock"):
        profile_path = state / "security_profile.json"
        if profile_path.exists():
            check_profile(read_json(profile_path), manifest)
        require(not (state / "retired" / manifest["build_id"]).exists(), "Retired bundle cannot be re-enabled")
        bundle_dir = state / "bundles"
        bundle_dir.mkdir(exist_ok=True)
        manifests = {path.stem: read_json(path) for path in bundle_dir.glob("*.json")}
        if manifest["build_id"] in manifests:
            require(manifests[manifest["build_id"]] == manifest, "Bundle ID already names a different manifest")
        manifests[manifest["build_id"]] = manifest
        active = [item for key, item in manifests.items() if not (state / "retired" / key).exists()]
        for item in manifests.values():
            check_profile(profile_identity(item), manifest)
        # Pin before publication; retirement never makes an instance reusable
        # for a different profile whose TCB references could broaden approvals.
        write_json(profile_path, profile_identity(manifest))
        policy = compose(active).encode()
        api(config, "POST", "resource-policy", canonical({"policy": encode(policy)}))
        verify_readback(policy, read_resource_policy(config))
        write_json(bundle_dir / (manifest["build_id"] + ".json"), manifest)
        write_json(
            state / "publication.json",
            {
                "resource_policy_sha256": hashlib.sha256(policy).hexdigest(),
                "bundle_ids": sorted(m["build_id"] for m in active),
            },
        )


def retire(config, build_id):
    check_migration(config)
    identifier(build_id)
    state = Path(config["state"])
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    with lock(state / "publisher.lock"):
        (state / "retired").mkdir(exist_ok=True)
        write_json(state / "retired" / build_id, {"build_id": build_id})
        active = [
            read_json(path)
            for path in (state / "bundles").glob("*.json")
            if not (state / "retired" / path.stem).exists()
        ]
        policy = compose(active).encode()
        api(config, "POST", "resource-policy", canonical({"policy": encode(policy)}))
        verify_readback(policy, read_resource_policy(config))
