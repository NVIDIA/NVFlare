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
from contextlib import contextmanager
from pathlib import Path

from ..artifacts.bundle import load_public_keys, verify_approval, verify_bundle
from ..common.contracts import identifier
from ..common.errors import BuildError, require
from ..common.io import canonical, digest_file, is_sha256, read_json, write_json
from ..common.linux import lock
from ..common.policy import merge
from ..common.references import TCB_NAMES
from ..common.versions import (
    COCO_ACTIX_HTTP_CRATE_SHA256,
    COCO_ACTIX_HTTP_SOURCE_SHA256,
    COCO_ACTIX_HTTP_VERSION,
    COCO_SERVICE_BUILD_PROFILE,
    COCO_SERVICE_PATCH_SHA256,
)
from .client import api, encode
from .references import check_profile, profile_identity


def read_resource_policy(config):
    """v0.22 lists policy IDs over HTTP; read bytes from its local_fs storage.

    Administration runs beside Trustee with read access to the same storage.
    The configured storage directory must be the one mounted into Trustee.
    """
    policies = json.loads(api(config, "GET", "resource-policy"))
    require(isinstance(policies, list) and "resource-policy" in policies, "KBS resource policy is absent")
    path = config.get("resource_policy_file")
    if path is None:
        path = Path(config["storage_directory"]) / "kbs/resource-policy.rego"
    return Path(path).read_bytes()


def read_resource_policy_text(config):
    try:
        return read_resource_policy(config).decode("utf-8")
    except UnicodeDecodeError:
        raise BuildError("KBS resource policy is not UTF-8") from None


def verify_readback(expected, returned):
    require(returned == expected, "KBS policy readback differs from published bytes")


def check_migration(config):
    # Policy publication must not silently discard the earlier backend's denials.
    require(
        "key_service_state" not in config,
        "Migrate legacy revocations and bundle retirements before removing key_service_state; see TRUSTEE_GUIDE.md",
    )


def verify_server_provenance(config, build, deployment):
    """Accept clean upstream builds or the exact shared CoCo service recipe."""
    require(
        digest_file(config["trustee_binary"]) == build.get("binary_sha256"),
        "Installed Trustee binary differs from build provenance",
    )
    clean = build.get("source_clean") is True
    coco = build.get("source_clean") is False and build.get("build_profile") == COCO_SERVICE_BUILD_PROFILE
    require(clean or coco, "Trustee server provenance is not an approved build")
    if clean:
        require(deployment.get("source_clean") is True, "Trustee deployment must use unmodified upstream source")
        return
    expected = {
        "build_profile": COCO_SERVICE_BUILD_PROFILE,
        "source_patch_sha256": COCO_SERVICE_PATCH_SHA256,
        "actix_http_version": COCO_ACTIX_HTTP_VERSION,
        "actix_http_crate_sha256": COCO_ACTIX_HTTP_CRATE_SHA256,
        "actix_http_source_sha256": COCO_ACTIX_HTTP_SOURCE_SHA256,
    }
    require(all(build.get(name) == value for name, value in expected.items()), "CoCo Trustee build recipe mismatch")
    require(
        deployment.get("source_clean") is False
        and all(deployment.get(name) == value for name, value in expected.items()),
        "Trustee deployment does not use the approved CoCo service build",
    )
    require(
        isinstance(build.get("kbs_image_id"), str)
        and build["kbs_image_id"].startswith("sha256:")
        and is_sha256(build["kbs_image_id"].removeprefix("sha256:"))
        and config.get("trustee_image_id") == build["kbs_image_id"]
        and deployment.get("kbs_image_id") == build["kbs_image_id"],
        "Running Trustee image differs from CoCo build provenance",
    )


@contextmanager
def publication_locks(config, state):
    """Serialize CVM state first, then the resource policy shared with CoCo."""
    state_lock = (state / "publisher.lock").resolve()
    value = config.get("policy_lock")
    require(
        isinstance(value, str) and value and Path(value).expanduser().is_absolute(),
        "Trustee administration requires an absolute shared policy_lock",
    )
    policy_lock = Path(value).expanduser().resolve()
    require(policy_lock != state_lock, "policy_lock must be the shared CoCo lock, separate from publisher state")
    with lock(state_lock):
        with lock(policy_lock):
            yield


def install(config, directory, candidate=False):
    check_migration(config)
    if candidate:
        manifest = verify_bundle(directory)
    else:
        # A receipt file alone is not approval: it must carry a signature by an
        # acceptance authority this deployment trusts.
        require(
            "approval_public_keys" in config,
            "Production installation requires approval_public_keys in the administration configuration",
        )
        manifest = verify_approval(directory, load_public_keys(config["approval_public_keys"]))
    state = Path(config["state"])
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Deployment receipts are per policy/revision, not per vault. The operator
    # must have tested selection/admin denial and installed immutable AS files.
    deployment = read_json(config["deployment_receipt"])
    build = read_json(config["trustee_build"])
    require(build["trustee_commit"] == manifest["contract"]["trustee_commit"], "Trustee build provenance mismatch")
    require(
        deployment["trustee_commit"] == manifest["contract"]["trustee_commit"], "Trustee deployment revision mismatch"
    )
    verify_server_provenance(config, build, deployment)
    require(
        deployment["policy_selection_tested"] is True and deployment["unauthorized_administration_denied"] is True,
        "Trustee administrative acceptance is incomplete",
    )
    pid = manifest["attestation_policy_id"]
    selector = manifest["contract"]["attestation_policy_selector"]
    require(
        deployment.get("policy_id_map", {}).get(selector) == [pid],
        "Trustee policy selector is not mapped to this profile's policy",
    )
    policies = {pid + "_cpu": "attestation_policy.rego"}
    if manifest["contract"].get("gpu") == "nvidia_cc":
        policies[pid + "_gpu"] = "gpu_attestation_policy.rego"
    for policy_name, artifact in policies.items():
        require(
            deployment["immutable_as_policies"].get(policy_name) == manifest["sha256"][artifact],
            "AS policy is not installed immutably: " + policy_name,
        )
    expected_refs = read_json(Path(directory) / "reference_values.json")
    reference_id = manifest["reference_value_id"]
    with publication_locks(config, state):
        # Reference import takes the same state lock. Read and validate its
        # profile-scoped record while holding that lock so publication cannot
        # race a concurrent import or expiry change.
        record = json.loads(api(config, "GET", "reference-value/" + reference_id))
        require(isinstance(record, dict), "Invalid profile-scoped RVPS record")
        refs = record.get("values", {})
        expirations = record.get("expirations", {})
        require(
            isinstance(refs, dict) and set(expected_refs) <= set(refs),
            "RVPS profile record lacks approved references",
        )
        require(
            isinstance(expirations, dict)
            and all(
                type(expirations.get(name)) in (int, float) and expirations[name] > time.time()
                for name in expected_refs
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
        profile_path = state / "profiles" / (reference_id + ".json")
        if profile_path.exists():
            check_profile(read_json(profile_path), manifest)
        for category, name in (("selectors", selector), ("policy_ids", pid)):
            identity_path = state / category / (name + ".json")
            if identity_path.exists():
                check_profile(read_json(identity_path), manifest)
        require(not (state / "retired" / manifest["build_id"]).exists(), "Retired bundle cannot be re-enabled")
        bundle_dir = state / "bundles"
        bundle_dir.mkdir(exist_ok=True)
        manifests = {path.stem: read_json(path) for path in bundle_dir.glob("*.json")}
        if manifest["build_id"] in manifests:
            require(manifests[manifest["build_id"]] == manifest, "Bundle ID already names a different manifest")
        previous = [item for key, item in manifests.items() if not (state / "retired" / key).exists()]
        manifests[manifest["build_id"]] = manifest
        active = [item for key, item in manifests.items() if not (state / "retired" / key).exists()]
        # Pin each selector, policy ID and RVPS record to one immutable profile;
        # unrelated profiles and CoCo's default policy remain independent.
        write_json(profile_path, profile_identity(manifest))
        write_json(state / "selectors" / (selector + ".json"), profile_identity(manifest))
        write_json(state / "policy_ids" / (pid + ".json"), profile_identity(manifest))
        policy = merge(read_resource_policy_text(config), active, previous).encode()
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
    with publication_locks(config, state):
        (state / "retired").mkdir(exist_ok=True)
        write_json(state / "retired" / build_id, {"build_id": build_id})
        previous = [
            read_json(path)
            for path in (state / "bundles").glob("*.json")
            if path.stem == build_id or not (state / "retired" / path.stem).exists()
        ]
        active = [
            read_json(path)
            for path in (state / "bundles").glob("*.json")
            if not (state / "retired" / path.stem).exists()
        ]
        policy = merge(read_resource_policy_text(config), active, previous).encode()
        api(config, "POST", "resource-policy", canonical({"policy": encode(policy)}))
        verify_readback(policy, read_resource_policy(config))
