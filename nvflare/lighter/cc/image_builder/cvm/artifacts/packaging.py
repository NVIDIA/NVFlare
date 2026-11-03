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

"""Package generic CVM bundles and application deliveries as OCI artifacts."""

import os
import tempfile
from pathlib import Path

from ..common.errors import require
from ..common.io import digest_file, read_json, write_json
from ..common.linux import lock
from .oci import (
    CVM_ARTIFACT_TYPE,
    CVM_CONFIG_MEDIA_TYPE,
    CVM_LAYER_MEDIA_TYPE,
    DELIVERY_ARTIFACT_TYPE,
    DELIVERY_CONFIG_MEDIA_TYPE,
    VAULT_LAYER_MEDIA_TYPE,
    create,
)


def package_bundle(directory):
    """Package one platform bundle as a single-manifest OCI image layout tar."""
    directory = Path(directory).resolve()
    final = (directory / "cvm_manifest.json").is_file()
    manifest_name = "cvm_manifest.json" if final else "cvm_manifest.pending.json"
    manifest = read_json(directory / manifest_name)
    platform = manifest["platform"]
    root = directory.parent
    for member, expected in manifest["sha256"].items():
        require(Path(member).name == member, "Invalid CVM artifact member")
        require(digest_file(directory / member) == expected, f"CVM artifact hash mismatch: {member}")
    members = [(directory / name, platform + "/" + name) for name in manifest["sha256"]]
    members.append((directory / manifest_name, platform + "/" + manifest_name))
    for name in ("resource_policy.rego", "approval.json"):
        if (directory / name).is_file():
            members.append((directory / name, platform + "/" + name))
    temporary_profile = None
    if final:
        profiles = read_json(root / "profile_set.json")
        entry = profiles["bundles"][platform]
        fd, temporary_profile = tempfile.mkstemp(prefix=".profile-set-", dir=root)
        os.close(fd)
        write_json(
            temporary_profile,
            {
                "schema_version": profiles["schema_version"],
                "profile_version": profiles["profile_version"],
                "contract": profiles["contract"],
                "bundles": {platform: entry},
            },
            mode=0o644,
        )
        members.append((Path(temporary_profile), "profile_set.json"))
    name = f"cvm_{manifest['profile_version']}_{platform}.oci.tar"
    destination = root / name
    try:
        descriptor = create(
            destination,
            CVM_ARTIFACT_TYPE,
            CVM_CONFIG_MEDIA_TYPE,
            {
                "schema_version": 1,
                "kind": "cvm_bundle",
                "state": (
                    "approved" if (directory / "approval.json").is_file() else ("finalized" if final else "pending")
                ),
                "profile_version": manifest["profile_version"],
                "platform": platform,
                "build_id": manifest["build_id"],
                "materialized_name": f"cvm_{manifest['profile_version']}",
            },
            [{"media_type": CVM_LAYER_MEDIA_TYPE, "title": "cvm-bundle.tar.gz", "members": members}],
            {
                "org.opencontainers.image.title": name,
                "org.opencontainers.image.version": manifest["profile_version"],
            },
        )
    finally:
        if temporary_profile:
            Path(temporary_profile).unlink(missing_ok=True)
    with lock(root / ".oci.lock"):
        record_path = root / "oci_artifacts.json"
        record = read_json(record_path) if record_path.exists() else {"schema_version": 1, "artifacts": {}}
        record["artifacts"][name] = {
            "artifact_type": CVM_ARTIFACT_TYPE,
            "manifest_digest": descriptor["digest"],
            "archive_sha256": digest_file(destination),
        }
        write_json(record_path, record, mode=0o644)
    return destination


def package_deliveries(output, deployment_id, copies):
    """Package self-contained runtime deliveries as OCI artifacts."""
    artifacts = {}
    for copy in copies:
        platform = copy["platform"]
        directory = Path(output) / platform
        name = f"vault_{deployment_id}_{platform}.oci.tar"
        destination = Path(output) / name
        runtime_members = [
            (directory / member, platform + "/" + member)
            for member in (
                "vault.qcow2",
                "applog.qcow2",
                "user_config.qcow2",
                "user_data.qcow2",
                "vault_manifest.json",
                "launch_cvm.sh",
                "shutdown_cvm.sh",
                "README.txt",
                "cvm",
            )
        ]
        descriptor = create(
            destination,
            DELIVERY_ARTIFACT_TYPE,
            DELIVERY_CONFIG_MEDIA_TYPE,
            {
                "schema_version": 1,
                "kind": "cvm_delivery",
                "deployment_id": deployment_id,
                "profile_version": copy["profile_version"],
                "platform": platform,
                "cvm_build_id": copy["cvm_build_id"],
                "materialized_name": f"vault_{deployment_id}_{platform}",
                "launch_directory": platform,
            },
            [
                {
                    "media_type": CVM_LAYER_MEDIA_TYPE,
                    "title": "cvm-bundle.tar.gz",
                    "members": [(directory / "cvm_bundle", platform + "/cvm_bundle")],
                },
                {"media_type": VAULT_LAYER_MEDIA_TYPE, "title": "vault-runtime.tar.gz", "members": runtime_members},
            ],
            {
                "org.opencontainers.image.title": name,
                "org.opencontainers.image.version": deployment_id,
            },
        )
        artifacts[name] = {
            "artifact_type": DELIVERY_ARTIFACT_TYPE,
            "manifest_digest": descriptor["digest"],
            "archive_sha256": digest_file(destination),
        }
    write_json(Path(output) / "oci_artifacts.json", {"schema_version": 1, "artifacts": artifacts}, mode=0o644)
