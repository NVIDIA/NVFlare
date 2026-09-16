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

"""OCI delivery integrity and Docker save identity tests."""

import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from builder.builder import package_bundle
from builder.common import BuildError, canonical, digest_file, read_json, write_json
from builder.launcher import find_bundle
from builder.oci import (
    CVM_ARTIFACT_TYPE,
    CVM_CONFIG_MEDIA_TYPE,
    CVM_LAYER_MEDIA_TYPE,
    DELIVERY_ARTIFACT_TYPE,
    create,
    materialize,
    publish,
)
from builder.vault import package_deliveries, validate_archive


class ArchiveTests(unittest.TestCase):
    def test_delivery_oci_artifact_is_self_contained_and_excludes_build_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            delivery = root / "intel_tdx"
            delivery.mkdir()
            for name in (
                "vault.qcow2",
                "applog.qcow2",
                "user_config.qcow2",
                "user_data.qcow2",
                "launch_cvm.sh",
                "shutdown_cvm.sh",
                "README.txt",
                "provisioning.json",
            ):
                (delivery / name).write_text("fixture")
            metadata = {
                "profile_version": "cpu-2026.09",
                "platform": "intel_tdx",
                "cvm_build_id": "cvm-example",
            }
            write_json(delivery / "vault_manifest.json", metadata)
            bundle = delivery / "cvm_bundle"
            bundle.mkdir()
            for name in ("verity_root.qcow2", "resource_policy.rego", "approval.json"):
                (bundle / name).write_text("fixture")
            write_json(
                bundle / "cvm_manifest.json",
                {"build_id": "cvm-example", "sha256": {"verity_root.qcow2": "fixture"}},
            )
            (delivery / "builder/__pycache__").mkdir(parents=True)
            (delivery / "builder/launcher.py").write_text("# launcher")
            (delivery / "builder/__pycache__/launcher.pyc").write_bytes(b"cache")
            package_deliveries(root, "test", [metadata])
            artifact = root / "vault_test_intel_tdx.oci.tar"
            with tarfile.open(artifact, "r:") as archive:
                names = archive.getnames()
            self.assertIn("oci-layout", names)
            self.assertIn("index.json", names)
            extracted = root / "extracted"
            _, descriptor, config = materialize(artifact, extracted)
            self.assertEqual(descriptor["artifactType"], DELIVERY_ARTIFACT_TYPE)
            self.assertEqual(config["launch_directory"], "intel_tdx")
            extracted_delivery = extracted / "intel_tdx"
            for name in (
                "README.txt",
                "vault.qcow2",
                "shutdown_cvm.sh",
                "cvm_bundle/cvm_manifest.json",
                "cvm_bundle/verity_root.qcow2",
                "cvm_bundle/resource_policy.rego",
                "cvm_bundle/approval.json",
            ):
                self.assertTrue((extracted_delivery / name).is_file(), name)
            self.assertFalse((extracted_delivery / "provisioning.json").exists())
            self.assertFalse(any("__pycache__" in path.parts for path in extracted.rglob("*")))
            self.assertEqual(find_bundle(extracted_delivery, metadata), (extracted_delivery / "cvm_bundle").resolve())
            record = read_json(root / "oci_artifacts.json")["artifacts"][artifact.name]
            self.assertEqual(record["archive_sha256"], digest_file(artifact))

    def test_cvm_bundle_is_a_materializable_oci_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cvm_cpu-2026.09"
            bundle = root / "intel_tdx"
            bundle.mkdir(parents=True)
            (bundle / "verity_root.qcow2").write_text("root")
            manifest = {
                "schema_version": 2,
                "profile_version": "cpu-2026.09",
                "platform": "intel_tdx",
                "build_id": "cvm-example",
                "sha256": {"verity_root.qcow2": digest_file(bundle / "verity_root.qcow2")},
            }
            write_json(bundle / "cvm_manifest.json", manifest)
            write_json(
                root / "profile_set.json",
                {
                    "schema_version": 2,
                    "profile_version": "cpu-2026.09",
                    "contract": {"fixture": True},
                    "bundles": {
                        "intel_tdx": {
                            "build_id": "cvm-example",
                            "manifest_sha256": digest_file(bundle / "cvm_manifest.json"),
                        }
                    },
                },
            )
            artifact = package_bundle(bundle)
            first_digest = digest_file(artifact)
            self.assertEqual(digest_file(package_bundle(bundle)), first_digest)
            extracted = Path(directory) / "extracted"
            _, descriptor, config = materialize(artifact, extracted)
            self.assertEqual(descriptor["artifactType"], CVM_ARTIFACT_TYPE)
            self.assertEqual(config["state"], "finalized")
            self.assertTrue((extracted / "profile_set.json").is_file())
            self.assertTrue((extracted / "intel_tdx/cvm_manifest.json").is_file())

            second_root = Path(directory) / "second" / "cvm_cpu-2026.09"
            second_bundle = second_root / "amd_sev_snp"
            second_bundle.mkdir(parents=True)
            (second_bundle / "verity_root.qcow2").write_text("snp-root")
            second_manifest = {
                "schema_version": 2,
                "profile_version": "cpu-2026.09",
                "platform": "amd_sev_snp",
                "build_id": "cvm-snp-example",
                "sha256": {"verity_root.qcow2": digest_file(second_bundle / "verity_root.qcow2")},
            }
            write_json(second_bundle / "cvm_manifest.json", second_manifest)
            write_json(
                second_root / "profile_set.json",
                {
                    "schema_version": 2,
                    "profile_version": "cpu-2026.09",
                    "contract": {"fixture": True},
                    "bundles": {
                        "amd_sev_snp": {
                            "build_id": "cvm-snp-example",
                            "manifest_sha256": digest_file(second_bundle / "cvm_manifest.json"),
                        }
                    },
                },
            )
            materialize(package_bundle(second_bundle), extracted, merge=True)
            self.assertTrue((extracted / "amd_sev_snp/cvm_manifest.json").is_file())
            self.assertEqual(set(read_json(extracted / "profile_set.json")["bundles"]), {"amd_sev_snp", "intel_tdx"})

            # Matching contracts cannot make differently versioned profiles mergeable.
            current = read_json(extracted / "profile_set.json")
            current["profile_version"] = "cpu-2026.10"
            write_json(extracted / "profile_set.json", current)
            with self.assertRaisesRegex(BuildError, "different profiles"):
                materialize(artifact, extracted, merge=True)

    def test_oci_blob_tampering_and_mutable_registry_tags_are_denied(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "payload"
            payload.mkdir()
            (payload / "data").write_text("authenticated")
            artifact = root / "cvm.oci.tar"
            create(
                artifact,
                CVM_ARTIFACT_TYPE,
                CVM_CONFIG_MEDIA_TYPE,
                {"kind": "cvm_bundle", "materialized_name": "cvm-test"},
                [
                    {
                        "media_type": CVM_LAYER_MEDIA_TYPE,
                        "title": "cvm-bundle.tar.gz",
                        "members": [(payload, "intel_tdx")],
                    }
                ],
            )
            with patch("builder.oci.run") as execute:
                immutable = publish(artifact, "10.0.0.1:5000/cvm/test:latest", plain_http=True)
            self.assertRegex(immutable, r"^10\.0\.0\.1:5000/cvm/test@sha256:[0-9a-f]{64}$")
            self.assertIn("--to-plain-http", execute.call_args.args[0])
            tampered = root / "tampered.oci.tar"
            changed = False
            with tarfile.open(artifact, "r:") as source, tarfile.open(tampered, "w") as output:
                for member in source:
                    content = source.extractfile(member) if member.isfile() else None
                    data = content.read() if content else None
                    if not changed and member.isfile() and member.name.startswith("blobs/sha256/"):
                        data += b"tamper"
                        member.size = len(data)
                        changed = True
                    output.addfile(member, io.BytesIO(data) if data is not None else None)
            with self.assertRaises(BuildError):
                materialize(tampered, root / "rejected")
            with self.assertRaises(BuildError):
                materialize("registry.example.org/cvm/example:latest", root / "mutable")

    def archive(self, directory, *, tamper=False, ambiguous=False):
        config = canonical({"architecture": "amd64", "os": "linux", "config": {"Cmd": ["serve"]}})
        config_id = "sha256:" + hashlib.sha256(config).hexdigest()
        manifest = canonical({"config": {"digest": config_id}})
        manifest_id = "sha256:" + hashlib.sha256(manifest).hexdigest()
        descriptor = {"digest": manifest_id, "platform": {"architecture": "amd64", "os": "linux"}}
        index = canonical({"manifests": [descriptor] * (2 if ambiguous else 1)})
        index_id = "sha256:" + hashlib.sha256(index).hexdigest()
        members = {
            "manifest.json": canonical([{"Config": "blobs/sha256/" + config_id[7:]}]),
            "blobs/sha256/" + config_id[7:]: config,
            "blobs/sha256/" + manifest_id[7:]: manifest + (b"x" if tamper else b""),
            "blobs/sha256/" + index_id[7:]: index,
        }
        path = Path(directory) / "save.tar"
        with tarfile.open(path, "w") as archive:
            for name, data in members.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        return path, config_id, index_id

    def test_classic_and_oci_ids_resolve_to_same_guest_image(self):
        with tempfile.TemporaryDirectory() as directory:
            path, config_id, index_id = self.archive(directory)
            self.assertEqual(validate_archive(path, config_id), config_id)
            self.assertEqual(validate_archive(path, index_id), config_id)

    def test_tampered_or_ambiguous_manifest_denied(self):
        for options in ({"tamper": True}, {"ambiguous": True}):
            with tempfile.TemporaryDirectory() as directory:
                path, _, index_id = self.archive(directory, **options)
                with self.assertRaises(BuildError):
                    validate_archive(path, index_id)
