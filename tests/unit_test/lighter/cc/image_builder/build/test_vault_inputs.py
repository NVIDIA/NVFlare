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

"""Generic CVM retrieval, platform selection, and generated vault identities."""

import contextlib
import copy
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from cvm.artifacts.oci import CVM_ARTIFACT_TYPE, DELIVERY_ARTIFACT_TYPE
from cvm.build import config, storage, vault
from cvm.common.errors import BuildError


class VaultInputTests(unittest.TestCase):
    def test_public_sidecars_rescan_copied_bytes(self):
        source = self.root / "public"
        source.mkdir()
        (source / "input.txt").write_text("public data")
        config.public_sidecar(source)
        mounted = self.root / "mounted"
        mounted.mkdir()

        def changed_during_copy(source, destination):
            (destination / "input.txt").write_text("-----BEGIN PRIVATE KEY-----\nsecret\n")

        app = dict(applog_drive_size=1, user_config_drive_size=1, user_data_drive_size=1, user_config=source)
        with (
            patch.object(storage, "create_image"),
            patch.object(storage, "nbd", side_effect=lambda *a: contextlib.nullcontext("device")),
            patch.object(storage, "mounted", side_effect=lambda *a: contextlib.nullcontext(mounted)),
            patch.object(storage, "run"),
            patch.object(storage, "copy_tree", side_effect=changed_during_copy),
            self.assertRaisesRegex(BuildError, "Private-key"),
        ):
            vault.create_sidecars(self.root, app)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.profiles = {
            "profile_version": "cpu-2026.09",
            "bundles": {"intel_tdx": {"manifest": {"build_id": "cvm-fixture"}}},
        }
        self.metadata = {
            "kind": "cvm_bundle",
            "state": "approved",
            "platform": "intel_tdx",
            "profile_version": "cpu-2026.09",
            "build_id": "cvm-fixture",
        }
        self.artifact_type = CVM_ARTIFACT_TYPE
        self.reference = "registry.example.org/cvm/cpu@sha256:" + "a" * 64

    def materialize(self, source, output, **options):
        self.materialized = output
        output.mkdir()
        return output, {"artifactType": self.artifact_type}, self.metadata

    def test_local_folder_does_not_contact_registry(self):
        with (
            patch("cvm.build.vault.load_profile_set", return_value=self.profiles) as load,
            patch("cvm.build.vault.materialize") as pull,
        ):
            with vault.profile_from_image(str(self.root)) as profiles:
                self.assertEqual(profiles, self.profiles)
            load.assert_called_once_with(self.root / "profile_set.json", approved=True)
            pull.assert_not_called()
        self.assertTrue(self.root.exists())

    def test_remote_bundle_is_verified_and_retained_through_packaging(self):
        with (
            patch("cvm.build.vault.materialize", side_effect=self.materialize) as pull,
            patch("cvm.build.vault.load_profile_set", return_value=self.profiles) as load,
        ):
            with vault.profile_from_image(self.reference, plain_http=True) as profiles:
                self.assertEqual(profiles, self.profiles)
                self.assertTrue(self.materialized.is_dir())
                load.assert_called_once_with(self.materialized / "profile_set.json", approved=True)
                pull.assert_called_once_with(self.reference, self.materialized, plain_http=True)
            self.assertFalse(self.materialized.exists())

    def test_remote_cleanup_preserves_vault_output_on_failure(self):
        output = self.root / "vault-output"
        output.mkdir()
        record = output / "build_failure.json"
        record.write_text('{"upload_acknowledged": false}')
        with (
            patch("cvm.build.vault.materialize", side_effect=self.materialize),
            patch("cvm.build.vault.load_profile_set", return_value=self.profiles),
        ):
            with self.assertRaisesRegex(BuildError, "uncertain key upload"):
                with vault.profile_from_image(self.reference):
                    raise BuildError("uncertain key upload")
            self.assertFalse(self.materialized.exists())
            self.assertTrue(record.is_file())

    def test_remote_artifact_type_state_and_identity_must_match(self):
        original = copy.deepcopy(self.metadata)
        for field, value in (
            ("artifact_type", DELIVERY_ARTIFACT_TYPE),
            ("state", "pending"),
            ("state", []),
            ("profile_version", "another-profile"),
            ("platform", "amd_sev_snp"),
            ("platform", []),
            ("build_id", "another-bundle"),
        ):
            self.metadata = dict(original)
            self.artifact_type = CVM_ARTIFACT_TYPE
            if field == "artifact_type":
                self.artifact_type = value
            else:
                self.metadata[field] = value
            with (
                self.subTest(field=field, value=value),
                patch("cvm.build.vault.materialize", side_effect=self.materialize),
                patch("cvm.build.vault.load_profile_set", return_value=self.profiles),
                self.assertRaises(BuildError),
            ):
                with vault.profile_from_image(self.reference):
                    self.fail("Invalid artifact was accepted")
            self.assertFalse(self.materialized.exists())

    def test_platforms_default_to_available_bundles_and_allow_a_subset(self):
        self.assertEqual(vault.requested_platforms({}, self.profiles), ["intel_tdx"])
        self.profiles["bundles"]["amd_sev_snp"] = {}
        self.assertEqual(set(vault.requested_platforms({}, self.profiles)), {"intel_tdx", "amd_sev_snp"})
        self.assertEqual(vault.requested_platforms({"platforms": ["amd_sev_snp"]}, self.profiles), ["amd_sev_snp"])
        for platforms in ([], ["other"], ["intel_tdx", "intel_tdx"], "intel_tdx", None):
            with self.subTest(platforms=platforms), self.assertRaises(BuildError):
                vault.requested_platforms({"platforms": platforms}, self.profiles)

    def test_build_generates_one_distinct_identity_per_invocation(self):
        identities = []

        def build(app, profiles, output, candidate, dev):
            identities.append(app["deployment_id"])
            self.assertEqual(profiles, self.profiles)
            return output

        with (
            patch("cvm.build.vault.config.application", side_effect=lambda _: {"cvm_image": str(self.root)}),
            patch("cvm.build.vault.config.project", return_value={"trustee": {}}),
            patch(
                "cvm.build.vault.profile_from_image", side_effect=lambda *a, **kw: contextlib.nullcontext(self.profiles)
            ),
            patch("cvm.build.vault.build_with_profile", side_effect=build),
        ):
            for output in (self.root / "first", self.root / "second"):
                self.assertEqual(vault.build("vault.yml", output=output), output)
        self.assertNotEqual(*identities)
        for identity in identities:
            self.assertEqual(uuid.UUID(identity).version, 4)
            self.assertEqual(len(identity), 32)

    def test_project_credentials_are_loaded_before_retrieval(self):
        service = {"url": "https://keys.test"}
        with (
            patch("cvm.build.vault.config.application", return_value={"cvm_image": self.reference}),
            patch("cvm.build.vault.config.project", return_value={"trustee": service}) as project,
            patch("cvm.build.vault.profile_from_image", return_value=contextlib.nullcontext(self.profiles)),
            patch("cvm.build.vault.build_with_profile") as seal,
        ):
            vault.build("/tmp/build.yml", project_config="/srv/project/cvm_project.yml")
            project.assert_called_once_with("/tmp/build.yml", "/srv/project/cvm_project.yml")
            self.assertEqual(seal.call_args.args[0]["trustee"], service)

        with (
            patch("cvm.build.vault.config.application", return_value={"cvm_image": self.reference}),
            patch("cvm.build.vault.config.project", side_effect=BuildError("Missing project")),
            patch("cvm.build.vault.profile_from_image") as retrieve,
            self.assertRaises(BuildError),
        ):
            vault.build("/tmp/build.yml")
        retrieve.assert_not_called()

    def test_dev_skips_project_credentials_and_rejects_explicit_project(self):
        with (
            patch("cvm.build.vault.config.application", side_effect=lambda _: {"cvm_image": str(self.root)}),
            patch("cvm.build.vault.config.project") as project,
            patch(
                "cvm.build.vault.profile_from_image", side_effect=lambda *a, **kw: contextlib.nullcontext(self.profiles)
            ),
            patch("cvm.build.vault.build_with_profile") as seal,
        ):
            vault.build("vault.yml", dev=True)
            self.assertNotIn("trustee", seal.call_args.args[0])
            with self.assertRaisesRegex(BuildError, "--dev"):
                vault.build("vault.yml", dev=True, project_config="cvm_project.yml")
            project.assert_not_called()
            seal.assert_called_once()
