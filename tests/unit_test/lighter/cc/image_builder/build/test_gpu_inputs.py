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

"""Reproducible GPU inputs on a clean construction image."""

import copy
import tempfile
import unittest
from pathlib import Path

from cvm.build.config import PROFILE_DEFAULTS, SOURCE, gpu_inputs
from cvm.build.cvm import contract, provisioning_payload
from cvm.build.provisioning import install_apt_repositories
from cvm.common.errors import BuildError
from cvm.common.io import digest_file, read_json, write_json
from cvm.common.versions import NVAT_COMMIT


class GpuInputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.keyring = self.root / "repository.gpg"
        self.keyring.write_bytes(b"fixture public keyring")
        self.library = self.root / "libnvat.so.1"
        self.library.write_bytes(b"fixture NVAT library")
        self.provenance = {
            "source_repository": "https://github.com/NVIDIA/attestation-sdk.git",
            "source_commit": NVAT_COMMIT,
            "patch_sha256": digest_file(SOURCE / "cvm/build/nvat_libxml2_const.patch"),
            "library_sha256": digest_file(self.library),
            "build_environment": "ubuntu-26.04-x86_64",
        }
        self.record = self.root / "nvat_build.json"
        write_json(self.record, self.provenance)
        self.repository = {
            "url": "https://nvidia.github.io/libnvidia-container/stable/deb/amd64",
            "suite": "/",
            "components": [],
            "keyring": self.keyring.name,
            "keyring_sha256": digest_file(self.keyring),
        }
        self.value = {
            "gpu_attestation_library": str(self.library),
            "gpu_attestation_provenance": str(self.record),
            "gpu_apt_repositories": [self.repository],
        }

    def test_repository_payload_installs_only_scoped_signing_key(self):
        gpu_inputs(self.root / "profile.yml", self.value)
        profile = dict(
            PROFILE_DEFAULTS,
            **self.value,
            gpu="nvidia_cc",
            kbs_cert=self.keyring,
            as_public_key=self.keyring,
            platforms={"intel_tdx": {"kbs_client": self.keyring}},
        )
        provisioning_payload(profile, "intel_tdx", "test-gpu", self.root, SOURCE, self.keyring)
        payload = self.root / "provision-payload"
        config = read_json(payload / "config.json")
        guest = self.root / "guest"
        install_apt_repositories(config, payload, guest)
        source = (guest / "etc/apt/sources.list.d/cvm_gpu_0.sources").read_text()
        self.assertIn("Signed-By: /usr/share/keyrings/cvm_gpu_0.gpg", source)
        self.assertIn("Architectures: amd64", source)
        self.assertNotIn("Trusted:", source)
        self.assertEqual((guest / "usr/share/keyrings/cvm_gpu_0.gpg").read_bytes(), self.keyring.read_bytes())
        (payload / "inputs/gpu_apt_0.gpg").write_bytes(b"substituted key")
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            install_apt_repositories(config, payload, guest)

    def test_repository_credentials_injections_and_unpinned_keys_rejected(self):
        for key, value in (
            ("url", "http://mirror.example.org"),
            ("url", "https://user:password@mirror.example.org"),
            ("suite", "/\nTrusted: yes"),
            ("components", ["main\nTrusted: yes"]),
            ("components", ["main"]),
            ("keyring_sha256", ""),
            ("keyring_sha256", "ab" * 32),
            ("trusted", True),
        ):
            value_config = copy.deepcopy(self.value)
            value_config["gpu_apt_repositories"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(BuildError):
                gpu_inputs(self.root / "profile.yml", value_config)
        with self.assertRaisesRegex(BuildError, "authenticated"):
            gpu_inputs(self.root / "profile.yml", dict(self.value, gpu_apt_repositories=[]))

    def test_nvat_revision_patch_environment_and_library_are_bound(self):
        gpu_inputs(self.root / "profile.yml", self.value)
        for key, value in (
            ("source_commit", "ab" * 20),
            ("source_repository", "https://example.org/unreviewed"),
            ("patch_sha256", "ab" * 32),
            ("library_sha256", "ab" * 32),
            ("build_environment", "ubuntu-24.04-x86_64"),
        ):
            write_json(self.record, dict(self.provenance, **{key: value}))
            with self.subTest(key=key), self.assertRaises(BuildError):
                gpu_inputs(self.root / "profile.yml", self.value)
        write_json(self.record, self.provenance)
        self.library.write_bytes(b"changed library under the same filename")
        with self.assertRaisesRegex(BuildError, "library digest"):
            gpu_inputs(self.root / "profile.yml", self.value)

    def test_contract_records_source_provenance_and_repository_identity(self):
        gpu_inputs(self.root / "profile.yml", self.value)
        profile = dict(PROFILE_DEFAULTS, **self.value, gpu="nvidia_cc", root_overlay_max_mib=4096)
        for key in (
            "base_image",
            "build_firmware",
            "kbs_cert",
            "as_public_key",
            "attestation_policy",
            "reference_values",
            "gpu_policy",
        ):
            profile[key] = self.keyring
        profile.update(
            gpu_packages=["nvidia-container-toolkit=1.20.0-1"], gpu_attestation_url="https://nras.example.org"
        )
        original = contract(profile, self.root)
        self.assertEqual(original["gpu_attestation_provenance"], self.provenance)
        self.assertEqual(original["gpu_apt_repositories"][0]["keyring_sha256"], digest_file(self.keyring))
        self.assertNotIn("keyring", original["gpu_apt_repositories"][0])
        profile["gpu_apt_repositories"][0]["url"] = "https://mirror.example.org"
        self.assertNotEqual(original, contract(profile, self.root))
