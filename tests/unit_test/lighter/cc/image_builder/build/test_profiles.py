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

"""Profile identity and fixed platform directories without redundant metadata."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cvm.artifacts.bundle import load_profile_set
from cvm.build import config
from cvm.build import cvm as builder
from cvm.common.errors import BuildError
from cvm.common.io import digest_file, read_json, write_json


class ProfileTests(unittest.TestCase):
    def test_gpu_profile_options_are_accepted_and_references_checked_once(self):
        settings = {
            "gpu": "nvidia_cc",
            "gpu_policy": "policy.json",
            "gpu_packages": ["nvidia-driver-580-open=version", "nvidia-container-toolkit=version"],
            "gpu_attestation_url": "https://nras.example.org",
            "gpu_attestation_library": "libnvat.so.1",
            "gpu_attestation_provenance": "nvat_build.json",
            "gpu_apt_repositories": [{"url": "https://packages.example.org"}],
            "acceptance_runner": "site_acceptance",
            "root_overlay_max_mib": 4096,
        }
        with (
            patch.object(config, "load_yaml", return_value=settings),
            patch.object(config, "local_path", side_effect=lambda path, value: value),
            patch.object(config, "read_json", return_value={}) as read,
            patch.object(config, "validate_references") as validate,
            patch.object(config, "gpu_inputs") as gpu_inputs,
            patch.object(config, "validate_gpu_policy"),
        ):
            profile = config.profile("profile.yml")
        self.assertEqual(profile["gpu_attestation_provenance"], "nvat_build.json")
        read.assert_called_once_with(profile["reference_values"])
        validate.assert_called_once_with({}, ["amd_sev_snp", "intel_tdx"], gpu=True)
        gpu_inputs.assert_called_once()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def candidate(self, platform, version="dev-cpu-2026.09"):
        directory = self.root / platform
        directory.mkdir()
        for name in (
            "verity_root.qcow2",
            "OVMF.fd",
            "vmlinuz",
            "initrd.img",
            "attestation_policy.rego",
            "launch_cvm.sh.tmpl",
            "shutdown_cvm.sh.tmpl",
        ):
            (directory / name).write_text("fixture")
        write_json(directory / "reference_values.json", {})
        cmdline = builder.kernel_command_line("ab" * 32, 4096, 4096)
        manifest = {
            "schema_version": 2,
            "profile_version": version,
            "platform": platform,
            "build_id": "cvm-" + platform,
            "dev_mode": True,
            "contract": {"gpu": "none", "root_overlay_max_mib": 4096},
            "launch_shape": {},
            "cmdline": cmdline,
            "cmdline_sha256": hashlib.sha256(cmdline.encode()).hexdigest(),
            "sha256": {path.name: digest_file(path) for path in directory.iterdir()},
        }
        write_json(directory / "cvm_manifest.pending.json", manifest)
        return directory

    def test_contract_does_not_depend_on_profile_label(self):
        profile = dict(config.PROFILE_DEFAULTS, root_overlay_max_mib=4096)
        with patch("cvm.build.cvm.digest_file", return_value="ab" * 32):
            first = builder.contract(profile, self.root)
            second = builder.contract(dict(profile, profile_version="cpu-2026.10"), self.root)
        self.assertNotIn("profile_version", first)
        self.assertEqual(first, second)

    def test_finalize_and_load_use_platform_directory_without_path(self):
        for platform in ("amd_sev_snp", "intel_tdx"):
            builder.finalize(self.candidate(platform), package=False)
        path = self.root / "profile_set.json"
        profiles = read_json(path)
        self.assertEqual(profiles["profile_version"], "dev-cpu-2026.09")
        for entry in profiles["bundles"].values():
            self.assertEqual(set(entry), {"build_id", "manifest_sha256"})
        loaded = load_profile_set(path, approved=False)
        for platform, entry in loaded["bundles"].items():
            self.assertEqual(Path(entry["directory"]), self.root.resolve() / platform)

        for platform in ("../intel_tdx", "/intel_tdx", "custom"):
            profiles["bundles"] = {platform: {}}
            write_json(path, profiles)
            with self.subTest(platform=platform), self.assertRaisesRegex(BuildError, "Invalid platform"):
                load_profile_set(path, approved=False)

    def test_different_profile_versions_cannot_be_finalized_or_loaded_together(self):
        builder.finalize(self.candidate("amd_sev_snp"), package=False)
        path = self.root / "profile_set.json"
        original = path.read_bytes()
        other = self.candidate("intel_tdx", "dev-cpu-2026.10")
        with self.assertRaisesRegex(BuildError, "Profile version differs"):
            builder.finalize(other, package=False)
        self.assertEqual(path.read_bytes(), original)

        profiles = read_json(path)
        profiles["profile_version"] = "dev-cpu-2026.10"
        write_json(path, profiles)
        with self.assertRaisesRegex(BuildError, "Profile identity mismatch"):
            load_profile_set(path, approved=False)

    def test_build_rejects_output_directory_for_another_profile_version(self):
        builder.finalize(self.candidate("amd_sev_snp"), package=False)
        shared = read_json(self.root / "profile_set.json")["contract"]
        with (
            patch("cvm.build.cvm.linux_root"),
            patch("cvm.build.cvm.config.profile", return_value={"profile_version": "dev-cpu-2026.10"}),
            patch("cvm.build.cvm.select_platform", return_value="intel_tdx"),
            patch("cvm.build.cvm.contract", return_value=shared),
            patch("cvm.build.cvm.plain_build") as construct,
        ):
            with self.assertRaisesRegex(BuildError, "Profile version differs"):
                builder.build(None, output=self.root, dev=True)
            construct.assert_not_called()
        self.assertFalse((self.root / "intel_tdx").exists())
