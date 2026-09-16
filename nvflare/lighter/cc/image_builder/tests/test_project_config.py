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

"""Project configuration discovery and credential path boundaries."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from builder import config
from builder.common import BuildError


class ProjectConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.build = self.root / "project/config/generated/vault_build.yml"
        self.build.parent.mkdir(parents=True)
        self.build.write_text("{}\n")

    def write_project(self, directory, hostname="keys.test"):
        directory.mkdir(parents=True, exist_ok=True)
        credentials = directory / "credentials"
        credentials.mkdir(exist_ok=True)
        for name in ("ca", "cert", "key"):
            (credentials / name).write_text("fixture")
        value = {"key_service": {"url": "https://" + hostname}}
        value["key_service"].update({name: "credentials/" + name for name in ("ca", "cert", "key")})
        path = directory / "cvm_project.yml"
        path.write_text(yaml.safe_dump(value))
        return path

    def test_ancestor_discovery_and_paths_use_project_not_build_directory(self):
        path = self.write_project(self.root / "project")
        service = config.project(self.build)["key_service"]
        self.assertEqual(service["key"], str(path.parent / "credentials/key"))
        self.assertEqual(service["url"], "https://keys.test")

    def test_nearest_project_wins_and_invalid_nearest_does_not_fall_back(self):
        self.write_project(self.root / "project", "outer.test")
        nearest = self.write_project(self.build.parent, "nearest.test")
        self.assertEqual(config.project(self.build)["key_service"]["url"], "https://nearest.test")
        nearest.write_text("{}\n")
        with self.assertRaises(BuildError):
            config.project(self.build)

    def test_explicit_project_overrides_discovery_for_staged_yaml(self):
        self.write_project(self.build.parent, "local.test")
        path = self.write_project(self.root / "outside", "selected.test")
        value = config.project(self.build, path)
        self.assertEqual(value["key_service"]["url"], "https://selected.test")
        self.assertEqual(value["key_service"]["ca"], str(path.parent / "credentials/ca"))
        with self.assertRaises(BuildError):
            config.project(self.build, self.root / "missing.yml")

    def test_missing_project_fails(self):
        # Do not depend on whether the workstation has its own ancestor config.
        with (
            patch.object(Path, "exists", return_value=False),
            patch.object(Path, "is_symlink", return_value=False),
            self.assertRaisesRegex(BuildError, "--project-config"),
        ):
            config.project(self.build)

    def test_invalid_project_schema_or_endpoint_is_rejected(self):
        path = self.write_project(self.root / "project")
        valid = yaml.safe_load(path.read_text())
        for value in (
            {},
            {"key_service": None},
            {"key_service": valid["key_service"], "unknown": True},
            {"key_service": {**valid["key_service"], "token": "unexpected"}},
            {"key_service": {"url": "https://keys.test"}},
        ):
            path.write_text(yaml.safe_dump(value))
            with self.subTest(value=value), self.assertRaises(BuildError):
                config.project(self.build)
        for url in (
            None,
            [],
            "http://keys.test",
            "https:///missing-host",
            "https://user:pass@keys.test",
            "https://keys.test?q=1",
        ):
            value = {"key_service": {**valid["key_service"], "url": url}}
            path.write_text(yaml.safe_dump(value))
            with self.subTest(url=url), self.assertRaises(BuildError):
                config.project(self.build)

    def test_missing_credential_and_duplicate_yaml_key_are_rejected(self):
        path = self.write_project(self.root / "project")
        (path.parent / "credentials/key").unlink()
        with self.assertRaises(BuildError):
            config.project(self.build)
        path.write_text("key_service: {}\nkey_service: {}\n")
        with self.assertRaises(BuildError):
            config.project(self.build)
