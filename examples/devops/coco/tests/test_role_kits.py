# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Verify reproducible assembly and isolated-role preflight without installing anything."""

import importlib.util
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("role_kits", ROOT / "role_kits.py")
kits = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kits)


class RoleKitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = kits.assemble(ROOT, Path(self.temp.name) / "assembled")

    def test_copies_match_shared_sources_and_inventory(self):
        kits.inventory(self.output)
        kits.validate_layout(self.output, assembled=True)
        for target, source in kits.GENERATED.items():
            self.assertEqual((self.output / target).read_bytes(), (ROOT / source).read_bytes())
            if target.endswith(".sh"):
                self.assertTrue(os.access(self.output / target, os.X_OK))

    def test_assembly_is_byte_reproducible(self):
        second = kits.assemble(ROOT, Path(self.temp.name) / "second")
        for name in kits.inventory(self.output):
            self.assertEqual((self.output / name).read_bytes(), (second / name).read_bytes(), name)

    def test_never_overwrites_or_writes_inside_source(self):
        for target in (self.output, ROOT / "not-an-output"):
            with self.assertRaises(ValueError):
                kits.assemble(ROOT, target)

    def test_changed_generated_dependency_is_rejected(self):
        (self.output / "coco/lib/validate-config.sh").write_text("exit 0\n")
        with self.assertRaisesRegex(ValueError, "differs"):
            kits.validate_layout(self.output, assembled=True)

    def test_used_package_or_symlink_is_not_assembled(self):
        extra = self.output / "private-input.txt"
        extra.write_text("not a public input")
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            kits.assemble(self.output, Path(self.temp.name) / "rejected")
        extra.unlink()
        extra.symlink_to(self.output / "README.md")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            kits.assemble(self.output, Path(self.temp.name) / "rejected")

    def test_roles_work_without_siblings_or_shared_tree(self):
        for role in kits.ROLES:
            with self.subTest(role=role):
                isolated = Path(self.temp.name) / ("isolated-" + role)
                shutil.copytree(self.output / role, isolated)
                result = subprocess.run(
                    [
                        "bash",
                        "-c",
                        'source "$1"; validate_service_host secure.unit.local',
                        "test",
                        str(isolated / "lib/validate-config.sh"),
                    ],
                    cwd=self.temp.name,
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                if role not in ("coco", "trusted_system"):
                    continue
                helper = isolated / "bootstrap/lib/common.sh"
                loaded = subprocess.run(
                    ["bash", "-c", 'source "$1"; printf "%s" "$SUITE_DIR"', "test", str(helper)],
                    cwd=self.temp.name,
                    text=True,
                    capture_output=True,
                    env={k: v for k, v in os.environ.items() if k != "COCO_BOOTSTRAP_DIR"},
                )
                self.assertEqual(loaded.returncode, 0, loaded.stderr)
                self.assertEqual(loaded.stdout, str(isolated / "bootstrap"))
                installer = isolated / (
                    "02-install-kubernetes.sh" if role == "trusted_system" else "bootstrap/10-install-kubernetes.sh"
                )
                result = subprocess.run(
                    ["bash", str(installer)],
                    cwd=self.temp.name,
                    text=True,
                    capture_output=True,
                    env={
                        **os.environ,
                        "COCO_CONFIG": str(isolated / "missing.env"),
                        "COCO_BOOTSTRAP_DIR": str(isolated / "bootstrap"),
                    },
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Copy config.env.example", result.stderr)
                self.assertNotIn("No such file", result.stderr)

    def test_slides_have_one_source_and_three_slides(self):
        names = kits.inventory(ROOT)
        self.assertNotIn("docs/coco-security-design-3-slides.html", names)
        self.assertFalse(any(name.endswith((".pptx", ".pdf", "CURRENT-STATE.md")) for name in names))
        markdown = (ROOT / "docs/coco-security-design-3-slides.md").read_text()
        self.assertIn("marp: true", markdown)
        self.assertEqual(len([line for line in markdown.splitlines() if line.startswith("# ")]), 3)
        self.assertEqual(markdown.count("\n---\n"), 3)  # Frontmatter end plus two slide breaks.

    def test_exporter_command_contract_and_no_overwrite(self):
        binaries = Path(self.temp.name) / "bin"
        binaries.mkdir()
        marp = binaries / "marp"
        marp.write_text(
            '#!/bin/sh\nif [ "$1" = --version ]; then echo "@marp-team/marp-cli v4.5.1 (fixture)"; exit; fi\n'
            'while [ "$#" -gt 0 ]; do\n'
            '  if [ "$1" = --output ]; then shift; printf fixture > "$1"; exit; fi\n'
            "  shift\ndone\nexit 1\n"
        )
        node = binaries / "node"
        node.write_text("#!/bin/sh\necho v24.0.0\n")
        for executable in (marp, node):
            executable.chmod(0o755)
        output = Path(self.temp.name) / "exports"
        env = {**os.environ, "MARP_BIN": str(marp), "BROWSER_BIN": "", "PATH": f"{binaries}:{os.environ['PATH']}"}
        command = ["bash", str(ROOT / "docs/export-slides.sh")]
        result = subprocess.run(command + [str(output)], env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for suffix in ("html", "pdf", "pptx"):
            self.assertEqual((output / f"coco-security-design-3-slides.{suffix}").read_text(), "fixture")
        for destination in (output, ROOT / "not-an-export"):
            result = subprocess.run(command + [str(destination)], env=env, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
        marp.write_text('#!/bin/sh\necho "@marp-team/marp-cli v0.0.0 (fixture)"\n')
        result = subprocess.run(
            command + [str(output.parent / "wrong-version")], env=env, text=True, capture_output=True
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((output.parent / "wrong-version").exists())


if __name__ == "__main__":
    unittest.main()
