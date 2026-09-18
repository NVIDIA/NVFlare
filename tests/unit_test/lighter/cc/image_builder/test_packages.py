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

"""Deployment isolation and dependency boundaries for the standalone CVM packages."""

import ast
import importlib.util
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cvm.build.config import PROFILE_DEFAULTS, SOURCE
from cvm.build.cvm import contract, provisioning_payload
from cvm.build.payload import GUEST_MODULES, HOST_MODULES, copy_modules
from cvm.build.provisioning import install_files
from cvm.common.firewall import firewall_rules


class PackageTests(unittest.TestCase):
    def isolated_imports(self, directory, modules):
        code = """
import importlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
for name in json.loads(sys.argv[2]):
    module = importlib.import_module("cvm." + name)
    assert pathlib.Path(module.__file__).is_relative_to(root), module.__file__
assert not any(name in sys.modules for name in ("cvm.build", "cvm.trustee"))
"""
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c", code, str(directory), json.dumps(modules)],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_guest_and_host_payloads_are_independently_importable(self):
        for modules, forbidden in (
            (GUEST_MODULES, ("build", "trustee", "host", "artifacts")),
            (HOST_MODULES, ("build", "trustee", "runtime")),
        ):
            with self.subTest(modules=modules), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                copy_modules(SOURCE, root, modules)
                self.isolated_imports(root, modules)
                for package in forbidden:
                    self.assertFalse((root / "cvm" / package).exists())
                self.assertFalse(any(root.rglob("*.pyc")))

    def test_package_dependencies_follow_execution_boundaries(self):
        allowed = {
            "common": {"common"},
            "artifacts": {"artifacts", "common"},
            "runtime": {"runtime", "common"},
            "host": {"host", "artifacts", "common"},
            "trustee": {"trustee", "artifacts", "common"},
            "build": {"build", "host", "artifacts", "trustee", "common"},
        }
        for path in (SOURCE / "cvm").rglob("*.py"):
            parts = path.relative_to(SOURCE).with_suffix("").parts
            if len(parts) < 3:
                continue
            package = ".".join(parts[:-1])
            for node in ast.walk(ast.parse(path.read_text())):
                imports = []
                if isinstance(node, ast.ImportFrom):
                    target = "." * node.level + (node.module or "")
                    imports = [importlib.util.resolve_name(target, package)]
                elif isinstance(node, ast.Import):
                    imports = [alias.name for alias in node.names]
                for target in imports:
                    if target.startswith("cvm."):
                        self.assertIn(target.split(".")[1], allowed[parts[1]], f"{path}: {target}")

    def test_payload_contains_every_local_dependency_including_lazy_imports(self):
        for modules in (GUEST_MODULES, HOST_MODULES):
            available = {"cvm", *("cvm." + module for module in modules)}
            available.update(module.rpartition(".")[0] for module in tuple(available) if "." in module)
            for name in modules:
                module = "cvm." + name
                path = SOURCE / (module.replace(".", "/") + ".py")
                for node in ast.walk(ast.parse(path.read_text())):
                    if isinstance(node, ast.ImportFrom):
                        target = importlib.util.resolve_name(
                            "." * node.level + (node.module or ""), module.rpartition(".")[0]
                        )
                        if target.startswith("cvm."):
                            self.assertIn(target, available, f"{module} needs {target}")

    def test_construction_payload_installs_only_the_guest_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            inputs = job / "input"
            inputs.write_text("fixture")
            profile = {
                "platforms": {"intel_tdx": {"kbs_client": inputs}},
                "kbs_cert": inputs,
                "as_public_key": inputs,
                "gpu": "none",
                "build_user": "ubuntu",
                "guest_release": "26.04",
                "kernel_version": "test-kernel",
                "profile_version": "test-profile",
                "required_system_packages": [],
                "bootstrap_egress": [443],
            }
            archive = provisioning_payload(profile, "intel_tdx", "test-build", job, SOURCE, inputs)
            with tarfile.open(archive) as stream:
                members = stream.getnames()
            self.assertIn("./provision_guest.py", members)
            self.assertNotIn("./source/cvm/build", members)
            self.assertNotIn("./source/cvm/trustee", members)
            payload = job / "provision-payload"
            self.assertEqual(
                (payload / "inputs/nftables.conf").read_text(), "flush ruleset\n" + firewall_rules([], [443])
            )
            config = json.loads((payload / "config.json").read_text())
            guest = job / "guest"
            vendor = guest / "usr/lib/systemd/system"
            vendor.mkdir(parents=True)
            (vendor / "docker.service").write_text(
                "[Unit]\nRequires=docker.socket\n[Service]\nExecStart=/usr/bin/dockerd -H fd://\n"
            )
            install_files(config, payload, guest)
            installed = guest / "usr/lib/cvm"
            self.isolated_imports(installed, GUEST_MODULES)
            for path in (SOURCE / "services").glob("*.service"):
                for line in path.read_text().splitlines():
                    if "python3 -m " in line:
                        module = line.split("python3 -m ", 1)[1].split()[0]
                        self.assertTrue((installed / (module.replace(".", "/") + ".py")).is_file(), module)
            self.assertFalse((installed / "cvm/build").exists())
            self.assertFalse((installed / "cvm/trustee").exists())

    def test_source_fingerprint_covers_runtime_provisioner_and_payload_manifest(self):
        profile = dict(PROFILE_DEFAULTS, root_overlay_max_mib=4096)
        with tempfile.TemporaryDirectory() as directory, patch("cvm.build.cvm.digest_file", return_value="ab" * 32):
            root = Path(directory)
            original = contract(profile, root)["runtime_source_sha256"]
            for name in ("cvm/runtime/bootstrap.py", "cvm/build/provisioning.py", "cvm/build/payload.py"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("changed source")
                changed = contract(profile, root)["runtime_source_sha256"]
                self.assertNotEqual(changed, original, name)
                original = changed
