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

"""Opt-in release packaging check, including a wheel built from the sdist."""

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[5]
BUILDER = Path("nvflare/lighter/cc/image_builder")


@unittest.skipUnless(os.environ.get("CVM_DISTRIBUTION_TESTS") == "1", "Opt-in distribution build")
class DistributionTests(unittest.TestCase):
    def test_wheel_from_sdist_contains_usable_builder(self):
        with tempfile.TemporaryDirectory(prefix="cvm-distribution-") as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            for name in ("nvflare", "job_templates"):
                shutil.copytree(
                    REPOSITORY / name,
                    source / name,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".venv", "target", "inputs"),
                )
            for name in (
                "setup.py",
                "setup.cfg",
                "pyproject.toml",
                "versioneer.py",
                "MANIFEST.in",
                "README.md",
                "LICENSE",
            ):
                shutil.copy2(REPOSITORY / name, source / name)
            env = dict(os.environ, NVFL_BASE_VERSION="2.9.0", NVFL_RELEASE="1", PYTHONPATH="")

            def run(command, cwd, environment=env):
                result = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, text=True, timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            distribution = root / "dist"
            distribution.mkdir()
            run(
                [
                    sys.executable,
                    "-c",
                    "import sys; from setuptools.build_meta import build_sdist; build_sdist(sys.argv[1])",
                    str(distribution),
                ],
                source,
            )
            with tarfile.open(next(distribution.glob("*.tar.gz"))) as archive:
                # This archive was just built from the isolated source copy.
                archive.extractall(root / "sdist", **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
            unpacked = next((root / "sdist").iterdir())
            run(
                [
                    sys.executable,
                    "-c",
                    "import sys; from setuptools.build_meta import build_wheel; build_wheel(sys.argv[1])",
                    str(distribution),
                ],
                unpacked,
            )
            installed = root / "installed"
            with zipfile.ZipFile(next(distribution.glob("*.whl"))) as wheel:
                wheel.extractall(installed)
                self.assertTrue(wheel.getinfo((BUILDER / "cvmctl").as_posix()).external_attr >> 16 & 0o111)
            expected = [
                path.relative_to(source)
                for path in (source / BUILDER).rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
            ]
            self.assertIn(BUILDER / "cvm/build/nvat_libxml2_const.patch", expected)
            for relative in expected:
                with self.subTest(asset=relative.as_posix()):
                    self.assertEqual((unpacked / relative).read_bytes(), (source / relative).read_bytes())
                    self.assertEqual((installed / relative).read_bytes(), (source / relative).read_bytes())
            builder = installed / BUILDER
            run(["sh", str(builder / "cvmctl"), "--help"], root, dict(env, CVM_BUILDER_PYTHON=sys.executable))
            run(
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    str(REPOSITORY / "tests/unit_test/lighter/cc/image_builder/build"),
                    "-p",
                    "test_gpu_inputs.py",
                    "-v",
                ],
                root,
                dict(env, PYTHONPATH=str(builder)),
            )
