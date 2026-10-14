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

"""Archive inspection regression cases without CVM hardware or host tools."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

LAB = Path(__file__).resolve().parents[4] / "integration_test/lighter/cc/image_builder"


def load_hardware_tests():
    spec = importlib.util.spec_from_file_location("acceptance_hardware_tests", LAB / "test_hardware.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InitramfsInspectionTests(unittest.TestCase):
    def inspect(self, make_entry):
        hardware = load_hardware_tests()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            bundle.mkdir()
            case = hardware.HardwareTests("test_clear_sidecars_contain_no_private_keys")
            case.directory = root
            case.bundle = bundle
            case.disposables = []
            case.manifest = {"contract": {"kbs_url": "https://kbs.example.test"}}

            def extract(command):
                self.assertEqual(command[:1], ["unmkinitramfs"])
                extracted = command[2]
                (extracted / "main").mkdir(parents=True)
                (extracted / "main/init").write_text("#!/bin/sh\n")
                make_entry(extracted)

            with patch.object(hardware, "run", side_effect=extract):
                return case.inspect_initramfs()

    def test_forbidden_name_symlinks_are_rejected(self):
        for target in ("../main/init", "missing"):
            with self.subTest(target=target):

                def make_entry(extracted):
                    (extracted / "bin").mkdir()
                    (extracted / "bin/kbs-client").symlink_to(target)

                with self.assertRaisesRegex(AssertionError, "bin/kbs-client"):
                    self.inspect(make_entry)

    def test_symlink_inside_etc_cvm_is_rejected(self):
        def make_entry(extracted):
            (extracted / "etc/cvm").mkdir(parents=True)
            (extracted / "etc/cvm/certificate").symlink_to("../../main/init")

        with self.assertRaisesRegex(AssertionError, "etc/cvm/certificate"):
            self.inspect(make_entry)

    def test_exact_etc_cvm_directory_or_symlink_is_rejected(self):
        for symlink in (False, True):
            with self.subTest(symlink=symlink):

                def make_entry(extracted):
                    (extracted / "etc").mkdir()
                    if symlink:
                        (extracted / "etc/cvm").symlink_to("../main", target_is_directory=True)
                    else:
                        (extracted / "etc/cvm").mkdir()

                with self.assertRaisesRegex(AssertionError, "etc/cvm"):
                    self.inspect(make_entry)

    def test_allowed_symlink_content_is_not_followed(self):
        def make_entry(extracted):
            (extracted / "bin").mkdir()
            (extracted / "bin/helper").symlink_to("../main/init")

        self.assertEqual(self.inspect(make_entry)["inspected_files"], 1)


if __name__ == "__main__":
    unittest.main()
