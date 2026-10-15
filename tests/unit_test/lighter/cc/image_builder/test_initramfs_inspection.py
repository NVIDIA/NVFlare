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
    layouts = ("init", "main/init")

    def inspect(self, make_entry=None, init_paths=("main/init",)):
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
                self.assertEqual(command, ["unmkinitramfs", bundle / "initrd.img", root / "initramfs"])
                extracted = command[2]
                extracted.mkdir()
                for name in init_paths:
                    init = extracted / name
                    init.parent.mkdir(parents=True, exist_ok=True)
                    init.write_text("#!/bin/sh\n")
                if make_entry:
                    make_entry(extracted)

            with patch.object(hardware, "run", side_effect=extract):
                return case.inspect_initramfs()

    def test_flat_and_split_layouts_are_supported(self):
        for layout in self.layouts:
            with self.subTest(layout=layout):
                self.assertEqual(self.inspect(init_paths=(layout,))["inspected_files"], 1)

    def test_both_init_locations_are_scanned(self):
        self.assertEqual(self.inspect(init_paths=self.layouts)["inspected_files"], 2)

    def test_missing_or_nested_only_init_is_rejected(self):
        for init_paths in ((), ("early/init",), ("other/main/init",)):
            with self.subTest(init_paths=init_paths):
                with self.assertRaisesRegex(AssertionError, "init"):
                    self.inspect(init_paths=init_paths)

    def test_invalid_init_candidates_are_rejected(self):
        for layout in self.layouts:
            for kind in ("symlink", "broken_symlink", "directory"):
                with self.subTest(layout=layout, kind=kind):

                    def make_entry(extracted):
                        candidate = extracted / layout
                        candidate.parent.mkdir(parents=True, exist_ok=True)
                        if kind == "directory":
                            candidate.mkdir()
                        else:
                            target = extracted / "real-init"
                            if kind == "symlink":
                                target.write_text("#!/bin/sh\n")
                            candidate.symlink_to(target)

                    with self.assertRaisesRegex(AssertionError, "init"):
                        self.inspect(make_entry, init_paths=())

    def test_main_parent_symlink_is_not_a_valid_init(self):
        def make_entry(extracted):
            (extracted / "main").symlink_to("other", target_is_directory=True)

        with self.assertRaisesRegex(AssertionError, "init"):
            self.inspect(make_entry, init_paths=("other/init",))

    def test_valid_flat_init_allows_unrelated_main_symlink(self):
        def make_entry(extracted):
            (extracted / "main").symlink_to("other", target_is_directory=True)

        self.assertEqual(self.inspect(make_entry, init_paths=("init", "other/init"))["inspected_files"], 2)

    def test_forbidden_name_symlinks_are_rejected(self):
        for layout in self.layouts:
            for target in ("../" + layout, "missing"):
                with self.subTest(layout=layout, target=target):

                    def make_entry(extracted):
                        (extracted / "bin").mkdir()
                        (extracted / "bin/kbs-client").symlink_to(target)

                    with self.assertRaisesRegex(AssertionError, "bin/kbs-client"):
                        self.inspect(make_entry, init_paths=(layout,))

    def test_symlink_inside_etc_cvm_is_rejected(self):
        for layout in self.layouts:
            with self.subTest(layout=layout):

                def make_entry(extracted):
                    (extracted / "etc/cvm").mkdir(parents=True)
                    (extracted / "etc/cvm/certificate").symlink_to("../../" + layout)

                with self.assertRaisesRegex(AssertionError, "etc/cvm/certificate"):
                    self.inspect(make_entry, init_paths=(layout,))

    def test_exact_etc_cvm_directory_or_symlink_is_rejected(self):
        for layout in self.layouts:
            for symlink in (False, True):
                with self.subTest(layout=layout, symlink=symlink):

                    def make_entry(extracted):
                        (extracted / "etc").mkdir()
                        if symlink:
                            (extracted / "etc/cvm").symlink_to("..", target_is_directory=True)
                        else:
                            (extracted / "etc/cvm").mkdir()

                    with self.assertRaisesRegex(AssertionError, "etc/cvm"):
                        self.inspect(make_entry, init_paths=(layout,))

    def test_allowed_symlink_content_is_not_followed(self):
        for layout in self.layouts:
            with self.subTest(layout=layout):

                def make_entry(extracted):
                    (extracted / "bin").mkdir()
                    (extracted / "bin/helper").symlink_to("../" + layout)

                self.assertEqual(self.inspect(make_entry, init_paths=(layout,))["inspected_files"], 1)

    def test_forbidden_files_in_any_archive_are_rejected(self):
        for layout in self.layouts:
            for archive in ("", "main", "early", "early2"):
                for name in ("kbs-client", "kbs-ca.pem", "as-public.pem", "runtime.json"):
                    with self.subTest(layout=layout, archive=archive, name=name):

                        def make_entry(extracted):
                            path = extracted / archive / "bin" / name
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_text("forbidden")

                        with self.assertRaisesRegex(AssertionError, "CVM KBS material"):
                            self.inspect(make_entry, init_paths=(layout,))

    def test_endpoint_in_any_archive_is_rejected_including_across_chunk_boundary(self):
        endpoint = b"https://kbs.example.test"
        for layout in self.layouts:
            for archive in ("", "main", "early", "early2"):
                for prefix_size in (0, 1024 * 1024 - len(endpoint) // 2):
                    with self.subTest(layout=layout, archive=archive, prefix_size=prefix_size):

                        def make_entry(extracted):
                            path = extracted / archive / "config"
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_bytes(b"x" * prefix_size + endpoint)

                        with self.assertRaisesRegex(AssertionError, "configured KBS endpoint"):
                            self.inspect(make_entry, init_paths=(layout,))


if __name__ == "__main__":
    unittest.main()
