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

"""Real Linux block-device tests. Run only on a dedicated Linux test host as root."""

import os
import tempfile
import unittest
from pathlib import Path

from cvm.build.config import public_sidecar
from cvm.build.storage import create_image, format_vault, mounted, nbd, opened_vault, sidecar
from cvm.common.contracts import binding
from cvm.common.errors import BuildError
from cvm.common.linux import memory_file, run
from cvm.common.luks import inspect_header, scan, snapshot_header


@unittest.skipUnless(os.environ.get("CVM_STORAGE_TESTS") == "1" and os.geteuid() == 0, "Opt-in root storage tests")
class AuthenticatedStorageTests(unittest.TestCase):
    def test_clear_input_sidecar_preserves_content(self):
        with tempfile.TemporaryDirectory(prefix="cvm-clear-sidecar-test-") as directory:
            source = Path(directory) / "source"
            source.mkdir()
            (source / "training.csv").write_text("feature,label\n1,2\n")
            image = Path(directory) / "user_data.qcow2"
            sidecar(image, 1, source, verify=public_sidecar, nfs_input=True)
            with nbd(image, readonly=True) as device, mounted(device, readonly=True) as root:
                self.assertEqual((root / "training.csv").read_text(), "feature,label\n1,2\n")
                self.assertTrue((root / "mnt").is_dir())

    def test_completed_public_sidecar_is_scanned_again(self):
        with tempfile.TemporaryDirectory(prefix="cvm-public-sidecar-test-") as directory:
            source = Path(directory) / "source"
            source.mkdir()
            # The first configuration scan is intentionally bypassed here to
            # model an input file swapped while the image is being copied.
            (source / "renamed.txt").write_text("-----BEGIN PRIVATE KEY-----\nopaque\n")
            with self.assertRaises(BuildError):
                sidecar(Path(directory) / "user_data.qcow2", 1, source, verify=public_sidecar)

    def test_payload_corruption_is_eio_with_unchanged_header(self):
        with tempfile.TemporaryDirectory(prefix="cvm-corruption-test-") as directory:
            image = Path(directory) / "vault.qcow2"
            create_image(image, 256 * 1024**2)
            with nbd(image) as device, memory_file(os.urandom(64)) as key:
                format_vault(device, key)
                header = snapshot_header(device)
                with opened_vault(device, key) as mapper:
                    scan(mapper)
                # Well beyond header and journal: modify a data/tag region.
                with open(device, "r+b", buffering=0) as stream:
                    stream.seek(64 * 1024**2)
                    original = stream.read(4096)
                    stream.seek(64 * 1024**2)
                    stream.write(bytes(value ^ 0x55 for value in original))
                    os.fsync(stream.fileno())
                self.assertEqual(snapshot_header(device), header)
                with opened_vault(device, key) as mapper:
                    with self.assertRaises(OSError):
                        scan(mapper)

    def test_write_reopen_and_frozen_header(self):
        with tempfile.TemporaryDirectory(prefix="cvm-storage-test-") as directory:
            image = Path(directory) / "vault.qcow2"
            create_image(image, 256 * 1024**2)
            with nbd(image) as device, memory_file(os.urandom(64)) as key:
                format_vault(device, key)
                inspect_header(device)
                header = snapshot_header(device)
                expected = binding(header)
                with memory_file(header, sealed=True) as frozen:
                    with opened_vault(device, key, header_fd=frozen) as mapper:
                        scan(mapper)
                        run(["mkfs.ext4", "-q", "-F", mapper])
                        with mounted(mapper) as root:
                            (root / "application.txt").write_text("generic application data\n")
                    self.assertEqual(expected, binding(snapshot_header(device)))
                    # A changed backing header must not replace the authenticated snapshot.
                    with open(device, "r+b", buffering=0) as stream:
                        stream.seek(4096)
                        original = stream.read(1)
                        stream.seek(4096)
                        stream.write(bytes([original[0] ^ 1]))
                        os.fsync(stream.fileno())
                    with opened_vault(device, key, header_fd=frozen) as mapper:
                        scan(mapper)
                        with mounted(mapper) as root:
                            self.assertEqual((root / "application.txt").read_text(), "generic application data\n")
                    self.assertNotEqual(expected, binding(snapshot_header(device)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
