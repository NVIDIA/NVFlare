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

"""Exercise the real Ubuntu finalrd image in a disposable mount namespace."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[5]
HOOK = REPOSITORY / "nvflare/lighter/cc/image_builder/initramfs/finalrd/cvm_shutdown.finalrd"


@unittest.skipUnless(os.environ.get("CVM_SHUTDOWN_TESTS") == "1", "Opt-in Ubuntu finalrd test; requires root")
class ShutdownTests(unittest.TestCase):
    def test_shutdown_libraries_load_without_access_to_the_guest_root(self):
        self.assertEqual(os.geteuid(), 0, "Run this integration test as root")
        with tempfile.TemporaryDirectory(prefix="cvm-finalrd-") as temporary:
            directory = Path(temporary)
            shutil.copy2(HOOK, directory / HOOK.name)
            # No power-off command runs here. The dynamic loader only lists
            # dependencies in a chroot containing the finished shutdown image.
            # /run and /etc are private mounts, never host writes. Only public
            # dynamic-linker configuration is copied into the temporary /etc.
            script = """set -eu
mount --make-rprivate /
mount -t tmpfs -o mode=0755 tmpfs /run
mkdir /run/host_etc
mount --bind /etc /run/host_etc
mount -t tmpfs -o mode=0755 tmpfs /etc
cp -a /run/host_etc/ld.so.conf /run/host_etc/ld.so.cache /run/host_etc/ld.so.conf.d /etc/
umount /run/host_etc
mkdir /etc/finalrd
mount --bind "$1" /etc/finalrd
/usr/bin/finalrd
test -x /run/initramfs/shutdown
test -x /run/initramfs/bin/sh
for library in libmount.so.1 libblkid.so.1; do
    chroot /run/initramfs /lib64/ld-linux-x86-64.so.2 --list /usr/lib/x86_64-linux-gnu/$library
done
"""
            result = subprocess.run(
                ["unshare", "--mount", "--fork", "sh", "-s", "--", str(directory)],
                input=script,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
