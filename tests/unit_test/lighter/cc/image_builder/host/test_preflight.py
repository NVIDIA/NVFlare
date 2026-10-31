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

"""QGS configurations supplied by Intel packages and sectioned variants."""

import datetime
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from cvm.common.errors import BuildError
from cvm.host.preflight import (
    check_gpu_host,
    check_host,
    check_qgs_config,
    check_snp_host,
    check_snp_vcek_store,
    snp_vcek_paths,
)


class PreflightTests(unittest.TestCase):
    VCEK_URL = "https://kdsintf.amd.com/vcek/v1/Milan/" + "ab" * 64 + "?blSPL=2&teeSPL=0&snpSPL=6&ucodeSPL=33"

    @staticmethod
    def certificate():
        key = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test VCEK")])
        return (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(key.public_key())
            .serial_number(1)
            .not_valid_before(datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1))
            .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1))
            .sign(key, hashes.SHA256())
            .public_bytes(serialization.Encoding.DER)
        )

    def test_snp_preflight_checks_cpu_module_device_iommu_and_firmware(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            procfs, sysfs, devfs = root / "proc", root / "sys", root / "dev"
            procfs.mkdir()
            (sysfs / "module/kvm_amd/parameters").mkdir(parents=True)
            (sysfs / "kernel/iommu_groups/1").mkdir(parents=True)
            devfs.mkdir()
            firmware = root / "OVMF.amdsev.fd"
            firmware.touch()
            (procfs / "cpuinfo").write_text("vendor_id : AuthenticAMD\nflags : sev sev_es sev_snp\n")
            (sysfs / "module/kvm_amd/parameters/sev_snp").write_text("Y\n")
            (devfs / "sev").touch()
            self.assertTrue(check_snp_host(firmware, procfs=procfs, sysfs=sysfs, devfs=devfs))

            (sysfs / "module/kvm_amd/parameters/sev_snp").write_text("N\n")
            with self.assertRaisesRegex(BuildError, "BIOS"):
                check_snp_host(firmware, procfs=procfs, sysfs=sysfs, devfs=devfs)

    def test_gpu_preflight_accepts_vfio_bound_passthrough_device(self):
        with tempfile.TemporaryDirectory() as directory:
            sysfs = Path(directory)
            device = sysfs / "bus/pci/devices/0000:01:00.0"
            device.mkdir(parents=True)
            (device / "vendor").write_text("0x10de\n")
            (device / "class").write_text("0x030200\n")
            (device / "iommu_group").mkdir()
            driver = sysfs / "bus/pci/drivers/vfio-pci"
            driver.mkdir(parents=True)
            (device / "driver").symlink_to(driver)
            with patch("cvm.host.preflight.subprocess.run") as run:
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    check_gpu_host(sysfs)
            run.assert_not_called()
            self.assertIn("GPU IOMMU/VFIO screening passed", stdout.getvalue())
            self.assertIn("confidential-computing qualification have not run", stdout.getvalue())

    def test_gpu_preflight_checks_host_driver_devices_on_mixed_host(self):
        with tempfile.TemporaryDirectory() as directory:
            sysfs = Path(directory)
            drivers = sysfs / "bus/pci/drivers"
            vfio_driver, nvidia_driver = drivers / "vfio-pci", drivers / "nvidia"
            vfio_driver.mkdir(parents=True)
            nvidia_driver.mkdir()
            for address, driver in (("0000:01:00.0", vfio_driver), ("0000:02:00.0", nvidia_driver)):
                device = sysfs / "bus/pci/devices" / address
                device.mkdir(parents=True)
                (device / "vendor").write_text("0x10de\n")
                (device / "class").write_text("0x030200\n")
                (device / "iommu_group").mkdir()
                (device / "driver").symlink_to(driver)
            result = type("Result", (), {"returncode": 0})()
            stdout = io.StringIO()
            with patch("cvm.host.preflight.subprocess.run", return_value=result) as run, redirect_stdout(stdout):
                check_gpu_host(sysfs)
            self.assertEqual(run.call_count, 2)
            self.assertIn("1 host-driver", stdout.getvalue())
            self.assertIn("1 VFIO-bound device(s) passed only IOMMU/VFIO screening", stdout.getvalue())

    def test_tdx_secure_boot_preflight_requires_and_accepts_configured_shim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            firmware = root / "OVMF.inteltdx.ms.fd"
            shim = root / "shim.efi"
            firmware.touch()
            shim.touch()
            with self.assertRaisesRegex(BuildError, "configured shim path"):
                check_host(firmware)
            with (
                patch("cvm.host.preflight.subprocess.run", return_value=type("Result", (), {"returncode": 0})()),
                patch("cvm.host.preflight.check_qgs_config"),
                patch.object(
                    Path,
                    "read_text",
                    autospec=True,
                    side_effect=lambda path: "nohibernate" if str(path) == "/proc/cmdline" else "Y",
                ),
                patch.object(Path, "is_file", autospec=True, return_value=True),
            ):
                self.assertFalse(check_host(firmware, shim=shim))

    def test_tdx_gpu_preflight_runs_without_quote_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            firmware = Path(directory) / "OVMF.inteltdx.fd"
            firmware.touch()
            with (
                patch("cvm.host.preflight.subprocess.run", return_value=type("Result", (), {"returncode": 0})()),
                patch("cvm.host.preflight.check_qgs_config"),
                patch("cvm.host.preflight.check_gpu_host") as check_gpu,
                patch.object(
                    Path,
                    "read_text",
                    autospec=True,
                    side_effect=lambda path: "nohibernate" if str(path) == "/proc/cmdline" else "Y",
                ),
                patch.object(Path, "is_file", autospec=True, return_value=True),
            ):
                self.assertFalse(check_host(firmware, gpu=True))
            check_gpu.assert_called_once_with()

    def test_headerless_and_sectioned_qgs_configs(self):
        for content in (
            "# Intel QGS\nport = 4050\nnumber_threads = 4\n",
            "[server]\nport = 4050 # vsock\n",
            "[DEFAULT]\nport = 4050\n",
        ):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "qgs.conf"
                path.write_text(content)
                check_qgs_config(path)

    def test_missing_malformed_commented_and_wrong_ports_fail_closed(self):
        for content in (None, "", "# port = 4050\n", "port=4051\n", "port=4050\nport=4051\n", "[broken\nport=4050"):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "qgs.conf"
                if content is not None:
                    path.write_text(content)
                with self.assertRaises(BuildError):
                    check_qgs_config(path)

    def test_snp_offline_store_uses_exact_hwid_and_tcb(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preferred, legacy = snp_vcek_paths(root, self.VCEK_URL)
            self.assertEqual(preferred.name, "bl02_tee00_snp06_ucode33_vcek.der")
            preferred.parent.mkdir(parents=True)
            preferred.write_bytes(self.certificate())
            self.assertEqual(check_snp_vcek_store(root, self.VCEK_URL), preferred)
            preferred.write_text("not a certificate")
            with self.assertRaisesRegex(BuildError, "DER X.509"):
                check_snp_vcek_store(root, self.VCEK_URL)

    def test_snp_offline_store_rejects_non_ascii_tcb_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            url = self.VCEK_URL.replace("blSPL=2", "blSPL=²")
            with self.assertRaisesRegex(BuildError, "valid blSPL"):
                snp_vcek_paths(directory, url)

    def test_snp_preflight_can_require_offline_store(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            procfs, sysfs, devfs = root / "proc", root / "sys", root / "dev"
            procfs.mkdir()
            (sysfs / "module/kvm_amd/parameters").mkdir(parents=True)
            (sysfs / "kernel/iommu_groups/1").mkdir(parents=True)
            devfs.mkdir()
            firmware = root / "OVMF.amdsev.fd"
            firmware.touch()
            (procfs / "cpuinfo").write_text("vendor_id : AuthenticAMD\nflags : sev sev_es sev_snp\n")
            (sysfs / "module/kvm_amd/parameters/sev_snp").write_text("Y\n")
            (devfs / "sev").touch()
            result = type("Result", (), {"returncode": 0, "stdout": self.VCEK_URL})()
            with (
                patch("cvm.host.preflight.subprocess.run", return_value=result),
                patch("cvm.host.preflight.check_snp_vcek_store") as check,
            ):
                check_snp_host(
                    firmware,
                    vcek_store=root / "vcek",
                    procfs=procfs,
                    sysfs=sysfs,
                    devfs=devfs,
                )
            check.assert_called_once_with(root / "vcek", self.VCEK_URL)

    def test_snp_preflight_rejects_turin_until_trustee_exposes_fmc(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            procfs, sysfs, devfs = root / "proc", root / "sys", root / "dev"
            procfs.mkdir()
            (sysfs / "module/kvm_amd/parameters").mkdir(parents=True)
            (sysfs / "kernel/iommu_groups/1").mkdir(parents=True)
            devfs.mkdir()
            firmware = root / "OVMF.amdsev.fd"
            firmware.touch()
            (procfs / "cpuinfo").write_text("vendor_id : AuthenticAMD\ncpu family : 26\nflags : sev sev_es sev_snp\n")
            (sysfs / "module/kvm_amd/parameters/sev_snp").write_text("Y\n")
            (devfs / "sev").touch()
            with self.assertRaisesRegex(BuildError, "Turin.*FMC"):
                check_snp_host(firmware, procfs=procfs, sysfs=sysfs, devfs=devfs)
