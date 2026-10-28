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

"""Check host TDX prerequisites before constructing an application image."""

import configparser
import os
import subprocess
from pathlib import Path

from ..common.errors import BuildError, require

SYSFS = Path("/sys")
PROCFS = Path("/proc")
DEVFS = Path("/dev")


def check_qgs_config(path=Path("/etc/qgs.conf")):
    """Accept Intel's headerless file and sectioned distribution variants."""
    qgs = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#", ";"))
    try:
        content = Path(path).read_text()
        try:
            qgs.read_string(content)
        except configparser.MissingSectionHeaderError:
            qgs.read_string("[DEFAULT]\n" + content)
    except (OSError, configparser.Error) as error:
        raise BuildError(f"Cannot read QGS configuration ({type(error).__name__})") from None
    require(
        any(section.get("port", "").strip() == "4050" for section in qgs.values()),
        "Set QGS port=4050 explicitly; the commented default selects a Unix socket",
    )


def _enabled(path):
    return Path(path).is_file() and Path(path).read_text().strip().lower() in ("y", "1", "yes")


def check_gpu_host(sysfs=SYSFS):
    """Screen an installed NVIDIA confidential-GPU host before sealing work."""
    devices = []
    for device in sorted((Path(sysfs) / "bus/pci/devices").glob("*")):
        try:
            vendor = (device / "vendor").read_text().strip().lower()
            class_code = int((device / "class").read_text().strip(), 16) >> 16
            if vendor == "0x10de" and class_code == 0x03 and (device / "iommu_group").exists():
                devices.append(device.name)
        except (OSError, ValueError):
            continue
    require(devices, "No NVIDIA display GPU with an IOMMU group was detected")
    try:
        inventory = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,vbios_version,driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        raise BuildError("Install a supported NVIDIA driver and nvidia-smi before GPU qualification") from None
    require(
        inventory.returncode == 0, "Install a supported NVIDIA driver and verify GPU/VBIOS inventory with nvidia-smi"
    )
    try:
        cc = subprocess.run(
            ["nvidia-smi", "conf-compute", "-q"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        raise BuildError("nvidia-smi disappeared during GPU qualification") from None
    require(cc.returncode == 0, "GPU does not expose NVIDIA confidential-computing status through nvidia-smi")
    print(f"GPU host screening passed for {len(devices)} assignable NVIDIA display device(s).")


def check_snp_host(firmware, *, gpu=False, procfs=PROCFS, sysfs=SYSFS, devfs=DEVFS):
    """Check SEV-SNP firmware, CPU, KVM, device, IOMMU and optional GPU gates."""
    firmware = Path(firmware)
    require(firmware.is_file(), "SNP OVMF firmware is missing")
    cpuinfo = (Path(procfs) / "cpuinfo").read_text()
    flags = set(cpuinfo.replace(":", " ").split())
    require("AuthenticAMD" in cpuinfo and "sev_snp" in flags, "Host CPU does not advertise AMD SEV-SNP")
    require(
        _enabled(Path(sysfs) / "module/kvm_amd/parameters/sev_snp"),
        "Enable SEV-SNP in BIOS and kvm_amd, then reboot and verify kvm_amd.sev_snp=Y",
    )
    require((Path(devfs) / "sev").exists(), "Host /dev/sev is missing; verify BIOS SEV-SNP and the kvm_amd driver")
    require(any((Path(sysfs) / "kernel/iommu_groups").glob("*")), "IOMMU groups are missing; enable AMD IOMMU in BIOS")
    if gpu:
        check_gpu_host(sysfs)
    print("AMD SEV-SNP host prerequisites passed.")
    return True


def check_host(firmware, quote_probe=None, *, platform="intel_tdx", gpu=False):
    """Check platform prerequisites; return whether attestation was exercised."""
    require(platform in ("intel_tdx", "amd_sev_snp"), "Unsupported host preflight platform")
    if platform == "amd_sev_snp":
        require(quote_probe is None, "SNP host preflight does not accept a TDX quote probe")
        return check_snp_host(firmware, gpu=gpu)
    firmware = Path(firmware)
    require(firmware.is_file(), "TDVF firmware is missing")
    require(
        not firmware.name.endswith(".ms.fd"),
        "Secure Boot .ms.fd needs a separately validated signed shim/kernel path; use the documented direct-boot TDVF",
    )
    cmdline = Path("/proc/cmdline").read_text().split()
    require("nohibernate" in cmdline, "Add nohibernate to the host kernel command line and reboot")
    enabled = Path("/sys/module/kvm_intel/parameters/tdx")
    require(
        enabled.is_file() and enabled.read_text().strip().lower() in ("y", "1"),
        "Enable kvm_intel.tdx=1 (or options kvm_intel tdx=1), reload safely or reboot",
    )
    require(
        subprocess.run(["systemctl", "is-active", "--quiet", "qgsd"], check=False).returncode == 0,
        "Install tdx-qgs and start qgsd",
    )
    check_qgs_config()
    require(
        Path("/etc/sgx_default_qcnl.conf").is_file(),
        "Configure the approved PCS/collateral service in sgx_default_qcnl.conf",
    )
    print("Host prerequisites passed; absence of early TDX dmesg is not an error (module initialization can be lazy).")
    if quote_probe is None:
        return False
    probe = Path(quote_probe).resolve()
    require(probe.is_file() and os.access(probe, os.X_OK), "Quote probe must be an executable")
    result = subprocess.run([str(probe)], timeout=180, check=False)
    require(
        result.returncode == 0,
        "Quote probe failed: inspect qgsd journal, collateral service credentials and platform registration before building",
    )
    print("Quote probe passed.")
    if gpu:
        check_gpu_host()
    return True
