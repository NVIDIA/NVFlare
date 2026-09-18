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


def check_host(firmware, quote_probe=None):
    """Check host prerequisites; return whether a quote probe was also verified."""
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
    return True
