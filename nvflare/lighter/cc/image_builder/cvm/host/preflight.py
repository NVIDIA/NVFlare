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

import argparse
import configparser
import os
import subprocess
from pathlib import Path

from ..common.errors import BuildError, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firmware", type=Path, required=True)
    parser.add_argument(
        "--quote-probe",
        type=Path,
        help="Site executable that boots a minimal TD and verifies a nonempty quote through the intended backend",
    )
    args = parser.parse_args()
    try:
        require(args.firmware.is_file(), "TDVF firmware is missing")
        require(
            not args.firmware.name.endswith(".ms.fd"),
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
        qgs = configparser.ConfigParser()
        require(qgs.read("/etc/qgs.conf"), "QGS configuration is missing")
        require(
            any(section.get("port", "").strip() == "4050" for section in qgs.values()),
            "Set QGS port=4050 explicitly; the commented default selects a Unix socket",
        )
        require(
            Path("/etc/sgx_default_qcnl.conf").is_file(),
            "Configure the approved PCS/collateral service in sgx_default_qcnl.conf",
        )
        print(
            "Host prerequisites passed; absence of early TDX dmesg is not an error (module initialization can be lazy)."
        )
        if args.quote_probe is None:
            parser.exit(
                2,
                "Quote generation NOT checked. Run --quote-probe with a site smoke test before building. Check collateral access and multi-package platform registration if quotes are empty.\n",
            )
        probe = args.quote_probe.resolve()
        require(probe.is_file() and os.access(probe, os.X_OK), "Quote probe must be an executable")
        result = subprocess.run([str(probe)], timeout=180, check=False)
        require(
            result.returncode == 0,
            "Quote probe failed: inspect qgsd journal, collateral service credentials and platform registration before building",
        )
        print("Quote probe passed.")
    except (BuildError, OSError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"TDX preflight failed: {error}\n")


if __name__ == "__main__":
    main()
