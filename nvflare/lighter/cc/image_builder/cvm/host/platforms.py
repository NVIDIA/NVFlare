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

"""Detect host TEE capabilities and select a build platform."""

from pathlib import Path

from ..common.contracts import PLATFORMS
from ..common.errors import require


def host_capabilities(proc=Path("/proc"), sys=Path("/sys")):
    info = (proc / "cpuinfo").read_text() if (proc / "cpuinfo").exists() else ""
    result = set()
    for name, vendor, flag, module in [
        ("amd_sev_snp", "AuthenticAMD", "sev_snp", "kvm_amd/parameters/sev_snp"),
        ("intel_tdx", "GenuineIntel", "tdx", "kvm_intel/parameters/tdx"),
    ]:
        capability = sys / "module" / module
        enabled = capability.exists() and capability.read_text().strip().lower() in ("1", "y", "yes")
        flags = set(info.replace(":", " ").split())
        # A loaded KVM module explicitly disabling the feature overrides CPU
        # enumeration. A CPU flag alone cannot turn a disabled host into a TEE.
        available = enabled if capability.exists() else flag in flags
        if vendor in info and available:
            result.add(name)
    return result


def select_platform(profile, explicit=None, capabilities=None):
    enabled = {k for k, v in profile["platforms"].items() if v.get("enabled", True)}
    if explicit is not None:
        require(explicit in PLATFORMS and explicit in enabled, "-p must name a supported, enabled platform")
        return explicit
    found = (host_capabilities() if capabilities is None else capabilities) & enabled
    require(len(found) == 1, "Cannot unambiguously auto-detect an enabled SNP/TDX host; specify -p <platform>")
    return next(iter(found))
