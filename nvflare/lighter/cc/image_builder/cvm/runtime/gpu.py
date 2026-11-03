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

"""Publish GPU readiness only after successful attestation."""

import re

from ..common.errors import require
from ..common.linux import run


def readiness(config, ready):
    if config.get("gpu") != "nvidia_cc":
        return
    if ready:
        devices = run(["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"], timeout=10).splitlines()
        require(len(devices) == config["gpu_count"], "GPU count changed after composite appraisal")
    run(["nvidia-smi", "conf-compute", "-srs", "1" if ready else "0"], timeout=10)


def confirm_not_ready(config):
    """Require the expected GPU inventory and an explicit system Not Ready readback."""
    count = config.get("gpu_count")
    require(type(count) is int and count > 0, "Invalid expected GPU count")
    inventory = run(["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"], timeout=10)
    require(isinstance(inventory, bytes), "Invalid GPU inventory during readiness verification")
    devices = inventory.splitlines()
    require(len(devices) == count, "GPU count changed during readiness verification")
    addresses = set()
    for device in devices:
        match = re.fullmatch(rb"([0-9a-fA-F]{4}|[0-9a-fA-F]{8}):([0-9a-fA-F]{2}):([01][0-9a-fA-F])\.([0-7])", device)
        require(match is not None, "Invalid GPU inventory during readiness verification")
        address = tuple(int(part, 16) for part in match.groups())
        require(address not in addresses, "Duplicate GPU inventory during readiness verification")
        addresses.add(address)
    # This is one system-wide state, rather than one readiness row per device.
    state = run(["nvidia-smi", "conf-compute", "-grs"], timeout=10)
    require(
        isinstance(state, bytes) and state.strip() == b"Confidential Compute GPUs Ready state: not-ready",
        "GPU Not Ready state is unconfirmed",
    )
