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

"""Coordinate a real periodic-denial check with an isolated backend operator.

Use the CVM_HARDWARE_* environment from test_hardware.py. The operator waits for
ready.json, changes only the disposable backend authorization, then writes the
proceed file. This script never holds backend credentials.
"""

import json
import os
import time
from pathlib import Path

from builder.common import require, write_json
from builder.storage import mounted, nbd
from test_hardware import HardwareTests


def main():
    require(os.environ.get("CVM_HARDWARE_TESTS") == "1", "Explicit hardware test opt-in required")
    HardwareTests.setUpClass()
    case = HardwareTests("test_generic_app_binding_and_exclusive_attachment")
    case.setUp()
    output = Path(os.environ["CVM_HARDWARE_OUTPUT"])
    try:
        case.boot()
        case.ready()
        write_json(output / "ready.json", {"ready": True})
        print("Guest ready for the backend denial test.", flush=True)
        deadline = time.monotonic() + 180
        while not (output / "proceed").exists():
            require(time.monotonic() < deadline, "Backend operator did not signal the denial change")
            require(case.process.poll() is None, "Guest exited before the backend change")
            time.sleep(1)
        case.request("/periodic", "POST")
        case.process.wait(timeout=120)
        require("Power down" in case.logs[-1].read_text(errors="replace"), "Periodic denial did not power off")
        with nbd(case.vault / "applog.qcow2", readonly=True) as device, mounted(device, readonly=True) as root:
            records = [json.loads(line) for line in (root / "attestation.log").read_text().splitlines()]
        require(records[0]["decision"] == "allow" and records[-1]["decision"] == "deny", "Missing appraisal audit")
        fields = {"ts", "cvm_build_id", "vault_id", "measurement", "policy_id", "decision"}
        require(all(set(record) == fields for record in records), "Unexpected audit data")
        require(
            all(record["measurement"] == case.manifest["measurements"] for record in records),
            "Audit measurement mismatch",
        )
        case.result(periodic_denial_powered_off=True, audit_schema_verified=True, audit_records=records)
        print("Periodic denial and audit records passed.")
    finally:
        case.cleanup()


if __name__ == "__main__":
    main()
