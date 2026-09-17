#!/usr/bin/env python3
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

"""Check signed NRAS claims through the real Rust verifier and generated policy."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from builder.gpu_policy import render

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("trustee", type=Path, help="Checkout with the current boundary patch applied")
parser.add_argument("--policy-eval", type=Path, required=True, help="Built tests/policy_engine executable")
args = parser.parse_args()
source = args.trustee.resolve()
if (source / "deps/verifier/src/nvidia.rs").read_bytes() != (ROOT / "trustee/nvidia_verifier.rs").read_bytes():
    parser.error("Trustee does not contain the current NVIDIA verifier")
with tempfile.TemporaryDirectory(prefix="cvm-nras-policy-") as directory:
    work = Path(directory)
    (work / "device.json").write_bytes((ROOT / "tests/fixtures/nras_gpu_v3.json").read_bytes())
    (work / "policy.rego").write_text(render(json.loads((ROOT / "config/gpu_policy.json").read_text())))
    (work / "data.json").write_text(
        json.dumps({"reference": {"gpu_driver_versions": ["575.28"], "gpu_vbios_versions": ["96.00.AF.00.01"]}})
    )
    subprocess.run(
        [
            "cargo",
            "test",
            "--locked",
            "--release",
            "-p",
            "verifier",
            "--no-default-features",
            "--features",
            "nvidia-verifier",
            "nvidia::tests::signed_v3_device_claims_reach_generated_policy",
            "--",
            "--ignored",
            "--exact",
        ],
        cwd=source,
        env=dict(os.environ, CVM_NRAS_POLICY_TEST=str(work), CVM_POLICY_EVAL=str(args.policy_eval.resolve())),
        check=True,
    )
