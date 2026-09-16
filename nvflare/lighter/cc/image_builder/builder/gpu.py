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

"""Shared NVIDIA GPU appraisal, with policy anchored in the measured root."""

import json
import os

from .common import BuildError, read_json, require, run


def validate_gpu_count(config):
    devices = [
        line for line in run(["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"]).splitlines() if line
    ]
    require(len(devices) == config["gpu_count"], "Guest GPU count does not match the measured profile")
    return devices


def policy_mismatch(expected, actual, path="required-claims"):
    """Return the first failed policy path without logging device evidence."""
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return path
    for key, value in expected.items():
        child = f"{path}.{key}"
        if key not in actual:
            return child
        if isinstance(value, dict):
            mismatch = policy_mismatch(value, actual[key], child)
            if mismatch:
                return mismatch
        elif type(actual[key]) is not type(value) or actual[key] != value:
            return child
    return None


def validate_result(result, policy, nonce, gpu_count):
    require(isinstance(result, dict) and result.get("result_code") == 0, "GPU remote appraisal failed")
    require(result.get("detached_eat"), "GPU appraisal did not return signed evidence")
    claims = result.get("claims")
    require(isinstance(claims, list) and len(claims) == gpu_count, "GPU appraisal result count mismatch")
    required = policy.get("required-claims")
    for index, claim_set in enumerate(claims):
        mismatch = policy_mismatch(required, claim_set)
        require(mismatch is None, f"GPU policy denied for device {index} at {mismatch}")
        conditional = {key: value for key, value in policy["claims-if-present"].items() if key in claim_set}
        mismatch = policy_mismatch(conditional, claim_set, "claims-if-present")
        require(mismatch is None, f"GPU policy denied for device {index} at {mismatch}")
        reported_nonce = claim_set.get("eat_nonce")
        require(
            isinstance(reported_nonce, str) and reported_nonce.lower() == nonce.lower(),
            "GPU appraisal nonce mismatch",
        )


def attest(config):
    validate_gpu_count(config)
    nonce = os.urandom(32).hex()
    output = run(
        [
            "/usr/bin/nvattest",
            "--log-level",
            "off",
            "--format",
            "json",
            "attest",
            "--nonce",
            nonce,
            "--device",
            "gpu",
            "--verifier",
            "remote",
            "--nras-url",
            config["gpu_attestation_url"],
        ],
        timeout=180,
    )
    try:
        result = json.loads(output)
    except (TypeError, ValueError):
        raise ValueError("GPU attestation returned invalid JSON") from None
    validate_result(result, read_json(config["gpu_policy"]), nonce, config["gpu_count"])


def main():
    config = read_json("/etc/cvm/runtime.json")
    require(config["gpu"] == "nvidia_cc", "GPU attestation requires a GPU profile")
    try:
        attest(config)
        # NVAT returns appraisal results without enabling CUDA work. Only the
        # measured policy gate may mark the GPUs ready after all checks pass.
        run(["nvidia-smi", "conf-compute", "-srs", "1"], timeout=10)
    except Exception:
        # Revoke readiness on a failed periodic appraisal before the caller
        # stops the workload and powers off. Preserve the original failure if
        # the driver also refuses the readiness reset.
        try:
            run(["nvidia-smi", "conf-compute", "-srs", "0"], timeout=10)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    try:
        main()
    except BuildError as error:
        # BuildError messages contain only deliberately sanitized diagnostics;
        # never print raw NVAT output, tokens, nonces, or device evidence.
        raise SystemExit(f"GPU appraisal failed: {error}") from None
    except Exception:
        raise SystemExit("GPU appraisal failed: invalid appraisal response") from None
