#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


from nv_attestation_sdk import attestation
import os
import json
import sys


def run_attestation():
    NRAS_URL = "https://nras.attestation-stg.nvidia.com/v4/attest/gpu"
    client = attestation.Attestation()
    client.set_name("AttestationServiceNode")
    client.set_nonce("931d8dd0add203ac3d8b4fbde75e115278eefcdceac5b87671a748f32364dfcb")
    client.set_claims_version("3.0")

    print("[AttestationService] Node name:", client.get_name())

    client.add_verifier(attestation.Devices.GPU, attestation.Environment.REMOTE, NRAS_URL, "")
    print("[AttestationService] Verifiers:", client.get_verifiers())

    print("[AttestationService] Calling get_evidence()...")
    evidence_list = client.get_evidence()

    print("[AttestationService] Calling attest() - expecting True")
    if not client.attest(evidence_list):
        print("[AttestationService] Attestation failed")
        return None

    token = client.get_token()
    print("[AttestationService] Attestation succeeded")
    print("[AttestationService] Token:", token)

    return client


def validate_attestation_token(client):
    print("[AttestationService] Validating token against policy...")

    policy_file = (
        "/home/nvidia/nvtrust/guest_tools/attestation_sdk/tests/policies/remote/v4/NVGPURemotePolicyExample.json"
    )
    try:
        with open(policy_file) as json_file:
            json_data = json.load(json_file)
            remote_att_result_policy = json.dumps(json_data)

        valid = client.validate_token(remote_att_result_policy)
        if valid:
            print("[AttestationService] Token validation passed")
        else:
            print("[AttestationService] Token validation failed")
        return valid

    except Exception as e:
        print(f"[AttestationService] Policy validation error: {e}")
        return False


def main():
    client = run_attestation()
    if not client:
        sys.exit("[AttestationService] Shutting down due to attestation failure")

    if not validate_attestation_token(client):
        sys.exit("[AttestationService] Token validation failed - shutting down")

    print("[AttestationService] Token validation complete and passed")


if __name__ == "__main__":
    main()
