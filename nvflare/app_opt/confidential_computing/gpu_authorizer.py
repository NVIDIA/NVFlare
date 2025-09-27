# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


import json
import logging
import uuid

import jwt
from nv_attestation_sdk import attestation

from nvflare.app_opt.confidential_computing.cc_authorizer import CCAuthorizer

from .utils import NonceHistory

GPU_NAMESPACE = "x-nv-gpu"
default_policy = """{
  "version":"4.0",
  "authorization-rules":{
    "type": "JWT",
    "overall-claims": {
      "x-nvidia-overall-att-result": true,
      "x-nvidia-ver": "3.0"
    },
    "detached-claims":{
      "measres": "success",
      "x-nvidia-gpu-arch-check": true,
      "x-nvidia-gpu-attestation-report-parsed": true,
      "x-nvidia-gpu-attestation-report-nonce-match": true,
      "x-nvidia-gpu-attestation-report-signature-verified": true,
      "x-nvidia-gpu-attestation-report-cert-chain":
      {
        "x-nvidia-cert-status": "valid",
        "x-nvidia-cert-ocsp-status": "good"
      },
      "x-nvidia-gpu-attestation-report-cert-chain-fwid-match": true,
      "x-nvidia-gpu-driver-rim-fetched": true,
      "x-nvidia-gpu-driver-rim-schema-validated": true,
      "x-nvidia-gpu-driver-rim-signature-verified": true,
      "x-nvidia-gpu-driver-rim-version-match": true,
      "x-nvidia-gpu-driver-rim-cert-chain":
      {
        "x-nvidia-cert-status": "valid",
        "x-nvidia-cert-ocsp-status": "good"
      },
      "x-nvidia-gpu-driver-rim-measurements-available": true,
      "x-nvidia-gpu-vbios-rim-fetched": true,
      "x-nvidia-gpu-vbios-rim-schema-validated": true,
      "x-nvidia-gpu-vbios-rim-signature-verified": true,
      "x-nvidia-gpu-vbios-rim-version-match": true,
      "x-nvidia-gpu-vbios-rim-cert-chain":
      {
        "x-nvidia-cert-status": "valid",
        "x-nvidia-cert-ocsp-status": "good"
      },
      "x-nvidia-gpu-vbios-rim-measurements-available": true,
      "x-nvidia-gpu-vbios-index-no-conflict": true
    }
  }
}
"""


class GPUAuthorizer(CCAuthorizer):
    def __init__(
        self, verifier_url="https://nras.attestation.nvidia.com/v4/attest/gpu", policy_file=None, max_nonce_history=1000
    ):
        self._can_generate = True
        self.client = attestation.Attestation()
        self.client.set_name("nvflare_node")
        self.my_nonce_history = NonceHistory(max_nonce_history)
        self.seen_nonce_history = NonceHistory(max_nonce_history)
        self.client.set_claims_version("3.0")

        if policy_file is None:
            self.remote_att_result_policy = default_policy
        else:
            self.remote_att_result_policy = open(policy_file).read()
        self.client.add_verifier(attestation.Devices.GPU, attestation.Environment.REMOTE, verifier_url, "")
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate(self):
        try:
            nonce = uuid.uuid4().hex + uuid.uuid1().hex
            self.client.set_nonce(nonce)
            self.my_nonce_history.add(nonce)
            evidence_list = self.client.get_evidence()
            self.client.attest(evidence_list)
            token = self.client.get_token()
        except BaseException:
            self.can_generate = False
            token = "[[],{}]"
        return token

    def verify(self, eat_token):
        try:
            jwt_token = json.loads(eat_token)[1]
            remote_gpu_claims = jwt_token.get("REMOTE_GPU_CLAIMS")
            if (
                isinstance(remote_gpu_claims, list)
                and len(remote_gpu_claims) > 0
                and isinstance(remote_gpu_claims[0], list)
                and len(remote_gpu_claims[0]) > 1
            ):
                claims = jwt.decode(remote_gpu_claims[0][1], options={"verify_signature": False})
            else:
                self.logger.info("Invalid structure for REMOTE_GPU_CLAIMS")
                return False
            # With claims, we will retrieve the nonce
            nonce = claims.get("eat_nonce")
            if not self.seen_nonce_history.add(nonce):
                return False
            self.client.set_nonce(nonce)
            self.client.set_token(name="nvflare_node", eat_token=eat_token)
            result = self.client.validate_token(self.remote_att_result_policy)
        except BaseException as e:
            self.logger.info(f"Token verification failed {e=}")
            result = False
        return result

    def get_namespace(self) -> str:
        return GPU_NAMESPACE
