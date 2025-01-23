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

GPU_NAMESPACE = "x-nv-gpu"
default_policy = """{
  "version":"1.0",
  "authorization-rules":{
    "sub":"NVIDIA-GPU-ATTESTATION",
    "secboot":true,
    "x-nvidia-gpu-manufacturer":"NVIDIA Corporation",
    "x-nvidia-attestation-type":"GPU",
    "x-nvidia-attestation-detailed-result":{
      "x-nvidia-gpu-driver-rim-schema-validated":true,
      "x-nvidia-gpu-vbios-rim-cert-validated":true,
      "x-nvidia-gpu-attestation-report-cert-chain-validated":true,
      "x-nvidia-gpu-driver-rim-schema-fetched":true,
      "x-nvidia-gpu-attestation-report-parsed":true,
      "x-nvidia-gpu-nonce-match":true,
      "x-nvidia-gpu-vbios-rim-signature-verified":true,
      "x-nvidia-gpu-driver-rim-signature-verified":true,
      "x-nvidia-gpu-arch-check":true,
      "x-nvidia-gpu-measurements-match":true,
      "x-nvidia-gpu-attestation-report-signature-verified":true,
      "x-nvidia-gpu-vbios-rim-schema-validated":true,
      "x-nvidia-gpu-driver-rim-cert-validated":true,
      "x-nvidia-gpu-vbios-rim-schema-fetched":true,
      "x-nvidia-gpu-vbios-rim-measurements-available":true
    },
    "x-nvidia-gpu-driver-version":"535.104.05",
    "hwmodel":"GH100 A01 GSP BROM",
    "measres":"comparison-successful",
    "x-nvidia-gpu-vbios-version":"96.00.5E.00.02"
  }
}
"""


class GPUAuthorizer(CCAuthorizer):
    def __init__(self, verifier_url="https://nras.attestation.nvidia.com/v1/attest/gpu", policy_file=None):
        self._can_generate = True
        self.client = attestation.Attestation()
        self.client.set_name("nvflare_node")
        nonce = uuid.uuid4().hex + uuid.uuid1().hex
        self.client.set_nonce(nonce)
        if policy_file is None:
            self.remote_att_result_policy = default_policy
        else:
            self.remote_att_result_policy = open(policy_file).read()
        self.client.add_verifier(attestation.Devices.GPU, attestation.Environment.REMOTE, verifier_url, "")
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate(self):
        try:
            self.client.attest()
            token = self.client.get_token()
        except BaseException:
            self.can_generate = False
            token = "[[],{}]"
        return token

    def verify(self, eat_token):
        try:
            jwt_token = json.loads(eat_token)[1]
            claims = jwt.decode(jwt_token.get("REMOTE_GPU_CLAIMS"), options={"verify_signature": False})
            # With claims, we will retrieve the nonce
            nonce = claims.get("eat_nonce")
            self.client.set_nonce(nonce)
            self.client.set_token(name="nvflare_node", eat_token=eat_token)
            result = self.client.validate_token(self.remote_att_result_policy)
        except BaseException as e:
            self.logger.info(f"Token verification failed {e=}")
            result = False
        return result

    def get_namespace(self) -> str:
        return GPU_NAMESPACE
