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
import subprocess

import jwt

from nvflare.app_opt.confidential_computing.cc_authorizer import CCAuthorizer

AZ_CVM_NAMESPACE = "x-az-cvm"


class AZCVMAuthorizer(CCAuthorizer):
    def __init__(self, attester_binary="AttestationClient", maa_endpoint="sharedeus2.eus2.attest.azure.net"):
        self.attester_binary = attester_binary
        self.maa_endpoint = maa_endpoint

    def generate(self):
        cmd = ["sudo", self.attester_binary, "-a", f"https://{self.maa_endpoint}/", "-o", "token"]
        result = subprocess.run(cmd, capture_output=True, check=False)
        if result.returncode != 0:
            return ""
        token = result.stdout.decode().strip()
        return token

    def verify(self, token):
        try:
            header = jwt.get_unverified_header(token)
            alg = header.get("alg")
            jwks_client = jwt.PyJWKClient(f"https://{self.maa_endpoint}/certs")
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            claims = jwt.decode(token, signing_key.key, algorithms=[alg])
            if claims:
                return True
        except (jwt.PyJWTError, Exception):
            return False
        return False

    def get_namespace(self) -> str:
        return AZ_CVM_NAMESPACE
