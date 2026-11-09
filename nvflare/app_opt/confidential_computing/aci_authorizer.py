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
import re
import time

import jwt
import requests

from nvflare.app_opt.confidential_computing.cc_authorizer import CCAuthorizer

ACI_NAMESPACE = "x-az-aci"
POLICY_HASH_PATTERN = re.compile(r"[0-9a-fA-F]{64}")


class ACIAuthorizer(CCAuthorizer):
    def __init__(
        self,
        maa_endpoint="sharedeus2.eus2.attest.azure.net",
        retry_count=5,
        retry_sleep=2,
        namespace=ACI_NAMESPACE,
        *,
        expected_policy_hash=None,
    ):
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("namespace must be a non-empty string")
        if not isinstance(expected_policy_hash, str) or not POLICY_HASH_PATTERN.fullmatch(expected_policy_hash):
            raise ValueError(
                "expected_policy_hash must be the approved CCE policy's 64-character SHA-256 hexadecimal hash"
            )
        self.maa_endpoint = maa_endpoint
        self.retry_count = retry_count
        self.retry_sleep = retry_sleep
        self.namespace = namespace
        self.expected_policy_hash = expected_policy_hash.lower()

    def generate(self):
        count = 0
        token = ""
        while True:
            count = count + 1
            try:
                r = requests.post(
                    "http://localhost:8284/attest/maa",
                    data=json.dumps({"maa_endpoint": self.maa_endpoint, "runtime_data": "ewp9"}),
                    headers={"Content-Type": "application/json"},
                )
                if r.status_code == requests.codes.ok:
                    token = r.json().get("token")
                break
            except:
                if count > self.retry_count:
                    break
                time.sleep(self.retry_sleep)
        return token

    def verify(self, token):
        try:
            header = jwt.get_unverified_header(token)
            alg = header.get("alg")
            jwks_client = jwt.PyJWKClient(f"https://{self.maa_endpoint}/certs")
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            claims = jwt.decode(token, signing_key.key, algorithms=[alg])
            policy_hash = claims.get("x-ms-sevsnpvm-hostdata")
            return (
                isinstance(policy_hash, str)
                and POLICY_HASH_PATTERN.fullmatch(policy_hash) is not None
                and policy_hash.lower() == self.expected_policy_hash
            )
        except:
            return False
        return False

    def get_namespace(self) -> str:
        return self.namespace
