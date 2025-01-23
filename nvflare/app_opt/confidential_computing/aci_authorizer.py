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
import time

import jwt
import requests
from jwt import PyJWKClient

from nvflare.app_opt.confidential_computing.cc_authorizer import CCAuthorizer

ACI_NAMESPACE = "x-ms"
maa_endpoint = "sharedeus2.eus2.attest.azure.net"


class ACIAuthorizer(CCAuthorizer):
    def __init__(self, retry_count=5, retry_sleep=2):
        self.retry_count = retry_count
        self.retry_sleep = retry_sleep

    def generate(self):
        count = 0
        token = ""
        while True:
            count = count + 1
            try:
                r = requests.post(
                    "http://localhost:8284/attest/maa",
                    data=json.dumps({"maa_endpoint": maa_endpoint, "runtime_data": "ewp9"}),
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
            jwks_client = PyJWKClient(f"https://{maa_endpoint}/certs")
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            claims = jwt.decode(token, signing_key.key, algorithms=[alg])
            if claims:
                return True
        except:
            return False
        return False

    def get_namespace(self) -> str:
        return ACI_NAMESPACE
