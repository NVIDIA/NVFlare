# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

import os
import random
from base64 import b64encode

import yaml
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding


def generate_password():
    s = "abcdefghijklmnopqrstuvwxyz01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    passlen = 16
    p = "".join(random.sample(s, passlen))
    return p


def sign_all(content_folder, signing_pri_key):
    signatures = dict()
    for f in os.listdir(content_folder):
        path = os.path.join(content_folder, f)
        if os.path.isfile(path):
            signature = signing_pri_key.sign(
                data=open(path, "rb").read(),
                padding=padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                algorithm=hashes.SHA256(),
            )
            signatures[f] = b64encode(signature).decode("utf-8")
    return signatures


def load_yaml(file):
    if isinstance(file, str):
        return yaml.safe_load(open(file, "r"))
    elif isinstance(file, bytes):
        return yaml.safe_load(file)
    else:
        return None


def sh_replace(src, mapping_dict):
    result = src
    for k, v in mapping_dict.items():
        result = result.replace("{~~" + k + "~~}", str(v))
    return result
