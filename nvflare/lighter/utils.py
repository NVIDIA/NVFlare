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

import json
import os
import random
import shutil
from base64 import b64decode, b64encode

import yaml
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from nvflare.lighter.impl.cert import load_crt


def generate_password(passlen=16):
    s = "abcdefghijklmnopqrstuvwxyz01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    p = "".join(random.sample(s, passlen))
    return p


def sign_one(content, signing_pri_key):
    signature = signing_pri_key.sign(
        data=content,
        padding=padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        algorithm=hashes.SHA256(),
    )
    return b64encode(signature).decode("utf-8")


def load_private_key_file(file_path):
    with open(file_path, "rt") as f:
        pri_key = serialization.load_pem_private_key(f.read().encode("ascii"), password=None, backend=default_backend())
    return pri_key


def sign_folders(folder, signing_pri_key, crt_path):
    for root, folders, files in os.walk(folder):
        signatures = dict()
        for file in files:
            if file == ".__nvfl_sig.json":
                continue
            signature = signing_pri_key.sign(
                data=open(os.path.join(root, file), "rb").read(),
                padding=padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                algorithm=hashes.SHA256(),
            )
            signatures[file] = b64encode(signature).decode("utf-8")
        for folder in folders:
            signature = signing_pri_key.sign(
                data=folder.encode("utf-8"),
                padding=padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                algorithm=hashes.SHA256(),
            )
            signatures[folder] = b64encode(signature).decode("utf-8")
        json.dump(signatures, open(os.path.join(root, ".__nvfl_sig.json"), "wt"))
        shutil.copyfile(crt_path, os.path.join(root, ".__nvfl_submitter.crt"))


def verify_folder_signature(folder):
    try:
        for root, folders, files in os.walk(folder):
            try:
                signatures = json.load(open(os.path.join(root, ".__nvfl_sig.json"), "rt"))
                cert = load_crt(os.path.join(root, ".__nvfl_submitter.crt"))
                public_key = cert.public_key()
            except:
                continue
            for k in signatures:
                signatures[k] = b64decode(signatures[k].encode("utf-8"))
            for file in files:
                if file == ".__nvfl_sig.json" or file == ".__nvfl_submitter.crt":
                    continue
                signature = signatures.get(file)
                if signature:
                    public_key.verify(
                        signature=signature,
                        data=open(os.path.join(root, file), "rb").read(),
                        padding=padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                        algorithm=hashes.SHA256(),
                    )
            for folder in folders:
                signature = signatures.get(folder)
                if signature:
                    public_key.verify(
                        signature=signature,
                        data=folder.encode("utf-8"),
                        padding=padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                        algorithm=hashes.SHA256(),
                    )
        return True
    except BaseException as e:
        return False


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
