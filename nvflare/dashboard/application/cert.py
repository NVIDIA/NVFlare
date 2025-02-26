# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from dataclasses import dataclass

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

from nvflare.lighter.utils import Identity, generate_cert, generate_keys, serialize_cert, serialize_pri_key

dashboard_pp = os.environ.get("NVFL_DASHBOARD_PP")
if dashboard_pp is not None:
    dashboard_pp = dashboard_pp.encode("utf-8")


@dataclass
class Entity:
    """Class for keeping track of each certificate owner."""

    name: str
    org: str = None
    role: str = None


@dataclass
class CertPair:
    """Class for serialized private key and certificate."""

    owner: Entity = None
    ser_pri_key: str = None
    ser_cert: str = None


def _pack(entity, pri_key, cert, passphrase=None):
    ser_pri_key = serialize_pri_key(pri_key, passphrase)
    ser_cert = serialize_cert(cert)
    cert_pair = CertPair(entity, ser_pri_key, ser_cert)
    return cert_pair


def make_root_cert(subject: Entity):
    pri_key, pub_key = generate_keys()
    cert = generate_cert(
        subject=Identity(subject.name, subject.org, subject.role),
        issuer=Identity(subject.name, subject.org, subject.role),
        signing_pri_key=pri_key,
        subject_pub_key=pub_key,
        ca=True,
    )
    return _pack(subject, pri_key, cert, passphrase=dashboard_pp)


def deserialize_ca_key(ser_pri_key):
    pri_key = serialization.load_pem_private_key(ser_pri_key, password=dashboard_pp, backend=default_backend())
    return pri_key
