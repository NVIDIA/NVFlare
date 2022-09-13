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

import datetime
import os
from dataclasses import dataclass

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

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


def serialize_pri_key(pri_key, passphrase=None):
    if passphrase is None or not isinstance(passphrase, bytes):
        return pri_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    else:
        return pri_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.BestAvailableEncryption(password=passphrase),
        )


def serialize_cert(cert):
    return cert.public_bytes(serialization.Encoding.PEM)


def generate_keys():
    pri_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
    pub_key = pri_key.public_key()
    return pri_key, pub_key


def x509_name(cn_name, org_name=None, role=None):
    name = [x509.NameAttribute(NameOID.COMMON_NAME, cn_name)]
    if org_name is not None:
        name.append(x509.NameAttribute(NameOID.ORGANIZATION_NAME, org_name))
    if role:
        name.append(x509.NameAttribute(NameOID.UNSTRUCTURED_NAME, role))
    return x509.Name(name)


def generate_cert(subject, issuer, signing_pri_key, subject_pub_key, valid_days=360, ca=False):
    x509_subject = x509_name(subject.name, subject.org, subject.role)
    x509_issuer = x509_name(issuer.name, issuer.org, issuer.role)
    builder = (
        x509.CertificateBuilder()
        .subject_name(x509_subject)
        .issuer_name(x509_issuer)
        .public_key(subject_pub_key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(
            # Our certificate will be valid for 360 days
            datetime.datetime.utcnow()
            + datetime.timedelta(days=valid_days)
            # Sign our certificate with our private key
        )
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(subject.name)]), critical=False)
    )
    if ca:
        builder = (
            builder.add_extension(
                x509.SubjectKeyIdentifier.from_public_key(subject_pub_key),
                critical=False,
            )
            .add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_public_key(subject_pub_key),
                critical=False,
            )
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=False)
        )
    return builder.sign(signing_pri_key, hashes.SHA256(), default_backend())


def _pack(entity, pri_key, cert, passphrase=None):
    ser_pri_key = serialize_pri_key(pri_key, passphrase)
    ser_cert = serialize_cert(cert)
    cert_pair = CertPair(entity, ser_pri_key, ser_cert)
    return cert_pair


def make_root_cert(subject: Entity):
    pri_key, pub_key = generate_keys()
    cert = generate_cert(subject=subject, issuer=subject, signing_pri_key=pri_key, subject_pub_key=pub_key, ca=True)
    return _pack(subject, pri_key, cert, passphrase=dashboard_pp)


def make_cert(subject: Entity, issuer_cert_pair: CertPair):
    pri_key, pub_key = generate_keys()
    issuer_pri_key = deserialize_ca_key(issuer_cert_pair.ser_pri_key)
    cert = generate_cert(subject, issuer_cert_pair.owner, issuer_pri_key, pub_key, valid_days=360, ca=False)
    return _pack(subject, pri_key, cert, passphrase=None)


def deserialize_ca_key(ser_pri_key):
    pri_key = serialization.load_pem_private_key(ser_pri_key, password=dashboard_pp, backend=default_backend())
    return pri_key
