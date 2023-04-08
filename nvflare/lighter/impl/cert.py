# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
import json
import os

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from nvflare.lighter.spec import Builder


def serialize_pri_key(pri_key):
    return pri_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )


def serialize_cert(cert):
    return cert.public_bytes(serialization.Encoding.PEM)


def load_crt(path):
    serialized_cert = open(path, "rb").read()
    return x509.load_pem_x509_certificate(serialized_cert, default_backend())


class CertBuilder(Builder):
    def __init__(self):
        """Build certificate chain for every participant.

        Handles building (creating and self-signing) the root CA certificates, creating server, client and
        admin certificates, and having them signed by the root CA for secure communication. If the state folder has
        information about previously generated certs, it loads them back and reuses them.
        """
        self.root_cert = None
        self.persistent_state = dict()

    def initialize(self, ctx):
        state_dir = self.get_state_dir(ctx)
        cert_file = os.path.join(state_dir, "cert.json")
        if os.path.exists(cert_file):
            self.persistent_state = json.load(open(cert_file, "rt"))
            self.serialized_cert = self.persistent_state["root_cert"].encode("ascii")
            self.root_cert = x509.load_pem_x509_certificate(self.serialized_cert, default_backend())
            self.pri_key = serialization.load_pem_private_key(
                self.persistent_state["root_pri_key"].encode("ascii"), password=None, backend=default_backend()
            )
            self.pub_key = self.pri_key.public_key()
            self.subject = self.root_cert.subject
            self.issuer = self.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value

    def _build_root(self, subject, subject_org):
        if not self.persistent_state:
            pri_key, pub_key = self._generate_keys()
            self.issuer = subject
            self.root_cert = self._generate_cert(subject, subject_org, self.issuer, pri_key, pub_key, ca=True)
            self.pri_key = pri_key
            self.pub_key = pub_key
            self.serialized_cert = serialize_cert(self.root_cert)
            self.persistent_state["root_cert"] = self.serialized_cert.decode("ascii")
            self.persistent_state["root_pri_key"] = serialize_pri_key(self.pri_key).decode("ascii")

    def _build_write_cert_pair(self, participant, base_name, ctx):
        subject = participant.subject
        if self.persistent_state and subject in self.persistent_state:
            cert = x509.load_pem_x509_certificate(
                self.persistent_state[subject]["cert"].encode("ascii"), default_backend()
            )
            pri_key = serialization.load_pem_private_key(
                self.persistent_state[subject]["pri_key"].encode("ascii"), password=None, backend=default_backend()
            )
        else:
            pri_key, cert = self.get_pri_key_cert(participant)
            self.persistent_state[subject] = dict(
                cert=serialize_cert(cert).decode("ascii"), pri_key=serialize_pri_key(pri_key).decode("ascii")
            )
        dest_dir = self.get_kit_dir(participant, ctx)
        with open(os.path.join(dest_dir, f"{base_name}.crt"), "wb") as f:
            f.write(serialize_cert(cert))
        with open(os.path.join(dest_dir, f"{base_name}.key"), "wb") as f:
            f.write(serialize_pri_key(pri_key))
        pkcs12 = serialization.pkcs12.serialize_key_and_certificates(
            subject.encode("ascii"), pri_key, cert, None, serialization.BestAvailableEncryption(subject.encode("ascii"))
        )
        with open(os.path.join(dest_dir, f"{base_name}.pfx"), "wb") as f:
            f.write(pkcs12)
        with open(os.path.join(dest_dir, "rootCA.pem"), "wb") as f:
            f.write(self.serialized_cert)

    def build(self, project, ctx):
        self._build_root(project.name, subject_org=None)
        ctx["root_cert"] = self.root_cert
        ctx["root_pri_key"] = self.pri_key
        overseer = project.get_participants_by_type("overseer")
        if overseer:
            self._build_write_cert_pair(overseer, "overseer", ctx)

        servers = project.get_participants_by_type("server", first_only=False)
        for server in servers:
            self._build_write_cert_pair(server, "server", ctx)

        for client in project.get_participants_by_type("client", first_only=False):
            self._build_write_cert_pair(client, "client", ctx)

        for admin in project.get_participants_by_type("admin", first_only=False):
            self._build_write_cert_pair(admin, "client", ctx)

    def get_pri_key_cert(self, participant):
        pri_key, pub_key = self._generate_keys()
        subject = participant.subject
        subject_org = participant.org
        if participant.type == "admin":
            role = participant.props.get("role")
        else:
            role = None
        cert = self._generate_cert(subject, subject_org, self.issuer, self.pri_key, pub_key, role=role)
        return pri_key, cert

    def _generate_keys(self):
        pri_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
        pub_key = pri_key.public_key()
        return pri_key, pub_key

    def _generate_cert(
        self, subject, subject_org, issuer, signing_pri_key, subject_pub_key, valid_days=360, ca=False, role=None
    ):
        x509_subject = self._x509_name(subject, subject_org, role)
        x509_issuer = self._x509_name(issuer)
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
            .add_extension(x509.SubjectAlternativeName([x509.DNSName(subject)]), critical=False)
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

    def _x509_name(self, cn_name, org_name=None, role=None):
        name = [x509.NameAttribute(NameOID.COMMON_NAME, cn_name)]
        if org_name is not None:
            name.append(x509.NameAttribute(NameOID.ORGANIZATION_NAME, org_name))
        if role:
            name.append(x509.NameAttribute(NameOID.UNSTRUCTURED_NAME, role))
        return x509.Name(name)

    def finalize(self, ctx):
        state_dir = self.get_state_dir(ctx)
        cert_file = os.path.join(state_dir, "cert.json")
        json.dump(self.persistent_state, open(cert_file, "wt"))
