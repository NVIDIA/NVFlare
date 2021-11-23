# Copyright (c) 2021, NVIDIA CORPORATION.
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
import pickle

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


class CertBuilder(Builder):
    def __init__(self):
        self.root_cert = None
        self.persistent_state = dict()

    def initialize(self, ctx):
        state_dir = self.get_state_dir(ctx)
        cert_file = os.path.join(state_dir, "cert.pkl")
        if os.path.exists(cert_file):
            self.persistent_state = pickle.load(open(cert_file, "rb"))
            self.serialized_cert = self.persistent_state["root_cert"]
            self.root_cert = x509.load_pem_x509_certificate(self.serialized_cert, default_backend())
            self.pri_key = serialization.load_pem_private_key(
                self.persistent_state["root_pri_key"], password=None, backend=default_backend()
            )
            self.pub_key = self.pri_key.public_key()
            self.subject = self.root_cert.subject

    def _build_root(self, subject):
        if not self.persistent_state:
            pri_key, pub_key = self._generate_keys()
            self.subject = self._x509_name(subject)
            issuer = self.subject
            self.root_cert = self._generate_cert(self.subject, issuer, pri_key, pub_key, ca=True)
            self.pri_key = pri_key
            self.pub_key = pub_key
            self.serialized_cert = serialize_cert(self.root_cert)
            self.persistent_state["root_cert"] = self.serialized_cert
            self.persistent_state["root_pri_key"] = serialize_pri_key(self.pri_key)

    def _build_write_cert_pair(self, participant, base_name, ctx):
        subject = participant.subject
        if self.persistent_state and subject in self.persistent_state:
            cert = x509.load_pem_x509_certificate(self.persistent_state[subject]["cert"], default_backend())
            pri_key = serialization.load_pem_private_key(
                self.persistent_state[subject]["pri_key"], password=None, backend=default_backend()
            )
        else:
            pri_key, cert = self.get_pri_key_cert(participant)
            self.persistent_state[subject] = dict(cert=serialize_cert(cert), pri_key=serialize_pri_key(pri_key))
        dest_dir = self.get_kit_dir(participant, ctx)
        with open(os.path.join(dest_dir, f"{base_name}.crt"), "wb") as f:
            f.write(serialize_cert(cert))
        with open(os.path.join(dest_dir, f"{base_name}.key"), "wb") as f:
            f.write(serialize_pri_key(pri_key))
        with open(os.path.join(dest_dir, "rootCA.pem"), "wb") as f:
            f.write(self.serialized_cert)

    def build(self, study, ctx):
        self._build_root(study.name)
        ctx["root_cert"] = self.root_cert
        ctx["root_pri_key"] = self.pri_key
        server = study.get_participants_by_type("server")
        self._build_write_cert_pair(server, "server", ctx)

        for client in study.get_participants_by_type("client", first_only=False):
            self._build_write_cert_pair(client, "client", ctx)

        for admin in study.get_participants_by_type("admin", first_only=False):
            self._build_write_cert_pair(admin, "client", ctx)

    def get_pri_key_cert(self, participant):
        pri_key, pub_key = self._generate_keys()
        subject = self._x509_name(participant.subject)
        issuer = self.subject
        cert = self._generate_cert(subject, issuer, self.pri_key, pub_key)
        return pri_key, cert

    def _generate_keys(self):
        pri_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
        pub_key = pri_key.public_key()
        return pri_key, pub_key

    def _generate_cert(self, subject, issuer, signing_pri_key, subject_pub_key, valid_days=360, ca=False):
        builder = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(subject_pub_key)
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.utcnow())
            .not_valid_after(
                # Our certificate will be valid for 360 days
                datetime.datetime.utcnow()
                + datetime.timedelta(days=valid_days)
                # Sign our certificate with our private key
            )
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

    def _x509_name(self, cn_name, org_name=None):
        name = [x509.NameAttribute(NameOID.COMMON_NAME, cn_name)]
        if org_name is not None:
            name.append(x509.NameAttribute(NameOID.ORGANIZATION_NAME, org_name))
        return x509.Name(name)

    def finalize(self, ctx):
        state_dir = self.get_state_dir(ctx)
        cert_file = os.path.join(state_dir, "cert.pkl")
        pickle.dump(self.persistent_state, open(cert_file, "wb"))
