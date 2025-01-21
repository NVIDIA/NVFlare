# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import shutil
import tempfile

import pytest
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from nvflare.lighter.impl.cert import serialize_cert
from nvflare.lighter.utils import load_yaml, sign_folders, verify_folder_signature

folders = ["folder1", "folder2"]
files = ["file1", "file2"]


def generate_cert(subject, subject_org, issuer, signing_pri_key, subject_pub_key, valid_days=360, ca=False):
    def _x509_name(cn_name, org_name):
        name = [
            x509.NameAttribute(NameOID.COMMON_NAME, cn_name),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, org_name),
        ]
        return x509.Name(name)

    x509_subject = _x509_name(subject, subject_org)
    x509_issuer = _x509_name(issuer, "ORG")
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


def get_test_certs():
    root_pri_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
    root_pub_key = root_pri_key.public_key()
    root_cert = generate_cert("root", "nvidia", "root", root_pri_key, root_pub_key, ca=True)

    client_pri_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
    client_pub_key = client_pri_key.public_key()
    client_cert = generate_cert("client", "nvidia", "root", root_pri_key, client_pub_key)

    server_pri_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
    server_pub_key = server_pri_key.public_key()
    server_cert = generate_cert("client", "nvidia", "root", root_pri_key, server_pub_key)
    return root_cert, client_pri_key, client_cert, server_pri_key, server_cert


def create_folder():
    tmp_dir = tempfile.TemporaryDirectory().name
    for folder in folders:
        os.makedirs(os.path.join(tmp_dir, folder))
        for file in files:
            with open(os.path.join(tmp_dir, folder, file), "wb") as f:
                f.write(open("/dev/urandom", "rb").read(1024))

    return tmp_dir


def tamper_one_file(folder):
    with open(os.path.join(folder, folders[0], files[0]), "wt") as f:
        f.write("fail case")


def update_and_sign_one_folder(folder, pri_key, cert):
    tmp_dir = tempfile.TemporaryDirectory().name
    new_folder = os.path.join(tmp_dir, "new_folder")
    shutil.move(folder, new_folder)
    with open(os.path.join(tmp_dir, "test_file"), "wt") as f:
        f.write("fail case")
    with open("server.crt", "wb") as f:
        f.write(serialize_cert(cert))
    sign_folders(tmp_dir, pri_key, "server.crt", max_depth=1)
    return tmp_dir


def prepare_folders():
    root_cert, client_pri_key, client_cert, server_pri_key, server_cert = get_test_certs()
    folder = create_folder()
    with open("client.crt", "wb") as f:
        f.write(serialize_cert(client_cert))
    with open("root.crt", "wb") as f:
        f.write(serialize_cert(root_cert))
    sign_folders(folder, client_pri_key, "client.crt")
    return folder, server_pri_key, server_cert


@pytest.mark.xdist_group(name="lighter_utils_group")
class TestSignFolder:
    def test_verify_folder(self):
        folder, server_pri_key, server_cert = prepare_folders()
        assert verify_folder_signature(folder, "root.crt") is True
        tamper_one_file(folder)
        assert verify_folder_signature(folder, "root.crt") is False
        os.unlink("client.crt")
        os.unlink("root.crt")
        shutil.rmtree(folder)

    def test_verify_updated_folder(self):
        folder, server_pri_key, server_cert = prepare_folders()
        assert verify_folder_signature(folder, "root.crt") is True
        folder = update_and_sign_one_folder(folder, server_pri_key, server_cert)
        assert verify_folder_signature(folder, "root.crt") is True
        os.unlink("client.crt")
        os.unlink("root.crt")
        shutil.rmtree(folder)

    def _get_participant(self, name, participants):
        for p in participants:
            if p.get("name") == name:
                return p

    def test_load_yaml(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        data = load_yaml(os.path.join(dir_path, "0.yml"))

        assert data.get("server_name") == "server"

        participant = self._get_participant("server", data.get("participants"))
        assert participant.get("server_name") == "server"
        assert participant.get("extra").get("gpus") == "large"

        participant = self._get_participant("client", data.get("participants"))
        assert participant.get("client_name") == "client-1"
