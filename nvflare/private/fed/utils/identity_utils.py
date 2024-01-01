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

from cryptography.x509.oid import NameOID

from nvflare.lighter.impl.cert import load_crt, load_crt_bytes
from nvflare.lighter.utils import load_private_key_file, sign_one, verify_cert, verify_one
from nvflare.security.logging import secure_format_exception


class CNMismatch(Exception):
    pass


class MissingCN(Exception):
    pass


class InvalidAsserterCert(Exception):
    pass


class InvalidCNSignature(Exception):
    pass


def get_cn_from_cert(cert):
    subject = cert.subject
    attr = subject.get_attributes_for_oid(NameOID.COMMON_NAME)
    if not attr:
        raise MissingCN()
    return attr[0].value


def load_cert_file(path: str):
    return load_crt(path)


def load_cert_bytes(data: bytes):
    return load_crt_bytes(data)


class IdentityAsserter:
    def __init__(self, private_key_file: str, cert_file: str):
        self.private_key_file = private_key_file
        self.pri_key = load_private_key_file(private_key_file)
        self.cert_file = cert_file
        self.cert = load_cert_file(cert_file)

    def sign_common_name(self, asserted_cn: str) -> str:
        cn = get_cn_from_cert(self.cert)
        if cn != asserted_cn:
            raise CNMismatch(f"asserted_cn {asserted_cn} != CN from cert {cn}")
        return sign_one(cn, self.pri_key)


class IdentityVerifier:
    def __init__(self, root_cert_file: str):
        self.root_cert = load_cert_file(root_cert_file)
        self.root_public_key = self.root_cert.public_key()

    def verify_common_name(self, asserted_cn: str, asserter_cert, signature) -> bool:
        # verify asserter_cert
        try:
            verify_cert(
                cert_to_be_verified=asserter_cert,
                root_ca_public_key=self.root_public_key,
            )
        except:
            raise InvalidAsserterCert()

        # verify signature provided by the asserter
        asserter_public_key = asserter_cert.public_key()
        cn = get_cn_from_cert(asserter_cert)

        if cn != asserted_cn:
            raise CNMismatch()

        assert isinstance(cn, str)
        try:
            verify_one(content=cn, signature=signature, public_key=asserter_public_key)
        except Exception as ex:
            raise InvalidCNSignature(f"cannot verify common name signature: {secure_format_exception(ex)}")
        return True
