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
from typing import Optional

from cryptography import x509
from cryptography.x509.oid import NameOID

from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.lighter.utils import (
    load_crt,
    load_crt_bytes,
    load_crt_chain_bytes,
    load_private_key_file,
    sign_content,
    verify_cert,
    verify_cert_chain,
    verify_content,
)
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


def get_org_from_cert(cert) -> str:
    attr = cert.subject.get_attributes_for_oid(NameOID.ORGANIZATION_NAME)
    return attr[0].value if attr else ""


def load_cert_file(path: str):
    return load_crt(path)


def load_cert_bytes(data: bytes):
    return load_crt_bytes(data)


def load_cert_chain_bytes(data: bytes):
    return load_crt_chain_bytes(data)


def get_parent_site_name(fqsn: str) -> Optional[str]:
    """Get the parent site's name for a specified FQSN (fully qualified site name)

    Args:
        fqsn: the FQSN to find parent for

    Returns: the parent site's name or None if the FQSN doesn't have a parent

    """
    if not fqsn:
        return None

    if not isinstance(fqsn, str):
        raise ValueError(f"expect fqsn to be str but got {type(fqsn)}")

    parts = fqsn.split(".")
    if len(parts) <= 1:
        return None
    return parts[len(parts) - 2]


class IdentityAsserter:
    def __init__(self, private_key_file: str, cert_file: str):
        with open(cert_file, "rb") as f:
            self.cert_data = f.read()
        self.private_key_file = private_key_file
        self.pri_key = load_private_key_file(private_key_file)
        self.cert_file = cert_file
        self.cert = load_cert_bytes(self.cert_data)
        self.cn = get_cn_from_cert(self.cert)

    def sign_common_name(self, nonce: str) -> str:
        return sign_content(self.cn + nonce, self.pri_key, return_str=False)

    def sign(self, content, return_str: bool) -> str:
        return sign_content(content, self.pri_key, return_str=return_str)

    def verify_signature(self, content, signature) -> bool:
        pub_key = self.cert.public_key()
        try:
            verify_content(content=content, signature=signature, public_key=pub_key)
            return True
        except Exception:
            return False


class IdentityVerifier:
    def __init__(self, root_cert_file: str):
        self.root_cert = load_cert_file(root_cert_file)
        self.root_public_key = self.root_cert.public_key()

    def verify_common_name(
        self, asserted_cn: str, nonce: str, asserter_cert, signature, intermediate_certs=None, expected_eku=None
    ) -> bool:
        # verify asserter_cert
        try:
            if intermediate_certs is not None:
                verify_cert_chain(
                    leaf_cert=asserter_cert,
                    intermediate_certs=intermediate_certs,
                    root_ca_cert=self.root_cert,
                )
            else:
                verify_cert(
                    cert_to_be_verified=asserter_cert,
                    root_ca_public_key=self.root_public_key,
                )
            _validate_identity_cert_usage(asserter_cert, expected_eku)
        except Exception as ex:
            raise InvalidAsserterCert(str(ex)) from ex

        # verify signature provided by the asserter
        asserter_public_key = asserter_cert.public_key()
        cn = get_cn_from_cert(asserter_cert)

        if cn != asserted_cn:
            raise CNMismatch()

        assert isinstance(cn, str)
        try:
            verify_content(content=cn + nonce, signature=signature, public_key=asserter_public_key)
        except Exception as ex:
            raise InvalidCNSignature(f"cannot verify common name signature: {secure_format_exception(ex)}")
        return True


def _validate_identity_cert_usage(cert, expected_eku):
    """Enforce certificate usage restrictions for the common-name challenge.

    Legacy FLARE certificates may omit KeyUsage and ExtendedKeyUsage, so absent
    extensions remain unrestricted. When present, they must allow the signing
    operation and the caller-specific authentication purpose.
    """
    try:
        key_usage = cert.extensions.get_extension_for_class(x509.KeyUsage).value
    except x509.ExtensionNotFound:
        pass
    else:
        if not key_usage.digital_signature:
            raise ValueError("asserter certificate keyUsage must allow digitalSignature")

    if expected_eku is None:
        return

    try:
        extended_key_usage = cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    except x509.ExtensionNotFound:
        return
    if expected_eku not in extended_key_usage:
        raise ValueError(f"asserter certificate extendedKeyUsage must allow {expected_eku.dotted_string}")


class TokenVerifier:
    def __init__(self, cert):
        self.cert = cert
        self.public_key = cert.public_key()
        self.logger = get_obj_logger(self)

    def verify(self, client_name, token, signature):
        try:
            verify_content(content=client_name + token, signature=signature, public_key=self.public_key)
            return True
        except Exception as ex:
            self.logger.error(f"exception verifying token: {client_name=} {token=}: {secure_format_exception(ex)}")
            return False
