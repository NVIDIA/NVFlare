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
from base64 import b64decode
from enum import Enum

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding


class LoadResult(Enum):
    """Constants for different results when loading secure content."""

    OK = "ok"
    NOT_MANAGED = "notManaged"
    NO_SUCH_CONTENT = "noSuchContent"
    NOT_SIGNED = "notSigned"
    INVALID_SIGNATURE = "invalidSignature"
    INVALID_CONTENT = "invalidContent"


class SecurityContentManager(object):
    def __init__(self, content_folder, signature_filename="signature.json", root_cert="rootCA.pem"):
        """Content manager used by SecurityContentService to load secure content.

        Args:
            content_folder (str): the folder path that includes signature file
            signature_filename (str, optional): the signature file (signed dictionary). Defaults to "signature.json".
            root_cert (str, optional): root CA certificate filename. Defaults to "rootCA.pem".
        """
        self.content_folder = content_folder
        signature_path = os.path.join(self.content_folder, signature_filename)
        rootCA_cert_path = os.path.join(self.content_folder, root_cert)
        if os.path.exists(signature_path) and os.path.exists(rootCA_cert_path):
            self.signature = json.load(open(signature_path, "rt"))
            for k in self.signature:
                self.signature[k] = b64decode(self.signature[k].encode("utf-8"))
            cert = x509.load_pem_x509_certificate(open(rootCA_cert_path, "rb").read(), default_backend())
            self.public_key = cert.public_key()
            self.valid_config = True
        else:
            self.signature = dict()
            self.valid_config = False

    def load_content(self, file_under_verification):
        """Loads the data of the file under verification and verifies that the signature is valid.

        Args:
            file_under_verification: file to load and verify

        Returns:
            A tuple of the file data and the LoadResult. File data may be None if the data cannot be loaded.
        """
        full_path = os.path.join(self.content_folder, file_under_verification)
        data = None
        if not os.path.exists(full_path):
            return data, LoadResult.NO_SUCH_CONTENT

        with open(full_path, "rb") as f:
            data = f.read()
        if not data:
            return data, LoadResult.NO_SUCH_CONTENT

        if self.valid_config and file_under_verification in self.signature:
            signature = self.signature[file_under_verification]
            try:
                self.public_key.verify(
                    signature=signature,
                    data=data,
                    padding=padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                    algorithm=hashes.SHA256(),
                )
                result = LoadResult.OK
            except InvalidSignature:
                result = LoadResult.INVALID_SIGNATURE
        else:
            result = LoadResult.NOT_SIGNED
        return data, result


class SecurityContentService(object):
    """Uses SecurityContentManager to load secure content."""

    security_content_manager = None

    @staticmethod
    def initialize(content_folder: str, signature_filename="signature.json", root_cert="rootCA.pem"):
        if SecurityContentService.security_content_manager is None:
            SecurityContentService.security_content_manager = SecurityContentManager(
                content_folder, signature_filename, root_cert
            )

    @staticmethod
    def load_content(file_under_verification):
        if not SecurityContentService.security_content_manager:
            return None, LoadResult.NOT_MANAGED

        return SecurityContentService.security_content_manager.load_content(file_under_verification)

    @staticmethod
    def load_json(file_under_verification):
        if not SecurityContentService.security_content_manager:
            return None, LoadResult.NOT_MANAGED

        json_data = None

        data_bytes, result = SecurityContentService.security_content_manager.load_content(file_under_verification)

        if data_bytes:
            try:
                data_text = data_bytes.decode("ascii")
                json_data = json.loads(data_text)
            except json.JSONDecodeError:
                return None, LoadResult.INVALID_CONTENT

        return json_data, result
