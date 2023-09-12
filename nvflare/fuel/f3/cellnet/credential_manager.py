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
import logging
import threading

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from cryptography.x509 import Certificate

from nvflare.fuel.f3.cellnet.cell_cipher import SimpleCellCipher
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.endpoint import Endpoint

log = logging.getLogger(__name__)

CERT_ERROR = "cert_error"
CERT_TARGET = "cert_target"
CERT_ORIGIN = "cert_origin"
CERT_CONTENT = "cert_content"
CERT_CA_CONTENT = "cert_ca_content"
CERT_REQ_TIMEOUT = 10


class CredentialManager:
    """Helper class for secure message. It holds the local credentials and certificate cache"""

    def __init__(self, local_endpoint: Endpoint):

        self.local_endpoint = local_endpoint
        self.cert_cache = {}
        self.lock = threading.Lock()

        conn_props = self.local_endpoint.conn_props
        ca_cert_path = conn_props.get(DriverParams.CA_CERT)
        server_cert_path = conn_props.get(DriverParams.SERVER_CERT)
        if server_cert_path:
            local_cert_path = server_cert_path
            local_key_path = conn_props.get(DriverParams.SERVER_KEY)
        else:
            local_cert_path = conn_props.get(DriverParams.CLIENT_CERT)
            local_key_path = conn_props.get(DriverParams.CLIENT_KEY)

        if not local_cert_path:
            log.debug("Certificate is not configured, secure message is not supported")
            self.ca_cert = None
            self.local_cert = None
            self.local_key = None
            self.cell_cipher = None
        else:
            self.ca_cert = self.read_file(ca_cert_path)
            self.local_cert = self.read_file(local_cert_path)
            self.local_key = self.read_file(local_key_path)
            self.cell_cipher = SimpleCellCipher(self.get_ca_cert(), self.get_local_key(), self.get_local_cert())

        if not self.local_cert:
            log.debug("Certificate is not configured, secure message is not supported")
            self.cell_cipher = None
        else:
            self.cell_cipher = SimpleCellCipher(self.get_ca_cert(), self.get_local_key(), self.get_local_cert())

    def encrypt(self, target_cert: bytes, payload: bytes) -> bytes:

        if not self.cell_cipher:
            raise RuntimeError("Secure message not supported, Cell not running in secure mode")

        return self.cell_cipher.encrypt(payload, x509.load_pem_x509_certificate(target_cert))

    def decrypt(self, origin_cert: bytes, cipher: bytes) -> bytes:

        if not self.cell_cipher:
            raise RuntimeError("Secure message not supported, Cell not running in secure mode")

        return self.cell_cipher.decrypt(cipher, x509.load_pem_x509_certificate(origin_cert))

    def get_certificate(self, fqcn: str) -> bytes:

        if not self.cell_cipher:
            raise RuntimeError("This cell doesn't support certificate exchange, not running in secure mode")

        target = FQCN.get_root(fqcn)
        return self.cert_cache.get(target)

    def save_certificate(self, fqcn: str, cert: bytes):
        target = FQCN.get_root(fqcn)
        self.cert_cache[target] = cert

    def create_request(self, target: str) -> dict:

        req = {
            CERT_TARGET: target,
            CERT_ORIGIN: FQCN.get_root(self.local_endpoint.name),
            CERT_CONTENT: self.local_cert,
            CERT_CA_CONTENT: self.ca_cert,
        }

        return req

    def process_request(self, request: dict) -> dict:

        target = request.get(CERT_TARGET)
        origin = request.get(CERT_ORIGIN)

        reply = {CERT_TARGET: target, CERT_ORIGIN: origin}

        if not self.local_cert:
            reply[CERT_ERROR] = f"Target {target} is not running in secure mode"
        else:
            cert = request.get(CERT_CONTENT)

            # Save cert from requester in the cache
            self.cert_cache[origin] = cert

            reply[CERT_CONTENT] = self.local_cert
            reply[CERT_CA_CONTENT] = self.ca_cert

        return reply

    @staticmethod
    def process_response(reply: dict) -> bytes:

        error = reply.get(CERT_ERROR)
        if error:
            raise RuntimeError(f"Request to get certificate from {target} failed: {error}")

        return reply.get(CERT_CONTENT)

    def get_local_cert(self) -> Certificate:
        return x509.load_pem_x509_certificate(self.local_cert)

    def get_local_key(self) -> RSAPrivateKey:
        return serialization.load_pem_private_key(self.local_key, password=None)

    def get_ca_cert(self) -> Certificate:
        return x509.load_pem_x509_certificate(self.ca_cert)

    @staticmethod
    def read_file(file_name: str):
        if not file_name:
            return None

        with open(file_name, "rb") as f:
            return f.read()
