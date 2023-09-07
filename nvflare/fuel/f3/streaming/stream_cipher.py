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

from cryptography import x509
from cryptography.hazmat.primitives import serialization

from nvflare.fuel.f3.cellnet.cell_cipher import SimpleCellCipher
from nvflare.fuel.f3.cellnet.core_cell import CoreCell

from nvflare.fuel.f3.streaming.certificate_manager import CertificateManager

log = logging.getLogger(__name__)


class StreamCipher:

    def __init__(self, core_cell: CoreCell):

        self.cert_mgr = CertificateManager(core_cell)
        ca_cert = x509.load_pem_x509_certificate(self.cert_mgr.get_ca_cert())
        cert = x509.load_pem_x509_certificate(self.cert_mgr.get_local_cert())
        private_key = serialization.load_pem_private_key(self.cert_mgr.get_local_key(), password=None)
        self.cell_cipher = SimpleCellCipher(ca_cert, private_key, cert)

    def encrypt(self, target: str, payload: bytes) -> bytes:
        target_cert = x509.load_pem_x509_certificate(self.cert_mgr.get_certificate(target))
        return self.cell_cipher.encrypt(payload, target_cert)

    def decrypt(self, origin: str, cipher: bytes) -> bytes:
        origin_cert = x509.load_pem_x509_certificate(self.cert_mgr.get_certificate(origin))
        return self.cell_cipher.decrypt(cipher, origin_cert)
