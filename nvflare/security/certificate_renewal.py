# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Validate and report external certificate files during same-key renewal.

Files remain authoritative: validation is diagnostic, not an activation gate.
Enrollment tooling must publish valid credentials atomically; bad updates can
interrupt service. This observer retains no credential bytes or TLS contexts.
"""

import hashlib
import logging
import os
import time
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from nvflare.apis.fl_constant import ConnectionSecurity
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.sec.security_content_service import SecurityContentService
from nvflare.lighter.utils import verify_cert_chain

log = logging.getLogger(__name__)


def certificate_expiry(cert_data, root_data):
    chain = x509.load_pem_x509_certificates(cert_data)
    return min(c.not_valid_after_utc.timestamp() for c in chain + x509.load_pem_x509_certificates(root_data))


def _public_key(key):
    return key.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)


class RenewableCertificate:
    def __init__(self, cert_file, key_file, root_file, identity):
        self.cert_file = os.path.abspath(cert_file)
        self.root_file = root_file
        self.identity = identity
        self.error = None
        self.last_observed = None
        self._digest = None
        try:
            if not key_file:
                raise ValueError("private-key path is missing")
            if not root_file:
                raise ValueError("CA certificate path is missing")
            key = serialization.load_pem_private_key(Path(key_file).read_bytes(), password=None)
            if not isinstance(key, rsa.RSAPrivateKey) or key.key_size != 2048:
                raise ValueError("FLARE endpoint message encryption requires an RSA-2048 key")
            self.public_key = _public_key(key.public_key())
            self._inspect()
        except Exception as ex:
            raise ValueError(f"Enroll first: invalid endpoint credential at {self.cert_file}: {ex}") from ex

    def _inspect(self):
        data = Path(self.cert_file).read_bytes()
        chain = x509.load_pem_x509_certificates(data)
        cert = chain[0]
        if _public_key(cert.public_key()) != self.public_key:
            raise ValueError("private key change requires an endpoint restart")
        root_data = Path(self.root_file).read_bytes()
        verify_cert_chain(cert, chain[1:], x509.load_pem_x509_certificate(root_data))
        names = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
        if len(names) != 1 or names[0].value != self.identity:
            raise ValueError("certificate common name does not match configured endpoint identity")
        if any(isinstance(ext.value, x509.BasicConstraints) and ext.value.ca for ext in cert.extensions):
            raise ValueError("endpoint certificate must not be a CA certificate")
        self.expires = certificate_expiry(data, root_data)
        digest = hashlib.sha256(data).digest()
        changed = digest != self._digest
        self._digest = digest
        self.fingerprint = cert.fingerprint(hashes.SHA256()).hex()
        return changed

    def refresh(self):
        try:
            changed = self._inspect() or self.error is not None
            self.error = None
            if changed:
                self.last_observed = time.time()
                log.info("Observed renewed endpoint certificate: %s", self.cert_file)
            return changed
        except Exception as ex:
            error = str(ex)
            if error != self.error:
                log.error("Invalid endpoint credential for %s: %s", self.identity, error)
            self.error = error
            return False

    def status(self):
        return {
            "identity": self.identity,
            "fingerprint": self.fingerprint,
            "expires": self.expires,
            "expired": time.time() >= self.expires,
            "last_observed": self.last_observed,
            "key_age": "unknown (owned by enrollment policy)",
            "error": self.error,
        }


def watch_credentials(params, identity):
    content_manager = SecurityContentService.security_content_manager
    if content_manager and content_manager.valid_config:
        raise ValueError("certificate_renewal is not supported with signed HE/CC startup kits")
    if params.get(DriverParams.CONNECTION_SECURITY, ConnectionSecurity.MTLS) != ConnectionSecurity.MTLS:
        raise ValueError("certificate_renewal requires end-to-end mTLS")
    watched = {}
    for cert_param, key_param in (
        (DriverParams.SERVER_CERT, DriverParams.SERVER_KEY),
        (DriverParams.CLIENT_CERT, DriverParams.CLIENT_KEY),
    ):
        cert_file = params.get(cert_param)
        if cert_file:
            path = os.path.abspath(cert_file)
            if path not in watched:
                watched[path] = RenewableCertificate(
                    path, params.get(key_param), params.get(DriverParams.CA_CERT), identity
                )
    if not watched:
        raise ValueError("Enroll first: no server/client endpoint credential is configured")
    return list(watched.values())
