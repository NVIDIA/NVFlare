# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""FLARE Enrollment Certificate Requestor.

This module handles the client-side enrollment workflow:
1. Generate private key and CSR (Certificate Signing Request)
2. Submit CSR with enrollment token to Certificate Service via HTTP
3. Receive signed certificate and rootCA.pem
4. Save credentials to disk

Supports enrollment types:
- Client (site) enrollment: For FL clients (e.g., hospital-1, site-1)
- Admin (user) enrollment: For admin/researcher users
- Relay enrollment: For relay nodes
- Server enrollment: For FL servers

Uses HTTP to communicate with the Certificate Service.

Example usage:

    from nvflare.private.fed.client.enrollment import (
        CertRequestor,
        EnrollmentIdentity,
        EnrollmentOptions,
    )

    # Client (site) enrollment
    identity = EnrollmentIdentity.for_client("hospital-1", org="Hospital A")

    # Or admin (user) enrollment
    # identity = EnrollmentIdentity.for_admin("researcher@example.com", role="member")

    # Or relay enrollment
    # identity = EnrollmentIdentity.for_relay("relay-1", org="nvidia")

    options = EnrollmentOptions(
        timeout=30.0,
        output_dir="/workspace/startup",
    )

    requestor = CertRequestor(
        cert_service_url="https://cert-service.example.com",
        enrollment_token="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
        identity=identity,
        options=options,
    )

    try:
        result = requestor.request_certificate()
        print(f"Certificate saved to: {result.cert_path}")
        print(f"Root CA saved to: {result.ca_path}")
    except Exception as e:
        print(f"Enrollment failed: {e}")
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from pydantic import BaseModel, field_validator

from nvflare.lighter.constants import DEFINED_PARTICIPANT_TYPES, DEFINED_ROLES, AdminRole, ParticipantType
from nvflare.lighter.utils import generate_keys, serialize_pri_key

# =============================================================================
# Configuration Classes (Simple interface, Pydantic validation)
# =============================================================================


class EnrollmentIdentity(BaseModel):
    """Client identity information for certificate enrollment.

    Use factory methods for convenience:
        EnrollmentIdentity.for_client("hospital-1")
        EnrollmentIdentity.for_admin("admin@example.com", role="org_admin")
        EnrollmentIdentity.for_relay("relay-1")
        EnrollmentIdentity.for_server("server1")
    """

    name: str
    participant_type: str = ParticipantType.CLIENT
    org: Optional[str] = None
    role: Optional[str] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("name cannot be empty")
        return v.strip()

    @field_validator("participant_type")
    @classmethod
    def validate_participant_type(cls, v: str) -> str:
        if v not in DEFINED_PARTICIPANT_TYPES:
            raise ValueError(f"participant_type must be one of: {DEFINED_PARTICIPANT_TYPES}")
        return v

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in DEFINED_ROLES:
            raise ValueError(f"role must be one of: {DEFINED_ROLES}")
        return v

    @classmethod
    def for_client(cls, name: str, org: str = None) -> "EnrollmentIdentity":
        """Create identity for client (site) enrollment.

        Args:
            name: Site name (e.g., hospital-1, site-1)
            org: Organization name
        """
        return cls(
            name=name,
            participant_type=ParticipantType.CLIENT,
            org=org,
        )

    @classmethod
    def for_admin(
        cls,
        email: str,
        role: str = AdminRole.LEAD,
        org: str = None,
    ) -> "EnrollmentIdentity":
        """Create identity for admin (user) enrollment.

        Args:
            email: User email address
            role: Admin role (project_admin, org_admin, lead, member)
            org: Organization name
        """
        return cls(
            name=email,
            participant_type=ParticipantType.ADMIN,
            org=org,
            role=role,
        )

    @classmethod
    def for_relay(cls, name: str, org: str = None) -> "EnrollmentIdentity":
        """Create identity for relay node enrollment.

        Args:
            name: Relay node name (e.g., relay-1)
            org: Organization name
        """
        return cls(
            name=name,
            participant_type=ParticipantType.RELAY,
            org=org,
        )

    @classmethod
    def for_server(cls, name: str, org: str = None) -> "EnrollmentIdentity":
        """Create identity for server enrollment.

        Args:
            name: Server name (e.g., server1)
            org: Organization name
        """
        return cls(
            name=name,
            participant_type=ParticipantType.SERVER,
            org=org,
        )


class EnrollmentOptions(BaseModel):
    """Configuration options for certificate enrollment."""

    timeout: float = 30.0
    retry_count: int = 3
    cert_valid_days: int = 360
    output_dir: str = "."

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("timeout must be positive")
        return v

    @field_validator("retry_count")
    @classmethod
    def validate_retry_count(cls, v: int) -> int:
        if v < 0:
            raise ValueError("retry_count must be non-negative")
        return v

    @field_validator("cert_valid_days")
    @classmethod
    def validate_cert_valid_days(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("cert_valid_days must be positive")
        return v


@dataclass
class EnrollmentResult:
    """Result of enrollment - contains both in-memory certs and file paths."""

    # In-memory data
    private_key: Any  # RSA private key object
    certificate_pem: str  # PEM-encoded signed certificate
    ca_cert_pem: str  # PEM-encoded root CA certificate

    # File paths (files saved for restart persistence)
    cert_path: str
    key_path: str
    ca_path: str


# =============================================================================
# CertRequestor
# =============================================================================


class CertRequestor:
    """Requests certificate enrollment via CSR workflow using HTTP.

    This client-side component:
    1. Generates a new RSA key pair
    2. Creates a CSR following FLARE's certificate structure
    3. Submits the CSR via HTTP to the Certificate Service
    4. Receives signed certificate and rootCA.pem
    5. Saves credentials to disk

    Supports client, admin, relay, and server enrollment.
    Uses HTTP to communicate with the standalone Certificate Service.

    Security:
    - Private key is generated locally and never transmitted
    - rootCA.pem is only returned after token is validated
    """

    def __init__(
        self,
        cert_service_url: str,
        enrollment_token: str,
        identity: EnrollmentIdentity,
        options: Optional[EnrollmentOptions] = None,
    ):
        """Initialize the certificate requestor.

        Args:
            cert_service_url: URL of the Certificate Service (e.g., https://cert-svc:8443)
            enrollment_token: JWT token with embedded policy
            identity: Client identity information (client, admin, relay, or server)
            options: Optional configuration parameters

        Raises:
            ValueError: If cert_service_url or enrollment_token is empty
        """
        if not cert_service_url or not cert_service_url.strip():
            raise ValueError("cert_service_url cannot be empty")
        if not enrollment_token or not enrollment_token.strip():
            raise ValueError("enrollment_token cannot be empty")

        self.cert_service_url = cert_service_url.rstrip("/")
        self.enrollment_token = enrollment_token.strip()
        self.identity = identity
        self.options = options or EnrollmentOptions()

        self.logger = logging.getLogger(self.__class__.__name__)

        # Generated key pair
        self.private_key = None
        self.public_key = None

    def generate_key_pair(self) -> None:
        """Generate RSA key pair for CSR."""
        self.private_key, self.public_key = generate_keys()
        self.logger.debug(f"Generated key pair for: {self.identity.name}")

    def create_csr(self) -> bytes:
        """Create Certificate Signing Request.

        Returns:
            PEM-encoded CSR bytes
        """
        if not self.private_key:
            self.generate_key_pair()

        # Build CSR with identity attributes
        name_attributes = [
            x509.NameAttribute(x509.NameOID.COMMON_NAME, self.identity.name),
        ]
        if self.identity.org:
            name_attributes.append(x509.NameAttribute(x509.NameOID.ORGANIZATION_NAME, self.identity.org))
        # Include participant type and role in certificate
        if self.identity.participant_type:
            name_attributes.append(
                x509.NameAttribute(x509.NameOID.ORGANIZATIONAL_UNIT_NAME, self.identity.participant_type)
            )
        if self.identity.role:
            name_attributes.append(x509.NameAttribute(x509.NameOID.UNSTRUCTURED_NAME, self.identity.role))

        builder = x509.CertificateSigningRequestBuilder().subject_name(x509.Name(name_attributes))

        csr = builder.sign(self.private_key, hashes.SHA256())
        self.logger.debug(f"Created CSR for {self.identity.participant_type}: {self.identity.name}")

        return csr.public_bytes(serialization.Encoding.PEM)

    def submit_csr(self, csr_data: bytes) -> dict:
        """Submit CSR to Certificate Service for signing.

        Args:
            csr_data: PEM-encoded CSR

        Returns:
            Dict with 'certificate' and 'ca_cert' keys

        Raises:
            RuntimeError: If CSR submission fails
        """
        try:
            import requests
        except ImportError:
            raise ImportError("requests library is required. Install with: pip install requests")

        url = f"{self.cert_service_url}/api/v1/enroll"

        payload = {
            "token": self.enrollment_token,
            "csr": csr_data.decode("utf-8"),
            "metadata": {
                "name": self.identity.name,
                "type": self.identity.participant_type,
                "org": self.identity.org,
                "role": self.identity.role,
            },
        }

        self.logger.info(f"Submitting {self.identity.participant_type} CSR to: {url}")

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.options.timeout,
                # Certificate Service uses public CA (Let's Encrypt)
                # so no custom CA verification needed
            )

            if response.status_code == 401:
                error = response.json().get("error", "Invalid or expired token")
                raise RuntimeError(f"Authentication failed: {error}")
            elif response.status_code == 403:
                error = response.json().get("error", "Policy rejection")
                raise RuntimeError(f"Enrollment rejected: {error}")
            elif response.status_code != 200:
                error = response.json().get("error", response.text)
                raise RuntimeError(f"CSR submission failed: {error}")

            result = response.json()
            self.logger.info(f"{self.identity.participant_type} CSR signed successfully")
            return result

        except requests.RequestException as e:
            raise RuntimeError(f"Failed to connect to Certificate Service: {e}")

    def save_credentials(self, cert_pem: str, ca_cert_pem: str) -> tuple:
        """Save private key, certificate, and root CA to files.

        Args:
            cert_pem: PEM-encoded signed certificate
            ca_cert_pem: PEM-encoded root CA certificate

        Returns:
            Tuple of (cert_path, key_path, ca_path)
        """
        os.makedirs(self.options.output_dir, exist_ok=True)

        # Determine filenames based on participant type
        if self.identity.participant_type == ParticipantType.SERVER:
            cert_filename = "server.crt"
            key_filename = "server.key"
        else:
            cert_filename = "client.crt"
            key_filename = "client.key"

        # Save private key (with restricted permissions)
        key_path = os.path.join(self.options.output_dir, key_filename)
        with open(key_path, "wb") as f:
            f.write(serialize_pri_key(self.private_key))
        os.chmod(key_path, 0o600)
        self.logger.info(f"Saved private key to: {key_path}")

        # Save certificate
        cert_path = os.path.join(self.options.output_dir, cert_filename)
        with open(cert_path, "w") as f:
            f.write(cert_pem)
        self.logger.info(f"Saved certificate to: {cert_path}")

        # Save root CA
        ca_path = os.path.join(self.options.output_dir, "rootCA.pem")
        with open(ca_path, "w") as f:
            f.write(ca_cert_pem)
        self.logger.info(f"Saved root CA to: {ca_path}")

        return cert_path, key_path, ca_path

    def request_certificate(self) -> EnrollmentResult:
        """Complete certificate enrollment workflow.

        Returns:
            EnrollmentResult with in-memory certs and file paths

        Raises:
            RuntimeError: If enrollment fails
        """
        self.logger.info(f"Starting {self.identity.participant_type} enrollment for: {self.identity.name}")

        # Generate CSR
        csr_data = self.create_csr()

        # Submit to Certificate Service
        result = self.submit_csr(csr_data)
        cert_pem = result["certificate"]
        ca_cert_pem = result["ca_cert"]

        # Save credentials
        cert_path, key_path, ca_path = self.save_credentials(cert_pem, ca_cert_pem)

        self.logger.info(f"Enrollment complete: {cert_path}")

        return EnrollmentResult(
            private_key=self.private_key,
            certificate_pem=cert_pem,
            ca_cert_pem=ca_cert_pem,
            cert_path=cert_path,
            key_path=key_path,
            ca_path=ca_path,
        )
