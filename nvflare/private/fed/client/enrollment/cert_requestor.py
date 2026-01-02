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
2. Submit CSR with enrollment token to the server via CellNet
3. Receive and save the signed certificate

Supports enrollment types:
- Client (site) enrollment: For FL clients (e.g., hospital-1, site-1)
- Admin (user) enrollment: For admin/researcher users
- Relay enrollment: For relay nodes

Uses CellNet for protocol-agnostic communication (grpc, https, tcp).

Example usage:

    from nvflare.fuel.f3.cellnet.cell import Cell
    from nvflare.private.fed.client.enrollment import (
        CertRequestor,
        EnrollmentIdentity,
        EnrollmentOptions,
    )

    # Initialize CellNet
    cell = Cell(
        fqcn="client_1",
        root_url="grpc://server:8002",
        secure=True,
        credentials={},
    )
    cell.start()

    # Client (site) enrollment
    identity = EnrollmentIdentity.for_client("hospital-1", org="Hospital A")

    # Or admin (user) enrollment
    # identity = EnrollmentIdentity.for_admin("researcher@example.com", role="member")

    # Or relay enrollment
    # identity = EnrollmentIdentity.for_relay("relay-1", org="nvidia")

    options = EnrollmentOptions(
        timeout=30.0,
        output_dir="/workspace/certs",
    )

    requestor = CertRequestor(
        cell=cell,
        enrollment_token="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
        identity=identity,
        options=options,
    )

    try:
        cert_path = requestor.request_certificate()
        print(f"Certificate saved to: {cert_path}")
    except Exception as e:
        print(f"Enrollment failed: {e}")
    finally:
        cell.stop()
"""

import logging
import os
from typing import Optional

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from pydantic import BaseModel, field_validator

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import CellChannel, CellChannelTopic
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.lighter.constants import (
    AdminRole,
    DEFINED_PARTICIPANT_TYPES,
    DEFINED_ROLES,
    ParticipantType,
)
from nvflare.lighter.utils import generate_keys, serialize_pri_key


# =============================================================================
# Configuration Classes (Simple interface, Pydantic validation)
# =============================================================================

class EnrollmentIdentity(BaseModel):
    """Client identity information for certificate enrollment.
    
    Supports participant types from nvflare.lighter.constants.ParticipantType:
    - client: FL client nodes (hospital-1, site-1, etc.)
    - admin: Admin/researcher users with roles
    - relay: Relay nodes for network topology
    
    Use factory methods for convenience:
        EnrollmentIdentity.for_client("hospital-1")
        EnrollmentIdentity.for_admin("admin@example.com", role="org_admin")
        EnrollmentIdentity.for_relay("relay-1")
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
        role: str = AdminRole.MEMBER,
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


# =============================================================================
# CertRequestor
# =============================================================================

class CertRequestor:
    """Requests certificate enrollment via CSR workflow using CellNet.

    This client-side component:
    1. Generates a new RSA key pair
    2. Creates a CSR following FLARE's certificate structure
    3. Submits the CSR via CellNet to the server
    4. Receives and saves the signed certificate

    Supports client, admin, and relay enrollment.
    Uses CellNet for protocol-agnostic communication (grpc, https, tcp).
    """

    # Default certificate filenames
    CERT_FILENAME = "client.crt"
    KEY_FILENAME = "client.key"
    ROOT_CA_FILENAME = "rootCA.pem"

    def __init__(
        self,
        cell: Cell,
        enrollment_token: str,
        identity: EnrollmentIdentity,
        target_fqcn: str = "server",
        options: Optional[EnrollmentOptions] = None,
    ):
        """Initialize the certificate requestor.

        Args:
            cell: CellNet instance for communication
            enrollment_token: JWT token with embedded policy
            identity: Client identity information (client, admin, or relay)
            target_fqcn: Server FQCN to connect to
            options: Optional configuration parameters
            
        Raises:
            ValueError: If enrollment_token is empty
        """
        if not enrollment_token or not enrollment_token.strip():
            raise ValueError("enrollment_token cannot be empty")
            
        self.cell = cell
        self.enrollment_token = enrollment_token.strip()
        self.identity = identity
        self.target_fqcn = target_fqcn
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
            name_attributes.append(
                x509.NameAttribute(x509.NameOID.ORGANIZATION_NAME, self.identity.org)
            )
        # Include participant type and role in certificate
        if self.identity.participant_type:
            name_attributes.append(
                x509.NameAttribute(x509.NameOID.ORGANIZATIONAL_UNIT_NAME, self.identity.participant_type)
            )
        if self.identity.role:
            name_attributes.append(
                x509.NameAttribute(x509.NameOID.UNSTRUCTURED_NAME, self.identity.role)
            )

        builder = x509.CertificateSigningRequestBuilder().subject_name(
            x509.Name(name_attributes)
        )

        csr = builder.sign(self.private_key, hashes.SHA256())
        self.logger.debug(
            f"Created CSR for {self.identity.participant_type}: {self.identity.name}"
        )
        
        return csr.public_bytes(serialization.Encoding.PEM)

    def submit_csr(self, csr_data: bytes) -> bytes:
        """Submit CSR to server for signing.
        
        Args:
            csr_data: PEM-encoded CSR
            
        Returns:
            Signed certificate bytes
            
        Raises:
            RuntimeError: If CSR submission fails
        """
        headers = {
            "enrollment_token": self.enrollment_token,
            "identity": self.identity.name,
            "participant_type": self.identity.participant_type,
        }
        if self.identity.role:
            headers["role"] = self.identity.role

        message = new_cell_message(headers, csr_data)

        self.logger.info(
            f"Submitting {self.identity.participant_type} CSR to: {self.target_fqcn}"
        )
        
        result = self.cell.send_request(
            target=self.target_fqcn,
            channel=CellChannel.SERVER_MAIN,
            topic=CellChannelTopic.CSR_ENROLLMENT,
            request=message,
            timeout=self.options.timeout,
        )

        if result is None:
            raise RuntimeError("No response from server")

        return_code = result.get_header("return_code")
        if return_code != "OK":
            error = result.get_header("error", "Unknown error")
            raise RuntimeError(f"CSR submission failed: {error}")

        self.logger.info(f"{self.identity.participant_type} CSR signed successfully")
        return result.payload

    def save_credentials(self, cert_data: bytes) -> None:
        """Save private key and certificate to files.
        
        Args:
            cert_data: Signed certificate bytes
        """
        os.makedirs(self.options.output_dir, exist_ok=True)

        # Save private key (with restricted permissions)
        key_path = os.path.join(self.options.output_dir, self.KEY_FILENAME)
        with open(key_path, "wb") as f:
            f.write(serialize_pri_key(self.private_key))
        os.chmod(key_path, 0o600)
        self.logger.info(f"Saved private key to: {key_path}")

        # Save certificate
        cert_path = os.path.join(self.options.output_dir, self.CERT_FILENAME)
        with open(cert_path, "wb") as f:
            f.write(cert_data)
        self.logger.info(f"Saved certificate to: {cert_path}")

    def request_certificate(self) -> str:
        """Complete certificate enrollment workflow.
        
        Returns:
            Path to the saved certificate file
            
        Raises:
            RuntimeError: If enrollment fails
        """
        self.logger.info(
            f"Starting {self.identity.participant_type} enrollment for: {self.identity.name}"
        )

        # Generate CSR
        csr_data = self.create_csr()

        # Submit to server
        cert_data = self.submit_csr(csr_data)

        # Save credentials
        self.save_credentials(cert_data)

        cert_path = os.path.join(self.options.output_dir, self.CERT_FILENAME)
        self.logger.info(f"Enrollment complete: {cert_path}")
        
        return cert_path
