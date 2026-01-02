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
- Site enrollment: For FL clients (e.g., hospital-1, site-1)
- User enrollment: For admin/researcher users (e.g., researcher@example.com)
- Relay enrollment: For relay nodes

Uses CellNet for communication (supports grpc, https, tcp protocols).
Uses Pydantic models for input validation.

Example usage:

    # Create identity and requestor
    identity = EnrollmentIdentity.for_site(
        site_name="hospital-1",
        org_name="Hospital A",
    )
    
    # With existing Cell connection
    requestor = CertRequestor(
        cell=cell,  # CellNet Cell object
        enrollment_token="eyJ...",
        identity=identity,
    )
    result = requestor.enroll()
    
    # Check result
    if result.success:
        print(f"Certificate saved to: {result.cert_path}")
    elif result.pending_approval:
        print(f"Awaiting approval: {result.approval_message}")
    else:
        print(f"Failed: {result.error_message}")
"""

import logging
import os
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import (
    CellChannel,
    CellChannelTopic,
    MessageHeaderKey,
    ReturnCode,
)
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.fuel.f3.message import Message
from nvflare.lighter.constants import ParticipantType
from nvflare.lighter.utils import Identity, generate_csr, generate_keys, serialize_csr, serialize_pri_key


# =============================================================================
# Constants
# =============================================================================

# Valid participant types for enrollment
VALID_PARTICIPANT_TYPES = [ParticipantType.CLIENT, ParticipantType.ADMIN, ParticipantType.RELAY]

# Type alias for participant types
ParticipantTypeEnum = Literal["client", "admin", "relay"]


class UserRole:
    """User roles for enrollment."""
    PROJECT_ADMIN = "project_admin"
    ORG_ADMIN = "org_admin"
    LEAD = "lead"
    MEMBER = "member"
    RESEARCHER = "researcher"


# Valid user roles
VALID_ROLES = [
    UserRole.PROJECT_ADMIN,
    UserRole.ORG_ADMIN,
    UserRole.LEAD,
    UserRole.MEMBER,
    UserRole.RESEARCHER,
]

# Type alias for roles
UserRoleEnum = Literal["project_admin", "org_admin", "lead", "member", "researcher"]


# =============================================================================
# Configuration Models (Pydantic)
# =============================================================================

class EnrollmentIdentity(BaseModel):
    """Identity information for enrollment.
    
    Encapsulates who is enrolling - either a site or a user.
    """
    model_config = ConfigDict(validate_assignment=True)
    
    subject_name: str = Field(..., min_length=1, description="Site name or user email")
    participant_type: ParticipantTypeEnum = Field(
        default="client",
        description="Type of participant: client, admin, or relay"
    )
    org_name: Optional[str] = Field(default=None, description="Organization name")
    role: Optional[UserRoleEnum] = Field(default=None, description="Role for user enrollment")
    host_names: List[str] = Field(default_factory=list, description="Additional hostnames for cert")

    @classmethod
    def for_site(
        cls,
        site_name: str,
        org_name: Optional[str] = None,
        host_names: Optional[List[str]] = None,
    ) -> "EnrollmentIdentity":
        """Create identity for site enrollment."""
        return cls(
            subject_name=site_name,
            participant_type="client",
            org_name=org_name,
            host_names=host_names or [],
        )

    @classmethod
    def for_user(
        cls,
        user_email: str,
        role: UserRoleEnum = "researcher",
        org_name: Optional[str] = None,
    ) -> "EnrollmentIdentity":
        """Create identity for user enrollment."""
        return cls(
            subject_name=user_email,
            participant_type="admin",
            role=role,
            org_name=org_name,
        )

    @classmethod
    def for_relay(
        cls,
        relay_name: str,
        org_name: Optional[str] = None,
        host_names: Optional[List[str]] = None,
    ) -> "EnrollmentIdentity":
        """Create identity for relay enrollment."""
        return cls(
            subject_name=relay_name,
            participant_type="relay",
            org_name=org_name,
            host_names=host_names or [],
        )


class EnrollmentOptions(BaseModel):
    """Optional enrollment settings."""
    model_config = ConfigDict(validate_assignment=True)
    
    output_dir: Optional[str] = Field(default=None, description="Directory for certificates")
    source_ip: Optional[str] = Field(
        default=None,
        description="Client IP for policy verification (optional, for static instances only)"
    )
    timeout: float = Field(default=30.0, description="Request timeout in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata")

    def get_output_dir(self) -> str:
        """Get output directory, defaulting to ./certs."""
        return self.output_dir or os.path.join(os.getcwd(), "certs")


# =============================================================================
# Request/Response Models (Pydantic)
# =============================================================================

class EnrollmentResult(BaseModel):
    """Result of enrollment request."""
    model_config = ConfigDict(validate_assignment=True)
    
    success: bool
    subject_name: str
    participant_type: ParticipantTypeEnum
    cert_path: Optional[str] = None
    key_path: Optional[str] = None
    root_ca_path: Optional[str] = None
    error_message: Optional[str] = None
    pending_approval: bool = False
    approval_message: Optional[str] = None


class EnrollmentRequest(BaseModel):
    """Enrollment request payload sent to server."""
    model_config = ConfigDict(validate_assignment=True)
    
    subject_name: str = Field(..., min_length=1)
    participant_type: ParticipantTypeEnum
    csr: str = Field(..., min_length=1, description="PEM-encoded CSR")
    org_name: Optional[str] = None
    role: Optional[UserRoleEnum] = None
    source_ip: Optional[str] = None
    host_names: Optional[List[str]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization (excludes None values)."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


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
            cell: CellNet Cell for communication (handles grpc/https/tcp)
            enrollment_token: JWT enrollment token from admin
            identity: Identity configuration (who is enrolling)
            target_fqcn: FQCN of the server to send request to
            options: Enrollment options (timeout, output dir, etc.)
            
        Raises:
            ValueError: If enrollment_token is empty
        """
        if not enrollment_token:
            raise ValueError("enrollment_token is required")
            
        self.cell = cell
        self.enrollment_token = enrollment_token
        self.identity = identity
        self.target_fqcn = target_fqcn
        self.options = options or EnrollmentOptions()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Private key and CSR (generated during enrollment)
        self._private_key = None
        self._csr = None

    def enroll(self) -> EnrollmentResult:
        """Perform the full enrollment workflow.
        
        1. Generate key pair and CSR
        2. Submit enrollment request to server via CellNet
        3. Save certificates and keys to output directory
        
        Returns:
            EnrollmentResult with paths to saved files or error message
        """
        try:
            # Step 1: Generate key pair and CSR
            self.logger.info(
                f"Generating key pair and CSR for {self.identity.participant_type}: "
                f"{self.identity.subject_name}"
            )
            self._generate_key_and_csr()
            
            # Step 2: Submit enrollment request via CellNet
            self.logger.info(f"Submitting enrollment request to: {self.target_fqcn}")
            response = self._submit_enrollment()
            
            # Step 3: Process response and save files
            self.logger.info("Processing enrollment response")
            result = self._process_response(response)
            
            if result.success:
                self.logger.info(
                    f"Enrollment successful! Certificate saved to: {result.cert_path}"
                )
            elif result.pending_approval:
                self.logger.info(
                    f"Enrollment pending approval: {result.approval_message}"
                )
            else:
                self.logger.error(f"Enrollment failed: {result.error_message}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Enrollment failed with exception: {e}")
            return EnrollmentResult(
                success=False,
                subject_name=self.identity.subject_name,
                participant_type=self.identity.participant_type,
                error_message=str(e),
            )

    def _generate_key_and_csr(self):
        """Generate RSA key pair and Certificate Signing Request.
        
        Uses nvflare.lighter.utils functions for consistency with FLARE's
        certificate structure.
        """
        # Generate RSA key pair using lighter utils
        self._private_key, pub_key = generate_keys()
        
        # Build Identity for CSR
        subject = Identity(
            name=self.identity.subject_name,
            org=self.identity.org_name,
            role=self.identity.role,
        )
        
        # Generate CSR using lighter utils
        self._csr = generate_csr(
            subject=subject,
            pri_key=self._private_key,
            additional_hosts=list(self.identity.host_names) if self.identity.host_names else None,
        )
        
        self.logger.debug(
            f"Generated CSR for {self.identity.participant_type}: {self.identity.subject_name}"
            f" (org={self.identity.org_name}, role={self.identity.role})"
        )

    def _get_csr_pem(self) -> bytes:
        """Get CSR in PEM format using lighter utils."""
        return serialize_csr(self._csr)

    def _build_enrollment_request(self) -> EnrollmentRequest:
        """Build enrollment request with all policy-relevant information."""
        return EnrollmentRequest(
            subject_name=self.identity.subject_name,
            participant_type=self.identity.participant_type,
            csr=self._get_csr_pem().decode("utf-8"),
            org_name=self.identity.org_name,
            role=self.identity.role,
            source_ip=self.options.source_ip,
            host_names=self.identity.host_names if self.identity.host_names else None,
            metadata=self.options.metadata,
        )

    def _submit_enrollment(self) -> dict:
        """Submit enrollment request to server via CellNet.
        
        Uses CellNet's send_request with CSR_ENROLLMENT topic.
        Connection uses basic TLS (one-way) or clear text for initial CSR submission.
        """
        request_data = self._build_enrollment_request()
        
        # Build CellNet message
        message = new_cell_message(
            headers={
                "enrollment_token": self.enrollment_token,
            },
            payload=request_data.to_dict(),
        )
        
        # Send via CellNet - handles grpc/https/tcp transparently
        reply: Message = self.cell.send_request(
            channel=CellChannel.SERVER_MAIN,
            topic=CellChannelTopic.CSR_ENROLLMENT,
            target=self.target_fqcn,
            request=message,
            timeout=self.options.timeout,
            secure=False,  # One-way TLS or clear for initial enrollment
        )
        
        if reply is None:
            raise ConnectionError("No response from server")
        
        # Check return code
        return_code = reply.get_header(MessageHeaderKey.RETURN_CODE)
        
        if return_code == ReturnCode.OK:
            return reply.payload
        elif return_code == ReturnCode.UNAUTHENTICATED:
            raise ValueError("Invalid or expired enrollment token")
        elif return_code == ReturnCode.INVALID_REQUEST:
            error = reply.get_header(MessageHeaderKey.ERROR, "Policy rejected enrollment")
            raise ValueError(f"Enrollment rejected: {error}")
        else:
            error = reply.get_header(MessageHeaderKey.ERROR, f"Unknown error: {return_code}")
            raise ValueError(f"Enrollment failed: {error}")

    def _process_response(self, response: dict) -> EnrollmentResult:
        """Process enrollment response and save files if approved."""
        if response.get("pending"):
            return EnrollmentResult(
                success=False,
                subject_name=self.identity.subject_name,
                participant_type=self.identity.participant_type,
                pending_approval=True,
                approval_message=response.get("message"),
            )
        
        signed_cert = response.get("certificate")
        root_ca = response.get("root_ca")
        
        if not signed_cert:
            return EnrollmentResult(
                success=False,
                subject_name=self.identity.subject_name,
                participant_type=self.identity.participant_type,
                error_message="Server response missing certificate",
            )
        
        output_dir = self.options.get_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        
        cert_path = os.path.join(output_dir, self.CERT_FILENAME)
        key_path = os.path.join(output_dir, self.KEY_FILENAME)
        root_ca_path = os.path.join(output_dir, self.ROOT_CA_FILENAME) if root_ca else None
        
        try:
            with open(cert_path, "wb") as f:
                if isinstance(signed_cert, str):
                    f.write(signed_cert.encode("utf-8"))
                else:
                    f.write(signed_cert)
            
            with open(key_path, "wb") as f:
                f.write(serialize_pri_key(self._private_key))
            os.chmod(key_path, 0o600)
            
            if root_ca:
                with open(root_ca_path, "wb") as f:
                    if isinstance(root_ca, str):
                        f.write(root_ca.encode("utf-8"))
                    else:
                        f.write(root_ca)
            
            self.logger.info(f"Saved certificate to: {cert_path}")
            self.logger.info(f"Saved private key to: {key_path}")
            if root_ca_path:
                self.logger.info(f"Saved root CA to: {root_ca_path}")
            
            return EnrollmentResult(
                success=True,
                subject_name=self.identity.subject_name,
                participant_type=self.identity.participant_type,
                cert_path=cert_path,
                key_path=key_path,
                root_ca_path=root_ca_path,
            )
            
        except IOError as e:
            return EnrollmentResult(
                success=False,
                subject_name=self.identity.subject_name,
                participant_type=self.identity.participant_type,
                error_message=f"Failed to save certificates: {e}",
            )

    def get_csr_pem(self) -> Optional[bytes]:
        """Get the generated CSR in PEM format."""
        if self._csr is None:
            return None
        return self._get_csr_pem()

    def get_private_key_pem(self, passphrase: Optional[bytes] = None) -> Optional[bytes]:
        """Get the generated private key in PEM format."""
        if self._private_key is None:
            return None
        return serialize_pri_key(self._private_key, passphrase)
