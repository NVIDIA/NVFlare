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

"""FLARE Enrollment Certificate Service.

This module handles:
- JWT enrollment token validation
- Policy-based approval evaluation
- CSR signing and certificate issuance
- CellNet message handling for CSR_ENROLLMENT topic

Token generation is handled separately by TokenService.
"""

import fnmatch
import ipaddress
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import jwt
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.x509.oid import NameOID

from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.fuel.f3.message import Message
from nvflare.lighter.constants import (
    DEFINED_PARTICIPANT_TYPES,
    DEFINED_ROLES,
    ParticipantType,
)
from nvflare.lighter.utils import (
    Identity,
    generate_cert,
    load_crt,
    load_private_key_file,
    serialize_cert,
)


def _get_cert_cn(cert: x509.Certificate) -> str:
    """Extract Common Name from certificate."""
    for attr in cert.subject:
        if attr.oid == NameOID.COMMON_NAME:
            return attr.value
    return "unknown"


class ApprovalAction(str, Enum):
    """Enrollment approval actions."""
    APPROVE = "approve"
    REJECT = "reject"
    PENDING = "pending"


@dataclass
class TokenPayload:
    """Decoded enrollment token payload."""

    token_id: str
    subject: str
    subject_type: str
    policy: Dict[str, Any]
    issued_at: datetime
    expires_at: datetime
    issuer: str
    roles: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ApprovalResult:
    """Result of policy evaluation."""

    action: ApprovalAction
    rule_name: str
    description: Optional[str] = None
    message: Optional[str] = None
    notify: Optional[List[str]] = None


@dataclass
class EnrollmentContext:
    """Context for enrollment request evaluation."""

    name: str
    participant_type: str = ParticipantType.CLIENT
    role: Optional[str] = None
    source_ip: Optional[str] = None


class CertService:
    """Handle enrollment token validation, policy approval, and CSR signing.
    
    This service is responsible for:
    - Validating JWT enrollment tokens (signature, expiration, revocation)
    - Evaluating embedded policies against enrollment context
    - Signing CSRs and issuing certificates for approved enrollments
    - Handling CellNet messages for CSR_ENROLLMENT topic
    
    Token generation is handled by TokenService (separation of concerns).
    """

    # JWT algorithm - must match TokenService
    JWT_ALGORITHM = "RS256"
    
    def __init__(
        self,
        root_ca_cert_path: str,
        root_ca_key_path: str,
        verification_key_path: Optional[str] = None,
    ):
        """Initialize the certificate service.
        
        Args:
            root_ca_cert_path: Path to root CA certificate file (rootCA.pem)
            root_ca_key_path: Path to root CA private key file (rootCA.key)
            verification_key_path: Path to public key for JWT verification 
                                   (if different from root CA)
            
        Example:
            service = CertService(
                root_ca_cert_path="/path/to/rootCA.pem",
                root_ca_key_path="/path/to/rootCA.key",
            )
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load root CA certificate and private key using lighter utils
        self.root_cert = load_crt(root_ca_cert_path)
        self.root_key = load_private_key_file(root_ca_key_path)
        
        # JWT verification key
        if verification_key_path:
            self.jwt_public_key = self._load_public_key(verification_key_path)
        else:
            # Default: use root CA key for verification
            self.jwt_public_key = self.root_key.public_key()
        
        # Extract issuer from root cert
        self.issuer = _get_cert_cn(self.root_cert)

    def _load_public_key(self, key_path: str):
        """Load public key for JWT verification."""
        from cryptography.hazmat.primitives import serialization
        
        with open(key_path, "rb") as f:
            return serialization.load_pem_public_key(f.read(), backend=default_backend())

    # =========================================================================
    # CellNet Message Handler
    # =========================================================================

    def handle_csr_enrollment(self, request: Message) -> Message:
        """Handle CSR enrollment request from CellNet.
        
        This method should be registered with CellNet for CSR_ENROLLMENT topic.
        
        Args:
            request: CellNet message with enrollment token in header and CSR in payload
            
        Returns:
            CellNet message with signed certificate or error
        """
        try:
            # Extract headers
            enrollment_token = request.get_header("enrollment_token")
            identity_name = request.get_header("identity")
            participant_type = request.get_header("participant_type", ParticipantType.CLIENT)
            role = request.get_header("role")
            source_ip = request.get_header("source_ip")
            
            if not enrollment_token:
                return self._error_response("Missing enrollment token", ReturnCode.UNAUTHENTICATED)
            
            if not identity_name:
                return self._error_response("Missing identity", ReturnCode.INVALID_REQUEST)
            
            # Validate participant type
            if participant_type not in DEFINED_PARTICIPANT_TYPES:
                return self._error_response(
                    f"Invalid participant_type: {participant_type}", 
                    ReturnCode.INVALID_REQUEST
                )
            
            # Validate role if provided
            if role and role not in DEFINED_ROLES:
                return self._error_response(
                    f"Invalid role: {role}",
                    ReturnCode.INVALID_REQUEST
                )
            
            # Get CSR data from payload
            csr_data = request.payload
            if not csr_data:
                return self._error_response("Missing CSR data", ReturnCode.INVALID_REQUEST)
            
            # Ensure CSR is bytes
            if isinstance(csr_data, str):
                csr_data = csr_data.encode("utf-8")
            
            # Build enrollment context
            context = EnrollmentContext(
                name=identity_name,
                participant_type=participant_type,
                role=role,
                source_ip=source_ip,
            )
            
            # Sign CSR (validates token and evaluates policy)
            signed_cert = self.sign_csr(
                csr_data=csr_data,
                token=enrollment_token,
                context=context,
            )
            
            # Return success with signed certificate
            return new_cell_message(
                headers={
                    MessageHeaderKey.RETURN_CODE: ReturnCode.OK,
                },
                payload={
                    "certificate": signed_cert.decode("utf-8"),
                    "root_ca": serialize_cert(self.root_cert).decode("utf-8"),
                }
            )
            
        except ValueError as e:
            self.logger.warning(f"Enrollment rejected: {e}")
            return self._error_response(str(e), ReturnCode.INVALID_REQUEST)
        except Exception as e:
            self.logger.error(f"Enrollment error: {e}")
            return self._error_response(f"Internal error: {e}", ReturnCode.PROCESS_EXCEPTION)

    def _error_response(self, error_msg: str, return_code: str) -> Message:
        """Create error response message."""
        return new_cell_message(
            headers={
                MessageHeaderKey.RETURN_CODE: return_code,
                MessageHeaderKey.ERROR: error_msg,
            },
            payload=None
        )

    # =========================================================================
    # Token Validation
    # =========================================================================

    def validate_token(self, jwt_token: str, context: Optional[EnrollmentContext] = None) -> TokenPayload:
        """Validate JWT enrollment token and return decoded payload.
        
        Args:
            jwt_token: The JWT token string
            context: Optional enrollment context for policy evaluation
            
        Returns:
            TokenPayload with decoded token data
            
        Raises:
            ValueError: If token is invalid, expired, revoked, or policy check fails
        """
        try:
            # Decode and verify signature
            payload = jwt.decode(
                jwt_token,
                self.jwt_public_key,
                algorithms=[self.JWT_ALGORITHM],
                options={"require": ["exp", "sub", "jti", "policy"]}
            )
        except jwt.ExpiredSignatureError:
            raise ValueError("Enrollment token has expired")
        except jwt.InvalidSignatureError:
            raise ValueError("Invalid token signature - possible tampering detected")
        except jwt.DecodeError as e:
            raise ValueError(f"Malformed token: {e}")
        except jwt.MissingRequiredClaimError as e:
            raise ValueError(f"Token missing required claim: {e}")

        token_id = payload["jti"]

        # Build token payload object
        token_payload = TokenPayload(
            token_id=token_id,
            subject=payload["sub"],
            subject_type=payload.get("subject_type", ParticipantType.CLIENT),
            policy=payload["policy"],
            issued_at=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
            expires_at=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
            issuer=payload.get("iss", "unknown"),
            roles=payload.get("roles"),
            metadata=payload.get("metadata", {}),
        )

        # Validate against embedded policy if context provided
        if context:
            self._validate_token_for_context(token_payload, context)

        self.logger.info(f"Token validated: jti={token_id}, subject={payload['sub']}")
        return token_payload

    def _validate_token_for_context(self, token: TokenPayload, context: EnrollmentContext):
        """Validate token is applicable to the enrollment context.
        
        Checks name matches token subject (exact or pattern).
        """
        # Check name matches token subject
        if token.subject_type == "pattern":
            # Pattern matching (e.g., "hospital-*")
            if not fnmatch.fnmatch(context.name, token.subject):
                raise ValueError(
                    f"Name '{context.name}' does not match token pattern '{token.subject}'"
                )
        elif token.subject_type in DEFINED_PARTICIPANT_TYPES:
            # Exact match required
            if token.subject != context.name:
                raise ValueError(
                    f"Name '{context.name}' does not match token subject '{token.subject}'"
                )

    # =========================================================================
    # Policy Evaluation
    # =========================================================================

    def evaluate_policy(self, token: TokenPayload, context: EnrollmentContext) -> ApprovalResult:
        """Evaluate embedded policy against enrollment context.
        
        Args:
            token: Validated token payload
            context: Enrollment request context
            
        Returns:
            ApprovalResult with action and rule details
        """
        policy = token.policy
        approval_config = policy.get("approval", {})
        rules = approval_config.get("rules", [])

        # Check admin role constraints if applicable
        if context.participant_type == ParticipantType.ADMIN and context.role:
            user_policy = policy.get("user", {})
            allowed_roles = user_policy.get("allowed_roles", [])
            if allowed_roles and context.role not in allowed_roles:
                return ApprovalResult(
                    action=ApprovalAction.REJECT,
                    rule_name="role-constraint",
                    message=f"Role '{context.role}' not in allowed roles: {allowed_roles}"
                )

        # If no rules, default to approve
        if not rules:
            return ApprovalResult(
                action=ApprovalAction.APPROVE,
                rule_name="default",
                message="No rules defined, defaulting to approve"
            )

        # Evaluate rules in order - first match wins
        for rule in rules:
            rule_name = rule.get("name", "unnamed")
            match_conditions = rule.get("match", {})
            
            if self._evaluate_match(match_conditions, context):
                action_str = rule.get("action", "reject")
                try:
                    action = ApprovalAction(action_str)
                except ValueError:
                    self.logger.warning(f"Unknown action '{action_str}' in rule '{rule_name}', defaulting to reject")
                    action = ApprovalAction.REJECT

                result = ApprovalResult(
                    action=action,
                    rule_name=rule_name,
                    description=rule.get("description"),
                    message=rule.get("message"),
                    notify=rule.get("notify"),
                )
                
                # Always log policy matches (audit logging always enabled)
                ip_info = f", ip={context.source_ip}" if context.source_ip else ""
                self.logger.info(
                    f"Policy rule matched: {rule_name} -> {action.value} "
                    f"(name={context.name}, type={context.participant_type}{ip_info})"
                )
                
                return result

        # No rule matched - default deny
        return ApprovalResult(
            action=ApprovalAction.REJECT,
            rule_name="default-deny",
            message="No matching rule found"
        )

    def _evaluate_match(self, conditions: Dict[str, Any], context: EnrollmentContext) -> bool:
        """Evaluate if context matches all conditions.
        
        Match conditions:
        - site_name_pattern: Wildcard pattern matching (e.g., "hospital-*")
        - source_ips: CIDR ranges (e.g., ["10.0.0.0/8"])
                      If specified, client MUST provide source_ip
        """
        # Empty match = matches everything
        if not conditions:
            return True

        # Check site name pattern
        if "site_name_pattern" in conditions:
            pattern = conditions["site_name_pattern"]
            if not fnmatch.fnmatch(context.name, pattern):
                return False

        # Check source IPs - if policy specifies IPs, client must provide IP
        if "source_ips" in conditions:
            if not context.source_ip:
                # Policy requires IP but client didn't provide one - reject
                self.logger.warning(
                    f"Policy requires source_ip but client '{context.name}' did not provide one"
                )
                return False
            if not self._ip_in_ranges(context.source_ip, conditions["source_ips"]):
                return False

        # All conditions matched
        return True

    def _ip_in_ranges(self, ip_str: str, ranges: List[str]) -> bool:
        """Check if IP address is in any of the specified CIDR ranges."""
        try:
            ip = ipaddress.ip_address(ip_str)
            for range_str in ranges:
                try:
                    network = ipaddress.ip_network(range_str, strict=False)
                    if ip in network:
                        return True
                except ValueError:
                    continue
            return False
        except ValueError:
            return False

    # =========================================================================
    # Certificate Signing
    # =========================================================================

    def sign_csr(
        self,
        csr_data: bytes,
        token: str,
        context: EnrollmentContext,
        valid_days: int = 365,
    ) -> bytes:
        """Sign CSR using root CA after token validation and policy evaluation.
        
        Args:
            csr_data: PEM-encoded CSR data
            token: JWT enrollment token
            context: Enrollment context (name, participant type, role, source IP)
            valid_days: Certificate validity in days
            
        Returns:
            PEM-encoded signed certificate
            
        Raises:
            ValueError: If token is invalid or policy rejects enrollment
        """
        # Validate token
        token_payload = self.validate_token(token, context)
        
        # Evaluate policy
        result = self.evaluate_policy(token_payload, context)
        
        if result.action == ApprovalAction.REJECT:
            raise ValueError(f"Enrollment rejected by rule '{result.rule_name}': {result.message}")
        
        if result.action == ApprovalAction.PENDING:
            raise ValueError(
                f"Enrollment requires manual approval (rule: '{result.rule_name}'). "
                f"Notify: {result.notify}"
            )

        # Approved - sign the certificate
        cert = self._sign_certificate(csr_data, context.name, valid_days)
        
        self.logger.info(
            f"Certificate issued for '{context.name}' ({context.participant_type}) "
            f"(token: {token_payload.token_id}, rule: {result.rule_name})"
        )
        
        return serialize_cert(cert)

    def _sign_certificate(
        self,
        csr_data: bytes,
        subject_name: str,
        valid_days: int = 365,
    ) -> x509.Certificate:
        """Sign a CSR and return the certificate."""
        # Parse CSR
        csr = x509.load_pem_x509_csr(csr_data, default_backend())
        
        # Verify CSR signature
        if not csr.is_signature_valid:
            raise ValueError("CSR signature is invalid")

        # Extract subject from CSR or use provided name
        subject_cn = None
        for attr in csr.subject:
            if attr.oid == NameOID.COMMON_NAME:
                subject_cn = attr.value
                break
        
        if not subject_cn:
            subject_cn = subject_name

        # Build certificate using existing utility
        cert = generate_cert(
            subject=Identity(subject_cn),
            issuer=Identity(self.issuer),
            signing_pri_key=self.root_key,
            subject_pub_key=csr.public_key(),
            valid_days=valid_days,
        )
        
        return cert
