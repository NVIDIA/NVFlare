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

Token generation is handled separately by TokenService.
"""

import fnmatch
import ipaddress
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import jwt
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.x509.oid import NameOID

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
    """Decoded enrollment token payload.
    
    Maps to JWT claims:
    - token_id: jti (unique identifier)
    - subject: sub (site name, user ID, or pattern like "hospital-*")
    - subject_type: "site", "user", or "site_pattern"
    - policy: embedded approval policy (tamper-proof)
    - issued_at/expires_at: iat/exp
    - issuer: iss (from root CA CN)
    """
    token_id: str                         # JWT jti
    subject: str                          # JWT sub (site/user/pattern)
    subject_type: str                     # "site", "user", or "site_pattern"
    policy: Dict[str, Any]                # Embedded policy (signed)
    issued_at: datetime                   # JWT iat
    expires_at: datetime                  # JWT exp
    issuer: str                           # JWT iss
    roles: Optional[List[str]] = None     # For user tokens
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ApprovalResult:
    """Result of policy evaluation."""
    action: ApprovalAction
    rule_name: str
    description: Optional[str] = None     # Rule description
    message: Optional[str] = None         # Custom message
    notify: Optional[List[str]] = None    # Notification recipients


@dataclass
class EnrollmentContext:
    """Context for enrollment request evaluation.
    
    Represents the incoming enrollment request that will be
    matched against policy rules.
    
    source_ip: Required if policy has source_ips rules.
               For static instances (EC2, on-premise), client provides IP.
               For K8s/dynamic, omit source_ips from policy to skip IP check.
    """
    site_name: str                                # Site requesting enrollment
    source_ip: Optional[str] = None               # Required if policy has source_ips


class CertService:
    """Handle enrollment token validation, policy approval, and CSR signing.
    
    This service is responsible for:
    - Validating JWT enrollment tokens (signature, expiration, revocation)
    - Evaluating embedded policies against enrollment context
    - Signing CSRs and issuing certificates for approved enrollments
    
    Token generation is handled by TokenService (separation of concerns).
    """

    # JWT algorithm - must match TokenService
    JWT_ALGORITHM = "RS256"
    
    def __init__(
        self,
        root_ca_path: str,
        verification_key_path: Optional[str] = None,
    ):
        """Initialize the certificate service.
        
        Args:
            root_ca_path: Path to directory containing rootCA.pem and rootCA.key
            verification_key_path: Path to public key for JWT verification 
                                   (if different from root CA)
            
        Example:
            service = CertService(root_ca_path="/path/to/ca")
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load root CA certificate and private key using lighter utils
        cert_file = os.path.join(root_ca_path, "rootCA.pem")
        key_file = os.path.join(root_ca_path, "rootCA.key")
        
        self.root_cert = load_crt(cert_file)
        self.root_key = load_private_key_file(key_file)
        
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
            subject_type=payload.get("subject_type", "site"),
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
        
        Checks site name matches token subject (exact or pattern).
        """
        # Check site name matches token subject
        if token.subject_type == "site_pattern":
            # Pattern matching (e.g., "hospital-*")
            if not fnmatch.fnmatch(context.site_name, token.subject):
                raise ValueError(
                    f"Site name '{context.site_name}' does not match token pattern '{token.subject}'"
                )
        elif token.subject_type == "site":
            # Exact match required
            if token.subject != context.site_name:
                raise ValueError(
                    f"Site name '{context.site_name}' does not match token subject '{token.subject}'"
                )
        # For user tokens, site validation is not required

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
                    f"(site={context.site_name}{ip_info})"
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
            if not fnmatch.fnmatch(context.site_name, pattern):
                return False

        # Check source IPs - if policy specifies IPs, client must provide IP
        if "source_ips" in conditions:
            if not context.source_ip:
                # Policy requires IP but client didn't provide one - reject
                self.logger.warning(
                    f"Policy requires source_ip but client '{context.site_name}' did not provide one"
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
            context: Enrollment context (site name, source IP, etc.)
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
        cert = self._sign_certificate(csr_data, context.site_name, valid_days)
        
        self.logger.info(
            f"Certificate issued for '{context.site_name}' "
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

