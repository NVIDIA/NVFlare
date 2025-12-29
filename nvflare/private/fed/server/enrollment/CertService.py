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
Token state storage is handled by pluggable backends (see token_store.py).
"""

import fnmatch
import ipaddress
import logging
import os
import re
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
from nvflare.private.fed.server.enrollment.token_store import (
    TokenStateStore,
    create_token_store,
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
    subject: str                          # site_id or user_id
    subject_type: str                     # "site" or "user"
    policy: Dict[str, Any]                # Embedded policy
    issued_at: datetime
    expires_at: datetime
    site_pattern: Optional[str] = None    # For multi-site tokens
    ip_pattern: Optional[str] = None      # For IP-based matching
    source_ips: Optional[List[str]] = None
    max_uses: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    roles: Optional[List[str]] = None     # For user tokens


@dataclass
class ApprovalResult:
    """Result of policy evaluation."""
    action: ApprovalAction
    rule_name: str
    message: Optional[str] = None
    notify: Optional[List[str]] = None


@dataclass
class EnrollmentContext:
    """Context for enrollment request evaluation."""
    site_name: str
    source_ip: str
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    current_site_count: int = 0


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
        token_store: Optional[TokenStateStore] = None,
        store_backend: str = "sqlite",
        store_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the certificate service.
        
        Args:
            root_ca_path: Path to directory containing rootCA.pem and rootCA.key
            verification_key_path: Path to public key for JWT verification 
                                   (if different from root CA)
            token_store: Pre-configured TokenStateStore instance (takes precedence)
            store_backend: Storage backend type: "sqlite" (default) or "postgres"
            store_config: Backend-specific configuration dict
            
        Examples:
            # Default (SQLite)
            service = CertService(root_ca_path="/path/to/ca")
            
            # SQLite with custom path
            service = CertService(
                root_ca_path="/path/to/ca",
                store_config={"db_path": "/path/to/tokens.db"}
            )
            
            # Production with PostgreSQL
            service = CertService(
                root_ca_path="/path/to/ca",
                store_backend="postgres",
                store_config={
                    "host": "db.example.com",
                    "database": "nvflare",
                    "user": "nvflare",
                    "password": "secret"
                }
            )
            
            # Or inject your own store
            store = PostgresTokenStateStore(...)
            service = CertService(root_ca_path="/path/to/ca", token_store=store)
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
        
        # Token state storage - use provided store or create from config
        if token_store:
            self.token_store = token_store
        else:
            store_config = store_config or {}
            
            # Default config for sqlite backend
            if store_backend == "sqlite" and "db_path" not in store_config:
                store_config["db_path"] = os.path.join(root_ca_path, "token_state.db")
            
            self.token_store = create_token_store(store_backend, **store_config)
        
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
        
        # Check if token is revoked
        if self.token_store.is_revoked(token_id):
            raise ValueError("Token has been revoked")
        
        # Check usage count
        max_uses = payload.get("max_uses", 1)
        if max_uses > 0:  # -1 means unlimited
            current_uses = self.token_store.get_use_count(token_id)
            if current_uses >= max_uses:
                raise ValueError(f"Token has exceeded maximum uses ({max_uses})")

        # Build token payload object
        token_payload = TokenPayload(
            token_id=token_id,
            subject=payload["sub"],
            subject_type=payload.get("subject_type", "site"),
            policy=payload["policy"],
            issued_at=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
            expires_at=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
            site_pattern=payload.get("site_pattern"),
            ip_pattern=payload.get("ip_pattern"),
            source_ips=payload.get("source_ips"),
            max_uses=max_uses,
            metadata=payload.get("metadata", {}),
            roles=payload.get("roles"),
        )

        # Validate against embedded policy if context provided
        if context:
            self._validate_token_for_context(token_payload, context)

        self.logger.info(f"Token validated: jti={token_id}, subject={payload['sub']}")
        return token_payload

    def _validate_token_for_context(self, token: TokenPayload, context: EnrollmentContext):
        """Validate token is applicable to the enrollment context."""
        # Check site name matches pattern (if pattern specified)
        if token.site_pattern:
            if not fnmatch.fnmatch(context.site_name, token.site_pattern):
                raise ValueError(
                    f"Site name '{context.site_name}' does not match token pattern '{token.site_pattern}'"
                )
        elif token.subject_type == "site" and token.subject != context.site_name:
            raise ValueError(
                f"Site name '{context.site_name}' does not match token subject '{token.subject}'"
            )

        # Check source IP against token constraints
        if token.source_ips:
            if not self._ip_in_ranges(context.source_ip, token.source_ips):
                raise ValueError(f"Source IP {context.source_ip} not in allowed ranges")

        if token.ip_pattern:
            if not fnmatch.fnmatch(context.source_ip, token.ip_pattern):
                raise ValueError(f"Source IP {context.source_ip} does not match pattern")

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

                return ApprovalResult(
                    action=action,
                    rule_name=rule_name,
                    message=rule.get("message"),
                    notify=rule.get("notify"),
                )

        # No rule matched - default deny
        return ApprovalResult(
            action=ApprovalAction.REJECT,
            rule_name="default-deny",
            message="No matching rule found"
        )

    def _evaluate_match(self, conditions: Dict[str, Any], context: EnrollmentContext) -> bool:
        """Evaluate if context matches all conditions."""
        # Empty match = matches everything
        if not conditions:
            return True

        # Check source IPs
        if "source_ips" in conditions:
            if not self._ip_in_ranges(context.source_ip, conditions["source_ips"]):
                return False

        # Check site name pattern
        if "site_name_pattern" in conditions:
            pattern = conditions["site_name_pattern"]
            if not fnmatch.fnmatch(context.site_name, pattern):
                return False

        # Check current site count
        if "current_site_count" in conditions:
            count_condition = conditions["current_site_count"]
            if not self._evaluate_count_condition(context.current_site_count, count_condition):
                return False

        # Check time window (if specified)
        if "time_window" in conditions:
            if not self._evaluate_time_window(conditions["time_window"]):
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

    def _evaluate_count_condition(self, current: int, condition: str) -> bool:
        """Evaluate count condition like '<50' or '<=100'."""
        match = re.match(r"^([<>=]+)(\d+)$", condition)
        if not match:
            return True  # Invalid condition, ignore
        
        op, value = match.groups()
        value = int(value)
        
        if op == "<":
            return current < value
        elif op == "<=":
            return current <= value
        elif op == ">":
            return current > value
        elif op == ">=":
            return current >= value
        elif op == "==" or op == "=":
            return current == value
        return True

    def _evaluate_time_window(self, window: Dict[str, Any]) -> bool:
        """Evaluate if current time is within the specified window."""
        now = datetime.now()
        
        # Check day of week
        if "days" in window:
            day_names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
            current_day = day_names[now.weekday()]
            if current_day not in [d.lower() for d in window["days"]]:
                return False

        # Check hours
        if "hours" in window:
            for hour_range in window["hours"]:
                if "-" in hour_range:
                    start, end = hour_range.split("-")
                    start_h, start_m = map(int, start.split(":"))
                    end_h, end_m = map(int, end.split(":"))
                    
                    current_minutes = now.hour * 60 + now.minute
                    start_minutes = start_h * 60 + start_m
                    end_minutes = end_h * 60 + end_m
                    
                    if start_minutes <= current_minutes <= end_minutes:
                        return True
            return False

        return True

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
        
        # Record token usage
        self.token_store.record_use(token_payload.token_id, context.site_name)
        
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

