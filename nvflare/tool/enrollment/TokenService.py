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

"""FLARE Enrollment Token (FET) Service.

This module handles JWT-based enrollment token:
- Generation with embedded policies
- Revocation
- Inspection

Token generation is separate from certificate services to maintain separation of concerns.
"""

import logging
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import jwt
import yaml
from cryptography import x509
from cryptography.x509.oid import NameOID

from nvflare.lighter.utils import load_crt, load_private_key_file
from nvflare.private.fed.server.enrollment.token_store import (
    TokenStateStore,
    create_token_store,
)


def _get_cert_cn(cert: x509.Certificate) -> str:
    """Extract Common Name from certificate."""
    for attr in cert.subject:
        if attr.oid == NameOID.COMMON_NAME:
            return attr.value
    return "nvflare"


class TokenService:
    """Manage enrollment tokens (JWT) - generation, revocation, and inspection.
    
    The enrollment token (FET - FLARE Enrollment Token) is a signed JWT that embeds
    the approval policy. This ensures:
    - Token cannot be tampered (signature verification)
    - Policy is self-contained (no database lookup needed)
    - Lightweight validation (no external auth system required)
    
    This service handles:
    - Token generation (signed with root CA private key)
    - Token revocation
    - Token inspection
    
    Token validation and certificate signing are handled by CertService.
    Both services use the same root CA key pair for consistency.
    """

    # JWT algorithm - using RS256 (RSA + SHA256) for asymmetric signing
    JWT_ALGORITHM = "RS256"
    
    def __init__(
        self,
        root_ca_path: str,
        token_store: Optional[TokenStateStore] = None,
        store_backend: str = "sqlite",
        store_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the token service.
        
        Args:
            root_ca_path: Path to directory containing rootCA.pem and rootCA.key
            token_store: Pre-configured TokenStateStore instance
            store_backend: Storage backend type: "sqlite" (default) or "postgres"
            store_config: Backend-specific configuration dict
            
        Note:
            Uses the same root CA as CertService for consistent key pair.
            TokenService signs with private key, CertService verifies with public key.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load root CA certificate and private key using lighter utils
        cert_file = os.path.join(root_ca_path, "rootCA.pem")
        key_file = os.path.join(root_ca_path, "rootCA.key")
        
        self.root_cert = load_crt(cert_file)
        self.signing_key = load_private_key_file(key_file)
        self.issuer = _get_cert_cn(self.root_cert)
        
        # Token state storage for revocation and inspection
        if token_store:
            self.token_store = token_store
        else:
            store_config = store_config or {}
            if store_backend == "sqlite" and "db_path" not in store_config:
                store_config["db_path"] = os.path.join(root_ca_path, "token_state.db")
            self.token_store = create_token_store(store_backend, **store_config)

    def generate_token(
        self,
        policy: Dict[str, Any],
        site_id: Optional[str] = None,
        site_pattern: Optional[str] = None,
        user_id: Optional[str] = None,
        roles: Optional[List[str]] = None,
        validity: Optional[str] = None,
        max_uses: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source_ips: Optional[List[str]] = None,
        ip_pattern: Optional[str] = None,
    ) -> str:
        """Generate a signed enrollment token (JWT) with embedded policy.
        
        Args:
            policy: Approval policy dictionary (from YAML file)
            site_id: Specific site identifier
            site_pattern: Pattern for multiple sites (e.g., "hospital-*")
            user_id: User identifier (for user tokens)
            roles: List of roles for user token
            validity: Token validity duration (e.g., "7d", "24h")
            max_uses: Maximum number of times token can be used
            metadata: Additional metadata to embed
            source_ips: List of allowed source IP ranges
            ip_pattern: IP pattern for matching
            
        Returns:
            Signed JWT token string
        """
        # Determine subject and type
        if user_id:
            subject = user_id
            subject_type = "user"
        elif site_id:
            subject = site_id
            subject_type = "site"
        elif site_pattern:
            subject = site_pattern
            subject_type = "site_pattern"
        else:
            raise ValueError("Must specify site_id, site_pattern, or user_id")

        # Parse validity duration
        token_config = policy.get("token", {})
        validity_str = validity or token_config.get("validity", "7d")
        validity_delta = self._parse_duration(validity_str)
        
        # Determine max uses
        if max_uses is None:
            max_uses = token_config.get("max_uses", 1)

        now = datetime.now(timezone.utc)
        
        # Build JWT payload
        payload = {
            # Standard JWT claims
            "jti": str(uuid.uuid4()),           # Unique token ID
            "sub": subject,                      # Subject (site/user)
            "iat": now,                          # Issued at
            "exp": now + validity_delta,         # Expiration
            "iss": self.issuer,                  # Issuer
            
            # FLARE-specific claims
            "subject_type": subject_type,
            "policy": policy,                    # Embedded policy (signed, tamper-proof)
            "max_uses": max_uses,
        }

        # Optional claims
        if site_pattern:
            payload["site_pattern"] = site_pattern
        if ip_pattern:
            payload["ip_pattern"] = ip_pattern
        if source_ips:
            payload["source_ips"] = source_ips
        if roles:
            payload["roles"] = roles
        if metadata:
            payload["metadata"] = metadata

        # Sign and encode the token (jwt.encode signs with the private key)
        signed_token = jwt.encode(
            payload,
            self.signing_key,
            algorithm=self.JWT_ALGORITHM,  # RS256 = RSA signature with SHA-256
        )
        
        self.logger.info(f"Generated signed enrollment token: jti={payload['jti']}, subject={subject}")
        return signed_token

    def generate_token_from_file(self, policy_file: str, **kwargs) -> str:
        """Generate token from a policy YAML file.
        
        Args:
            policy_file: Path to policy YAML file
            **kwargs: Additional arguments passed to generate_token
            
        Returns:
            Signed JWT token string
        """
        with open(policy_file, "r") as f:
            policy = yaml.safe_load(f)
        return self.generate_token(policy=policy, **kwargs)

    def generate_site_token(
        self,
        policy_file: str,
        site_id: Optional[str] = None,
        site_pattern: Optional[str] = None,
        validity: Optional[str] = None,
        max_uses: Optional[int] = None,
        source_ips: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Convenience method for generating site enrollment tokens.
        
        Args:
            policy_file: Path to policy YAML file
            site_id: Specific site identifier
            site_pattern: Pattern for multiple sites
            validity: Token validity duration
            max_uses: Maximum uses
            source_ips: Allowed source IP ranges
            metadata: Additional metadata
            
        Returns:
            Signed JWT token string
        """
        return self.generate_token_from_file(
            policy_file,
            site_id=site_id,
            site_pattern=site_pattern,
            validity=validity,
            max_uses=max_uses,
            source_ips=source_ips,
            metadata=metadata,
        )

    def generate_user_token(
        self,
        policy_file: str,
        user_id: str,
        site_id: Optional[str] = None,
        roles: Optional[List[str]] = None,
        validity: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Convenience method for generating user enrollment tokens.
        
        Args:
            policy_file: Path to policy YAML file
            user_id: User identifier (email)
            site_id: Associated site
            roles: User roles
            validity: Token validity duration
            metadata: Additional metadata
            
        Returns:
            Signed JWT token string
        """
        with open(policy_file, "r") as f:
            policy = yaml.safe_load(f)
        
        # Add site_id to metadata if provided
        if site_id:
            metadata = metadata or {}
            metadata["site_id"] = site_id
        
        return self.generate_token(
            policy=policy,
            user_id=user_id,
            roles=roles,
            validity=validity,
            metadata=metadata,
        )

    def _parse_duration(self, duration_str: str) -> timedelta:
        """Parse duration string like '7d', '24h', '30m' to timedelta."""
        match = re.match(r"^(\d+)([dhms])$", duration_str.lower())
        if not match:
            raise ValueError(f"Invalid duration format: {duration_str}")
        
        value = int(match.group(1))
        unit = match.group(2)
        
        if unit == "d":
            return timedelta(days=value)
        elif unit == "h":
            return timedelta(hours=value)
        elif unit == "m":
            return timedelta(minutes=value)
        elif unit == "s":
            return timedelta(seconds=value)
        else:
            raise ValueError(f"Unknown duration unit: {unit}")

    def get_public_key_pem(self) -> bytes:
        """Get the public key in PEM format for token verification.
        
        This can be shared with CertService for token validation.
        """
        from cryptography.hazmat.primitives import serialization
        
        return self.signing_key.public_key().private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    # =========================================================================
    # Token Management
    # =========================================================================

    def revoke_token(self, token_id: str) -> None:
        """Revoke an enrollment token by its ID.
        
        Args:
            token_id: The unique token identifier (jti claim)
        """
        self.token_store.revoke(token_id)
        self.logger.info(f"Token revoked: {token_id}")

    def revoke_token_from_jwt(self, jwt_token: str) -> None:
        """Revoke an enrollment token by parsing the JWT.
        
        Args:
            jwt_token: The JWT token string
        """
        payload = jwt.decode(jwt_token, options={"verify_signature": False})
        token_id = payload.get("jti")
        if token_id:
            self.revoke_token(token_id)
        else:
            raise ValueError("Token does not contain a valid jti claim")

    def get_token_info(self, jwt_token: str) -> Dict[str, Any]:
        """Get token information without full validation (for inspection).
        
        Args:
            jwt_token: The JWT token string
            
        Returns:
            Dictionary with token details
        """
        try:
            # Decode without verification to inspect claims
            payload = jwt.decode(jwt_token, options={"verify_signature": False})
            
            token_id = payload.get("jti", "unknown")
            
            return {
                "token_id": token_id,
                "subject": payload.get("sub"),
                "subject_type": payload.get("subject_type"),
                "issuer": payload.get("iss"),
                "issued_at": datetime.fromtimestamp(
                    payload.get("iat", 0), tz=timezone.utc
                ).isoformat(),
                "expires_at": datetime.fromtimestamp(
                    payload.get("exp", 0), tz=timezone.utc
                ).isoformat(),
                "max_uses": payload.get("max_uses", 1),
                "current_uses": self.token_store.get_use_count(token_id),
                "is_revoked": self.token_store.is_revoked(token_id),
                "enrolled_entities": self.token_store.get_enrolled_entities(token_id),
                "site_pattern": payload.get("site_pattern"),
                "source_ips": payload.get("source_ips"),
                "roles": payload.get("roles"),
                "policy_project": payload.get("policy", {}).get("project"),
            }
        except Exception as e:
            raise ValueError(f"Failed to decode token: {e}")

    def list_token_usage(self, token_id: str) -> List[str]:
        """Get list of entities that have used this token.
        
        Args:
            token_id: The unique token identifier
            
        Returns:
            List of enrolled entity names
        """
        return self.token_store.get_enrolled_entities(token_id)

