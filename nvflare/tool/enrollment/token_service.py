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
- Single token generation with embedded policies
- Batch token generation for multiple sites
- Token inspection

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


def _get_cert_cn(cert: x509.Certificate) -> str:
    """Extract Common Name from certificate."""
    for attr in cert.subject:
        if attr.oid == NameOID.COMMON_NAME:
            return attr.value
    return "nvflare"


class TokenService:
    """Manage enrollment token (JWT) generation and inspection.
    
    The enrollment token (FET - FLARE Enrollment Token) is a signed JWT that embeds
    the approval policy. This ensures:
    - Token cannot be tampered (signature verification)
    - Policy is self-contained (no database lookup needed)
    - Lightweight validation (no external auth system required)
    
    This service handles:
    - Token generation (signed with root CA private key)
    - Batch token generation for multiple sites
    - Token inspection (decode without verification)
    
    Token validation and certificate signing are handled by CertService.
    Both services use the same root CA key pair for consistency.
    """

    # JWT algorithm - using RS256 (RSA + SHA256) for asymmetric signing
    JWT_ALGORITHM = "RS256"
    
    def __init__(self, root_ca_path: str):
        """Initialize the token service.
        
        Args:
            root_ca_path: Path to directory containing rootCA.pem and rootCA.key
            
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

    def generate_token(
        self,
        policy: Dict[str, Any],
        subject: str,
        subject_type: str = "site",
        validity: Optional[str] = None,
        **claims,
    ) -> str:
        """Generate a signed enrollment token (JWT) with embedded policy.
        
        Uses a flexible **claims approach - any additional claims are passed
        through to the JWT payload without requiring code changes when
        the policy schema evolves.
        
        Args:
            policy: Approval policy dictionary (from YAML file)
            subject: The subject identifier (site name, user id, or pattern)
            subject_type: Type of subject - "site", "user", or "site_pattern"
            validity: Token validity duration (e.g., "7d", "24h"). 
                      Defaults to policy token.validity or "7d"
            **claims: Additional claims to embed in the token (flexible).
                      Examples: roles, source_ips, metadata, etc.
            
        Returns:
            Signed JWT token string
            
        Examples:
            # Site token
            token = service.generate_token(policy, "hospital-01", "site")
            
            # User token with roles
            token = service.generate_token(
                policy, "user@example.com", "user",
                roles=["researcher", "admin"]
            )
            
            # Site token with optional IP restriction (AWS, GCP, Azure, on-premise)
            token = service.generate_token(
                policy, "dc-server-01", "site",
                source_ips=["10.0.0.0/8", "192.168.0.0/16"]
            )
        """
        if not subject:
            raise ValueError("subject is required")

        # Parse validity duration from argument or policy
        token_config = policy.get("token", {})
        validity_str = validity or token_config.get("validity", "7d")
        validity_delta = self._parse_duration(validity_str)

        now = datetime.now(timezone.utc)
        
        # Build JWT payload with standard claims
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
            "max_uses": 1,                       # Single-use token
        }

        # Merge any additional claims (flexible - no code changes needed)
        for key, value in claims.items():
            if value is not None:
                payload[key] = value

        # Sign and encode the token
        signed_token = jwt.encode(
            payload,
            self.signing_key,
            algorithm=self.JWT_ALGORITHM,  # RS256 = RSA signature with SHA-256
        )
        
        self.logger.info(f"Generated enrollment token: jti={payload['jti']}, subject={subject}, type={subject_type}")
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
        site_name: str,
        validity: Optional[str] = None,
        **claims,
    ) -> str:
        """Convenience method for generating site enrollment tokens.
        
        Args:
            policy_file: Path to policy YAML file
            site_name: Site identifier
            validity: Token validity duration
            **claims: Additional claims (source_ips, metadata, etc.)
            
        Returns:
            Signed JWT token string
        """
        return self.generate_token_from_file(
            policy_file,
            subject=site_name,
            subject_type="site",
            validity=validity,
            **claims,
        )

    def generate_user_token(
        self,
        policy_file: str,
        user_id: str,
        validity: Optional[str] = None,
        **claims,
    ) -> str:
        """Convenience method for generating user enrollment tokens.
        
        Args:
            policy_file: Path to policy YAML file
            user_id: User identifier (email)
            validity: Token validity duration
            **claims: Additional claims (roles, site_id, metadata, etc.)
            
        Returns:
            Signed JWT token string
        """
        return self.generate_token_from_file(
            policy_file,
            subject=user_id,
            subject_type="user",
            validity=validity,
            **claims,
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
        
        return self.signing_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    # =========================================================================
    # Batch Token Generation
    # =========================================================================

    def batch_generate_tokens(
        self,
        policy_file: str,
        count: int = 0,
        site_prefix: Optional[str] = None,
        site_names: Optional[List[str]] = None,
        validity: Optional[str] = None,
        output_file: Optional[str] = None,
        **claims,
    ) -> List[Dict[str, str]]:
        """Generate multiple enrollment tokens in batch.
        
        This is useful for pre-generating tokens for multiple sites.
        Each token is single-use (max_uses=1).
        
        Args:
            policy_file: Path to policy YAML file
            count: Number of tokens to generate (ignored if site_names provided)
            site_prefix: Prefix for auto-generated site names (e.g., "site" -> "site-001")
            site_names: Explicit list of site names (overrides count and site_prefix)
            validity: Token validity duration (e.g., "7d", "24h")
            output_file: Optional file path to save tokens (CSV or TXT format)
            **claims: Additional claims to embed in each token (source_ips, metadata, etc.)
            
        Returns:
            List of dicts with "site_name" and "token" keys
            
        Example:
            # Generate 100 tokens with auto-generated site names
            tokens = service.batch_generate_tokens(
                "policy.yaml",
                count=100,
                site_prefix="hospital",
                validity="30d"
            )
            # Result: [{"site_name": "hospital-001", "token": "eyJ..."}, ...]
            
            # Generate tokens for specific sites with metadata
            tokens = service.batch_generate_tokens(
                "policy.yaml",
                site_names=["site-a", "site-b", "site-c"],
                metadata={"region": "us-west"}
            )
        """
        with open(policy_file, "r") as f:
            policy = yaml.safe_load(f)
        
        # Determine site names
        if site_names:
            names = site_names
        else:
            if not site_prefix:
                site_prefix = "site"
            # Generate numbered names with zero-padding
            padding = max(len(str(count)), 3)  # At least 3 digits
            names = [f"{site_prefix}-{str(i+1).zfill(padding)}" for i in range(count)]
        
        results = []
        for site_name in names:
            token = self.generate_token(
                policy=policy,
                subject=site_name,
                subject_type="site",
                validity=validity,
                **claims,
            )
            results.append({
                "site_name": site_name,
                "token": token,
            })
        
        # Optionally save to file
        if output_file:
            self._save_tokens_to_file(results, output_file)
            self.logger.info(f"Saved {len(results)} tokens to {output_file}")
        
        self.logger.info(f"Generated {len(results)} enrollment tokens")
        return results

    def _save_tokens_to_file(
        self, 
        tokens: List[Dict[str, str]], 
        output_file: str
    ) -> None:
        """Save generated tokens to a file.
        
        Args:
            tokens: List of token dicts with site_name and token
            output_file: Output file path (.csv or .txt)
        """
        ext = os.path.splitext(output_file)[1].lower()
        
        with open(output_file, "w") as f:
            if ext == ".csv":
                f.write("site_name,token\n")
                for t in tokens:
                    f.write(f"{t['site_name']},{t['token']}\n")
            else:
                # Simple text format - one token per line with site name
                for t in tokens:
                    f.write(f"{t['site_name']}: {t['token']}\n")

    def get_token_info(self, jwt_token: str) -> Dict[str, Any]:
        """Get token information without full validation (for inspection).
        
        Args:
            jwt_token: The JWT token string
            
        Returns:
            Dictionary with token details (from embedded claims only)
        """
        try:
            # Decode without verification to inspect claims
            payload = jwt.decode(jwt_token, options={"verify_signature": False})
            
            return {
                "token_id": payload.get("jti"),
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
                "roles": payload.get("roles"),
                "policy_project": payload.get("policy", {}).get("metadata", {}).get("project"),
            }
        except Exception as e:
            raise ValueError(f"Failed to decode token: {e}")

