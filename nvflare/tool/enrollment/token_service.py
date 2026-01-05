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
- Batch token generation for multiple clients
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

from nvflare.lighter.constants import DEFINED_PARTICIPANT_TYPES, AdminRole, ParticipantType
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
    - Batch token generation for multiple clients
    - Token inspection (decode without verification)

    Token validation and certificate signing are handled by CertService.
    Both services use the same root CA key pair for consistency.
    """

    # JWT algorithm - using RS256 (RSA + SHA256) for asymmetric signing
    JWT_ALGORITHM = "RS256"

    # Subject type for pattern matching (not a participant type)
    SUBJECT_TYPE_PATTERN = "pattern"

    def __init__(
        self,
        root_ca_path: Optional[str] = None,
        root_ca_cert: Optional[Any] = None,
        signing_key: Optional[Any] = None,
        jwt_signing_key_path: Optional[str] = None,
    ):
        """Initialize the token service.

        Can be initialized with either file paths OR pre-loaded objects.
        If both are provided, pre-loaded objects take precedence (avoids re-loading).

        Args:
            root_ca_path: Path to the provisioned workspace directory.
                          This should be the directory created by 'nvflare provision'.
                          The service will look for the root CA in:
                          1. state/cert.json (provisioning state file)
                          2. rootCA.pem and rootCA.key files (fallback)
            root_ca_cert: Pre-loaded root CA certificate object
            signing_key: Pre-loaded signing key object (root CA private key or separate JWT key)
            jwt_signing_key_path: Optional path to a separate private key for JWT signing.
                                  If not provided, uses the root CA private key (default).
                                  If provided, CertService must use the matching public key
                                  for verification (via verification_key_path).

        Note:
            By default, uses the same root CA as CertService for consistent key pair.
            TokenService signs with private key, CertService verifies with public key.
            For separate JWT keys, ensure both services use the same key pair.

        Example (from paths):
            service = TokenService(root_ca_path="/path/to/ca")

        Example (from objects - avoids disk I/O):
            service = TokenService(root_ca_cert=cert, signing_key=key)
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        # Use pre-loaded objects if provided
        if root_ca_cert is not None and signing_key is not None:
            self.root_cert = root_ca_cert
            self.signing_key = signing_key
            self.issuer = _get_cert_cn(self.root_cert)
        elif root_ca_path:
            # Load from paths
            # Try to load from state/cert.json first (created by provisioning)
            state_file = os.path.join(root_ca_path, "state", "cert.json")
            if os.path.exists(state_file):
                self._load_from_state_file(state_file)
            else:
                # Fallback: look for rootCA.pem and rootCA.key directly
                cert_file = os.path.join(root_ca_path, "rootCA.pem")
                key_file = os.path.join(root_ca_path, "rootCA.key")

                if not os.path.exists(cert_file):
                    raise FileNotFoundError(
                        f"Root CA not found. Expected either:\n"
                        f"  - {state_file} (from 'nvflare provision')\n"
                        f"  - {cert_file} and {key_file}"
                    )

                self.root_cert = load_crt(cert_file)
                self.signing_key = load_private_key_file(key_file)
                self.issuer = _get_cert_cn(self.root_cert)
        else:
            raise ValueError("Must provide either root_ca_path or (root_ca_cert, signing_key)")

        # Optional: use separate JWT signing key (overrides signing_key)
        if jwt_signing_key_path:
            self.logger.info(f"Using separate JWT signing key: {jwt_signing_key_path}")
            self.signing_key = load_private_key_file(jwt_signing_key_path)

    def _load_from_state_file(self, state_file: str):
        """Load root CA certificate and key from provisioning state file."""
        import json

        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import serialization
        from cryptography.x509 import load_pem_x509_certificate

        with open(state_file, "r") as f:
            state = json.load(f)

        root_cert_pem = state.get("root_cert")
        root_key_pem = state.get("root_pri_key")

        if not root_cert_pem or not root_key_pem:
            raise ValueError(f"Invalid state file: missing root_cert or root_pri_key in {state_file}")

        self.root_cert = load_pem_x509_certificate(root_cert_pem.encode(), default_backend())
        self.signing_key = serialization.load_pem_private_key(
            root_key_pem.encode(), password=None, backend=default_backend()
        )
        self.issuer = _get_cert_cn(self.root_cert)

    def _load_policy(self, policy_file: str) -> Dict[str, Any]:
        """Load policy from a YAML file.

        Args:
            policy_file: Path to policy YAML file

        Returns:
            Policy dictionary

        Raises:
            FileNotFoundError: If policy file doesn't exist
        """
        if not os.path.exists(policy_file):
            raise FileNotFoundError(f"Policy file not found: {policy_file}")

        with open(policy_file, "r") as f:
            return yaml.safe_load(f) or {}

    def generate_token(
        self,
        policy: Dict[str, Any],
        subject: str,
        subject_type: str = ParticipantType.CLIENT,
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
            subject_type: Type of subject from ParticipantType (client, admin, relay)
                         or "pattern" for wildcard matching
            validity: Token validity duration (e.g., "7d", "24h").
                      Defaults to policy token.validity or "7d"
            **claims: Additional claims to embed in the token (flexible).
                      Examples: roles, source_ips, metadata, etc.

        Returns:
            Signed JWT token string

        Examples:
            # Client (site) token
            token = service.generate_token(policy, "hospital-01", ParticipantType.CLIENT)

            # Admin (user) token with roles
            token = service.generate_token(
                policy, "user@example.com", ParticipantType.ADMIN,
                roles=["org_admin", "lead"]
            )

            # Client token with optional IP restriction (AWS, GCP, Azure, on-premise)
            token = service.generate_token(
                policy, "dc-server-01", ParticipantType.CLIENT,
                source_ips=["10.0.0.0/8", "192.168.0.0/16"]
            )

            # Pattern token for wildcard site names
            token = service.generate_token(
                policy, "hospital-*", TokenService.SUBJECT_TYPE_PATTERN
            )
        """
        if not subject:
            raise ValueError("subject is required")

        # Validate subject_type
        valid_types = list(DEFINED_PARTICIPANT_TYPES) + [self.SUBJECT_TYPE_PATTERN]
        if subject_type not in valid_types:
            raise ValueError(f"subject_type must be one of: {valid_types}")

        # Parse validity duration from argument or policy
        token_config = policy.get("token", {})
        validity_str = validity or token_config.get("validity", "7d")
        validity_delta = self._parse_duration(validity_str)

        now = datetime.now(timezone.utc)

        # Build JWT payload with standard claims
        payload = {
            # Standard JWT claims
            "jti": str(uuid.uuid4()),  # Unique token ID
            "sub": subject,  # Subject (site/user)
            "iat": now,  # Issued at
            "exp": now + validity_delta,  # Expiration
            "iss": self.issuer,  # Issuer
            # FLARE-specific claims
            "subject_type": subject_type,
            "policy": policy,  # Embedded policy (signed, tamper-proof)
            "max_uses": 1,  # Single-use token
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

    def generate_client_token(
        self,
        policy_file: str,
        client_name: str,
        validity: Optional[str] = None,
        **claims,
    ) -> str:
        """Convenience method for generating client (site) enrollment tokens.

        Args:
            policy_file: Path to policy YAML file
            client_name: Client/site identifier
            validity: Token validity duration
            **claims: Additional claims (source_ips, metadata, etc.)

        Returns:
            Signed JWT token string
        """
        return self.generate_token_from_file(
            policy_file,
            subject=client_name,
            subject_type=ParticipantType.CLIENT,
            validity=validity,
            **claims,
        )

    def generate_site_token(
        self,
        site_name: str,
        valid_days: int = 7,
        policy_file: Optional[str] = None,
        **claims,
    ) -> str:
        """Convenience method for generating site enrollment tokens.

        Sites can be clients or servers. This is an alias for generate_client_token
        with simpler parameters.

        Args:
            site_name: Site identifier
            valid_days: Token validity in days (default: 7)
            policy_file: Path to policy YAML file (optional)
            **claims: Additional claims

        Returns:
            Signed JWT token string
        """
        return self.generate_token(
            subject=site_name,
            subject_type=ParticipantType.CLIENT,
            valid_days=valid_days,
            policy=self._load_policy(policy_file) if policy_file else {},
            **claims,
        )

    def generate_admin_token(
        self,
        user_id: str,
        valid_days: int = 7,
        policy_file: Optional[str] = None,
        roles: Optional[list] = None,
        **claims,
    ) -> str:
        """Convenience method for generating admin (user) enrollment tokens.

        Args:
            user_id: User identifier (email)
            valid_days: Token validity in days (default: 7)
            policy_file: Path to policy YAML file (optional)
            roles: List of roles for the admin (default: [AdminRole.LEAD])
            **claims: Additional claims

        Returns:
            Signed JWT token string
        """
        # Default to LEAD role (has job submission permissions)
        return self.generate_token(
            subject=user_id,
            subject_type=ParticipantType.ADMIN,
            valid_days=valid_days,
            policy=self._load_policy(policy_file) if policy_file else {},
            roles=roles or [AdminRole.LEAD],
            **claims,
        )

    def generate_relay_token(
        self,
        relay_name: str,
        valid_days: int = 7,
        policy_file: Optional[str] = None,
        **claims,
    ) -> str:
        """Convenience method for generating relay enrollment tokens.

        Args:
            relay_name: Relay node identifier
            valid_days: Token validity in days (default: 7)
            policy_file: Path to policy YAML file (optional)
            **claims: Additional claims

        Returns:
            Signed JWT token string
        """
        return self.generate_token(
            subject=relay_name,
            subject_type=ParticipantType.RELAY,
            valid_days=valid_days,
            policy=self._load_policy(policy_file) if policy_file else {},
            **claims,
        )

    def generate_admin_token_from_file(
        self,
        policy_file: str,
        user_id: str,
        validity: Optional[str] = None,
        **claims,
    ) -> str:
        """Generate admin token from policy file.

        Args:
            policy_file: Path to policy YAML file
            user_id: User identifier (email)
            validity: Token validity duration
            **claims: Additional claims (roles, metadata, etc.)

        Returns:
            Signed JWT token string
        """
        return self.generate_token_from_file(
            policy_file,
            subject=user_id,
            subject_type=ParticipantType.ADMIN,
            validity=validity,
            **claims,
        )

    def generate_relay_token_from_file(
        self,
        policy_file: str,
        relay_name: str,
        validity: Optional[str] = None,
        **claims,
    ) -> str:
        """Generate relay token from policy file.

        Args:
            policy_file: Path to policy YAML file
            relay_name: Relay node identifier
            validity: Token validity duration
            **claims: Additional claims (source_ips, metadata, etc.)

        Returns:
            Signed JWT token string
        """
        return self.generate_token_from_file(
            policy_file,
            subject=relay_name,
            subject_type=ParticipantType.RELAY,
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
        name_prefix: Optional[str] = None,
        names: Optional[List[str]] = None,
        subject_type: str = ParticipantType.CLIENT,
        validity: Optional[str] = None,
        output_file: Optional[str] = None,
        **claims,
    ) -> List[Dict[str, str]]:
        """Generate multiple enrollment tokens in batch.

        This is useful for pre-generating tokens for multiple clients.
        Each token is single-use (max_uses=1).

        Args:
            policy_file: Path to policy YAML file
            count: Number of tokens to generate (ignored if names provided)
            name_prefix: Prefix for auto-generated names (e.g., "site" -> "site-001")
            names: Explicit list of names (overrides count and name_prefix)
            subject_type: Type of participant (client, admin, relay)
            validity: Token validity duration (e.g., "7d", "24h")
            output_file: Optional file path to save tokens (CSV or TXT format)
            **claims: Additional claims to embed in each token (source_ips, metadata, etc.)

        Returns:
            List of dicts with "name" and "token" keys

        Example:
            # Generate 100 client tokens with auto-generated names
            tokens = service.batch_generate_tokens(
                "policy.yaml",
                count=100,
                name_prefix="hospital",
                validity="30d"
            )
            # Result: [{"name": "hospital-001", "token": "eyJ..."}, ...]

            # Generate tokens for specific clients with metadata
            tokens = service.batch_generate_tokens(
                "policy.yaml",
                names=["site-a", "site-b", "site-c"],
                metadata={"region": "us-west"}
            )
        """
        with open(policy_file, "r") as f:
            policy = yaml.safe_load(f)

        # Determine names
        if names:
            name_list = names
        else:
            if not name_prefix:
                name_prefix = "client"
            # Generate numbered names with zero-padding
            padding = max(len(str(count)), 3)  # At least 3 digits
            name_list = [f"{name_prefix}-{str(i+1).zfill(padding)}" for i in range(count)]

        results = []
        for name in name_list:
            token = self.generate_token(
                policy=policy,
                subject=name,
                subject_type=subject_type,
                validity=validity,
                **claims,
            )
            results.append(
                {
                    "name": name,
                    "token": token,
                }
            )

        # Optionally save to file
        if output_file:
            self._save_tokens_to_file(results, output_file)
            self.logger.info(f"Saved {len(results)} tokens to {output_file}")

        self.logger.info(f"Generated {len(results)} enrollment tokens")
        return results

    def _save_tokens_to_file(self, tokens: List[Dict[str, str]], output_file: str) -> None:
        """Save generated tokens to a file.

        Args:
            tokens: List of token dicts with name and token
            output_file: Output file path (.csv or .txt)
        """
        ext = os.path.splitext(output_file)[1].lower()

        with open(output_file, "w") as f:
            if ext == ".csv":
                f.write("name,token\n")
                for t in tokens:
                    f.write(f"{t['name']},{t['token']}\n")
            else:
                # Simple text format - one token per line with name
                for t in tokens:
                    f.write(f"{t['name']}: {t['token']}\n")

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
                "issued_at": datetime.fromtimestamp(payload.get("iat", 0), tz=timezone.utc).isoformat(),
                "expires_at": datetime.fromtimestamp(payload.get("exp", 0), tz=timezone.utc).isoformat(),
                "max_uses": payload.get("max_uses", 1),
                "roles": payload.get("roles"),
                "policy_project": payload.get("policy", {}).get("metadata", {}).get("project"),
            }
        except Exception as e:
            raise ValueError(f"Failed to decode token: {e}")
