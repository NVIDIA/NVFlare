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

"""FLARE Certificate Service - HTTP Application.

This module provides the HTTP wrapper around CertService for the
Auto-Scale enrollment workflow.

REST API Endpoints:

Enrollment (requires valid enrollment token):
- POST /api/v1/enroll         - Enroll with token and CSR, receive signed cert + rootCA.pem

Token Generation (requires API key):
- POST /api/v1/token          - Generate enrollment token(s)

Pending Request Management (requires API key):
- GET  /api/v1/pending                     - List pending enrollment requests
- GET  /api/v1/pending/<name>              - Get details of a pending request
- POST /api/v1/pending/<name>/approve      - Approve a pending request
- POST /api/v1/pending/<name>/reject       - Reject a pending request
- POST /api/v1/pending/approve_batch       - Batch approve by pattern

Enrolled Entities (requires API key):
- GET  /api/v1/enrolled       - List enrolled entities

Health:
- GET  /health                - Health check

Security:
- rootCA.pem is only returned after token validation and approval
- Admin endpoints require API key authentication (--api-key or NVFLARE_API_KEY)
- Enrollment endpoints require valid JWT enrollment token

Usage:
    from nvflare.cert_service import CertServiceApp

    app = CertServiceApp("/path/to/config.yaml")
    app.run(host="0.0.0.0", port=8443, ssl_context=("cert.pem", "key.pem"))
"""

import logging
import os
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import yaml

from nvflare.cert_service.cert_service import CertService
from nvflare.cert_service.routes import RoutesMixin
from nvflare.cert_service.store import EnrollmentStore, create_enrollment_store

if TYPE_CHECKING:
    from nvflare.tool.enrollment.token_service import TokenService


class CertServiceApp(RoutesMixin):
    """HTTP wrapper around CertService.

    Exposes CertService functionality via REST API for:
    - Token generation (for nvflare token CLI)
    - Certificate enrollment (for CertRequestor)
    - Pending request management (for nvflare enrollment CLI)

    Security:
    - rootCA.pem is only returned after token validation and approval
    - Admin endpoints require API key authentication
    - Enrollment endpoints require valid JWT enrollment token

    Route handlers are defined in RoutesMixin (routes.py) for better organization.
    """

    # Type hints for instance attributes
    token_service: Optional["TokenService"]

    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """Initialize the Certificate Service HTTP application.

        On first start, if rootCA does not exist, it will be auto-generated.
        The rootCA private key ONLY exists on this service - never distributed.

        Args:
            config_path: Path to configuration YAML file
            **kwargs: Direct configuration options (override config file):
                - data_dir: Directory for CA files and database (default: /var/lib/cert_service)
                - project_name: Project/CA name for certificate CN (default: NVFlare)
                - host: Server host (default: 0.0.0.0)
                - port: Server port (default: 8443)
                - tls_cert: Path to TLS certificate for HTTPS (None = HTTP only)
                - tls_key: Path to TLS private key for HTTPS (None = HTTP only)
                - api_key: API key for admin authentication. Generate with: nvflare cert api-key
                - policy_file: Path to default approval policy
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        # Load configuration
        self.config = self._load_config(config_path, kwargs)

        # Auto-initialize root CA if it doesn't exist (first-start)
        # Returns already-loaded cert and key objects (avoids loading twice)
        root_cert, root_key = self._ensure_root_ca_exists()

        # Initialize the core CertService with pre-loaded objects
        self.cert_service = CertService(
            root_ca_cert=root_cert,
            root_ca_key=root_key,
        )

        # Initialize TokenService for token generation with pre-loaded objects
        self._init_token_service(root_cert, root_key)

        # Initialize enrollment store
        storage_config = self.config.get("storage", {"type": "sqlite"})
        self.enrollment_store: EnrollmentStore = create_enrollment_store(storage_config)

        # API key for admin authentication
        # Must be configured via config file or NVFLARE_API_KEY env var
        self.api_key = self.config.get("api_key") or os.environ.get("NVFLARE_API_KEY")
        if not self.api_key:
            self.logger.warning("No API key configured - admin endpoints will be disabled")
            self.logger.warning("Generate one with: nvflare cert api-key")
            self.logger.warning("Then set NVFLARE_API_KEY or add to config file")

        # Pending request timeout (default: 7 days)
        self.pending_timeout_seconds = self.config.get("pending", {}).get("timeout", 604800)

        # Create Flask app
        try:
            from flask import Flask
        except ImportError:
            raise ImportError("Flask is required for CertServiceApp. Install with: pip install flask")

        self.flask_app = Flask(__name__)
        self._register_routes()  # Inherited from RoutesMixin

    def _init_token_service(self, root_cert, root_key):
        """Initialize TokenService for token generation with pre-loaded objects."""
        try:
            from nvflare.tool.enrollment.token_service import TokenService

            # Pass pre-loaded objects to avoid loading from disk again
            self.token_service = TokenService(
                root_ca_cert=root_cert,
                signing_key=root_key,
            )
        except ImportError:
            self.logger.warning("TokenService not available - token generation disabled")
            self.token_service = None

    def _load_config(self, config_path: Optional[str], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration from file and apply overrides."""
        config: Dict[str, Any] = {
            # Data directory - root CA and database stored here
            # All paths below are derived from this if not explicitly set
            "data_dir": "/var/lib/cert_service",
            # Project name - used as root CA Common Name (CN)
            "project_name": "NVFlare",
            # Server configuration
            "server": {
                "host": "0.0.0.0",
                "port": 8443,
            },
            # Root CA paths - derived from data_dir if not specified:
            #   ca.cert = {data_dir}/rootCA.pem
            #   ca.key  = {data_dir}/rootCA.key
            "ca": {},
            # Approval policy
            "policy": {},
            # Storage configuration
            "storage": {
                "type": "sqlite",
            },
            # Pending request settings
            "pending": {
                "timeout": 604800,  # 7 days in seconds
                "cleanup_interval": 3600,  # Clean up expired requests every hour
            },
            # Audit logging
            "audit": {
                "enabled": True,
            },
        }

        # Load from file if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    self._merge_config(config, file_config)

        # Apply overrides from kwargs
        if "data_dir" in overrides:
            config["data_dir"] = overrides["data_dir"]
        if "project_name" in overrides:
            config["project_name"] = overrides["project_name"]
        if "root_ca_cert_path" in overrides:
            config["ca"]["cert"] = overrides["root_ca_cert_path"]
        if "root_ca_key_path" in overrides:
            config["ca"]["key"] = overrides["root_ca_key_path"]
        if "host" in overrides:
            config["server"]["host"] = overrides["host"]
        if "port" in overrides:
            config["server"]["port"] = overrides["port"]
        if "tls_cert" in overrides:
            if "tls" not in config["server"]:
                config["server"]["tls"] = {}
            config["server"]["tls"]["cert"] = overrides["tls_cert"]
        if "tls_key" in overrides:
            if "tls" not in config["server"]:
                config["server"]["tls"] = {}
            config["server"]["tls"]["key"] = overrides["tls_key"]
        if "api_key" in overrides:
            config["api_key"] = overrides["api_key"]
        if "policy_file" in overrides:
            if "policy" not in config:
                config["policy"] = {}
            config["policy"]["file"] = overrides["policy_file"]
        if "db_path" in overrides:
            config["storage"]["path"] = overrides["db_path"]

        # Derive paths from data_dir if not explicitly set
        data_dir = config["data_dir"]

        # CA paths
        if not config.get("ca", {}).get("cert"):
            config.setdefault("ca", {})["cert"] = os.path.join(data_dir, "rootCA.pem")
        if not config.get("ca", {}).get("key"):
            config.setdefault("ca", {})["key"] = os.path.join(data_dir, "rootCA.key")

        # Storage path
        if not config.get("storage", {}).get("path"):
            config["storage"]["path"] = os.path.join(data_dir, "enrollment.db")

        # Audit log path
        if config.get("audit", {}).get("enabled") and not config.get("audit", {}).get("log_file"):
            config["audit"]["log_file"] = os.path.join(data_dir, "audit.log")

        return config

    def _merge_config(self, base: dict, override: dict):
        """Recursively merge override into base config."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def _ensure_root_ca_exists(self):
        """Ensure root CA exists and is valid, auto-generate on first start.

        The root CA private key is generated HERE and ONLY exists on this service.
        This is the security foundation - the key never leaves this service.

        Returns:
            Tuple of (root_cert, root_key) - loaded certificate and key objects
        """
        import json

        from nvflare.lighter.utils import Identity, generate_cert, generate_keys, serialize_cert, serialize_pri_key

        cert_path = self.config["ca"]["cert"]
        key_path = self.config["ca"]["key"]
        data_dir = self.config["data_dir"]

        # Check if CA already exists and is valid
        if os.path.exists(cert_path) and os.path.exists(key_path):
            result = self._validate_root_ca(cert_path, key_path)
            if result is not None:
                root_cert, root_key = result
                self.logger.info(f"Root CA validated: {cert_path}")
                return root_cert, root_key
            else:
                # Validation failed - files exist but are invalid
                raise ValueError(
                    f"Root CA files exist but are invalid. "
                    f"Check {cert_path} and {key_path} are valid PEM files and match."
                )

        # First start - generate new root CA
        self.logger.info("First start detected - generating root CA...")

        # Create data directory
        os.makedirs(data_dir, exist_ok=True)

        # Get project name for CA CN
        project_name = self.config.get("project_name", "NVFlare")
        valid_days = self.config.get("ca", {}).get("valid_days", 3650)  # 10 years

        # Generate root CA
        pri_key, pub_key = generate_keys()
        root_cert = generate_cert(
            subject=Identity(project_name),
            issuer=Identity(project_name),
            signing_pri_key=pri_key,
            subject_pub_key=pub_key,
            valid_days=valid_days,
            ca=True,
        )

        # Write rootCA.pem (public - can be distributed)
        with open(cert_path, "wb") as f:
            f.write(serialize_cert(root_cert))
        self.logger.info(f"Root CA certificate created: {cert_path}")

        # Write rootCA.key (private - NEVER leaves this service!)
        with open(key_path, "wb") as f:
            f.write(serialize_pri_key(pri_key))
        os.chmod(key_path, 0o600)  # Restrictive permissions
        self.logger.info(f"Root CA private key created: {key_path} (KEEP SECURE!)")

        # Write state/cert.json for TokenService compatibility
        state_dir = os.path.join(data_dir, "state")
        os.makedirs(state_dir, exist_ok=True)
        cert_json_path = os.path.join(state_dir, "cert.json")
        cert_state = {
            "root_cert": serialize_cert(root_cert).decode("utf-8"),
            "root_pri_key": serialize_pri_key(pri_key).decode("utf-8"),
            "issuer": project_name,
        }
        with open(cert_json_path, "w") as f:
            json.dump(cert_state, f, indent=2)
        os.chmod(cert_json_path, 0o600)

        self.logger.info("=" * 60)
        self.logger.info("ROOT CA GENERATED - FIRST START COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"Project: {project_name}")
        self.logger.info(f"Validity: {valid_days} days")
        self.logger.info("")
        self.logger.info("SECURITY NOTE:")
        self.logger.info("  The root CA private key exists ONLY on this service.")
        self.logger.info("  It is NEVER distributed to any other location.")
        self.logger.info("  Protect this server and its data directory!")

        # Return the generated cert and key (already in memory, no need to reload)
        return root_cert, pri_key

    def _validate_root_ca(self, cert_path: str, key_path: str):
        """Validate that root CA files are valid and match.

        Args:
            cert_path: Path to root CA certificate (PEM)
            key_path: Path to root CA private key (PEM)

        Returns:
            Tuple of (cert, key) if valid, None if invalid
        """
        from cryptography import x509
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        try:
            # Load and validate certificate
            with open(cert_path, "rb") as f:
                cert_pem = f.read()
            cert = x509.load_pem_x509_certificate(cert_pem)

            # Check if it's a CA certificate
            try:
                basic_constraints = cert.extensions.get_extension_for_class(x509.BasicConstraints)
                if not basic_constraints.value.ca:
                    self.logger.error(f"Certificate at {cert_path} is not a CA certificate")
                    return None
            except x509.ExtensionNotFound:
                self.logger.warning(f"Certificate at {cert_path} missing BasicConstraints extension")
                # Continue - some older certs may not have this

            # Load and validate private key
            with open(key_path, "rb") as f:
                key_pem = f.read()
            private_key = serialization.load_pem_private_key(key_pem, password=None)

            # Check key type
            if not isinstance(private_key, rsa.RSAPrivateKey):
                self.logger.error(f"Private key at {key_path} is not an RSA key")
                return None

            # Verify certificate and key match by comparing public keys
            cert_public_key = cert.public_key()
            key_public_key = private_key.public_key()

            cert_pub_bytes = cert_public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            key_pub_bytes = key_public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )

            if cert_pub_bytes != key_pub_bytes:
                self.logger.error("Certificate and private key do not match!")
                return None

            self.logger.debug(f"Root CA validation passed: {cert.subject}")
            return cert, private_key

        except Exception as e:
            self.logger.error(f"Failed to validate root CA: {e}")
            return None

    def _require_api_key(self, f: Callable) -> Callable:
        """Decorator to require API key authentication."""
        from flask import jsonify, request

        @wraps(f)
        def decorated(*args, **kwargs):
            if not self.api_key:
                return jsonify({"error": "API key not configured on server"}), 500

            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return jsonify({"error": "Missing or invalid Authorization header"}), 401

            provided_key = auth_header[7:]  # Remove "Bearer " prefix
            if provided_key != self.api_key:
                return jsonify({"error": "Invalid API key"}), 401

            return f(*args, **kwargs)

        return decorated

    # Route registration methods are inherited from RoutesMixin (routes.py)
    # See routes.py for: _register_routes, _register_enrollment_routes,
    # _register_token_routes, _register_pending_routes, _register_enrolled_routes,
    # _register_health_routes, _register_ca_routes

    def run(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        ssl_context: Optional[Tuple[str, str]] = None,
        debug: bool = False,
    ):
        """Run the Certificate Service HTTP server.

        Args:
            host: Server host (default from config)
            port: Server port (default from config)
            ssl_context: Tuple of (cert_path, key_path) for HTTPS
            debug: Enable Flask debug mode
        """
        host = host or self.config["server"]["host"]
        port = port or self.config["server"]["port"]

        # Get SSL context from config if not provided
        if not ssl_context:
            tls_config = self.config.get("server", {}).get("tls", {})
            if tls_config.get("cert") and tls_config.get("key"):
                ssl_context = (tls_config["cert"], tls_config["key"])

        self.logger.info(f"Starting Certificate Service on {host}:{port}")
        if ssl_context:
            self.logger.info("TLS enabled")
        else:
            self.logger.warning("TLS disabled - NOT recommended for production!")

        if not self.api_key:
            self.logger.warning("No API key configured - admin endpoints will fail")

        self.flask_app.run(
            host=host,
            port=port,
            ssl_context=ssl_context,
            debug=debug,
        )

    def get_wsgi_app(self):
        """Get the WSGI application for production deployment.

        Returns:
            Flask WSGI application

        Example:
            # For gunicorn: gunicorn "nvflare.cert_service.app:create_app()"
            app = CertServiceApp("/path/to/config.yaml")
            wsgi_app = app.get_wsgi_app()
        """
        return self.flask_app


def create_app(config_path: Optional[str] = None) -> "CertServiceApp":
    """Factory function for creating CertServiceApp.

    Useful for WSGI servers like gunicorn.

    Args:
        config_path: Path to configuration YAML file.
                     Can also be set via CERT_SERVICE_CONFIG env var.

    Returns:
        CertServiceApp instance
    """
    if not config_path:
        config_path = os.environ.get("CERT_SERVICE_CONFIG")
    return CertServiceApp(config_path)
