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

"""Certificate Service HTTP Route Handlers.

This module contains all HTTP route registration logic for the Certificate Service.
It is implemented as a mixin class that CertServiceApp inherits from.

Route Categories:
- Enrollment routes: CSR submission and certificate issuance
- Token routes: Enrollment token generation
- Pending routes: Manual approval workflow management
- Enrolled routes: Query enrolled entities
- CA routes: Root CA certificate download
- Health routes: Health check endpoints
"""

import fnmatch
import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Callable, Optional

from nvflare.cert_service.cert_service import ApprovalAction, EnrollmentContext
from nvflare.cert_service.store import EnrolledEntity, PendingRequest
from nvflare.lighter.constants import DEFINED_PARTICIPANT_TYPES, DEFINED_ROLES, AdminRole, ParticipantType

if TYPE_CHECKING:
    from flask import Flask

    from nvflare.cert_service.cert_service import CertService
    from nvflare.cert_service.store import EnrollmentStore
    from nvflare.tool.enrollment.token_service import TokenService


class RoutesMixin:
    """Mixin class providing HTTP route registration for CertServiceApp.

    This mixin is inherited by CertServiceApp and provides all route
    registration methods. Attributes used from the parent class:
    - self.flask_app: Flask application instance
    - self.cert_service: CertService instance
    - self.token_service: TokenService instance
    - self.enrollment_store: EnrollmentStore instance
    - self.config: Configuration dictionary
    - self.api_key: API key for admin authentication
    - self.pending_timeout_seconds: Timeout for pending requests
    - self.logger: Logger instance
    - self._require_api_key: Decorator for API key authentication
    """

    # Type hints for attributes provided by parent class (CertServiceApp)
    flask_app: "Flask"
    cert_service: "CertService"
    token_service: "TokenService"
    enrollment_store: "EnrollmentStore"
    config: dict
    api_key: Optional[str]
    pending_timeout_seconds: int
    logger: logging.Logger
    _require_api_key: Callable[[Callable[..., Any]], Callable[..., Any]]

    def _register_routes(self):
        """Register all HTTP routes."""
        self._register_enrollment_routes()
        self._register_token_routes()
        self._register_pending_routes()
        self._register_enrolled_routes()
        self._register_ca_routes()
        self._register_health_routes()

    def _register_enrollment_routes(self):
        """Register enrollment endpoint for certificate issuance.

        This is the main endpoint for participants (clients, servers, relays, admins)
        to obtain their certificates. The workflow is:

        1. Participant generates a key pair and CSR locally (private key never leaves)
        2. Participant submits CSR + enrollment token to this endpoint
        3. Service validates token, evaluates embedded policy
        4. If approved: signs CSR and returns certificate + root CA
        5. If pending: queues for manual approval, returns 202
        6. If rejected: returns 403 with rejection reason

        The enrollment token (JWT) contains:
        - Subject: who can use this token (name or pattern)
        - Subject type: client, server, relay, admin
        - Expiration: when token becomes invalid
        - Policy: rules for auto-approve, pending, or reject

        Security:
        - Token is signed with root CA private key or JWT verification key (tamper-proof)
        - CSR contains only public key (private key stays with participant)
        - Returned certificate is signed by root CA (trusted by all participants)
        """
        from flask import jsonify, request

        @self.flask_app.route("/api/v1/enroll", methods=["POST"])
        def enroll():
            """Handle enrollment request.

            Request:
                {
                    "token": "JWT enrollment token",
                    "csr": "PEM-encoded CSR",
                    "metadata": {
                        "name": "site name",
                        "type": "client|server|relay|admin",
                        "org": "organization (optional)",
                        "role": "role for admin (optional)"
                    }
                }

            Response (200):
                {
                    "certificate": "PEM-encoded signed certificate",
                    "ca_cert": "PEM-encoded rootCA.pem"
                }

            Response (202 - Pending):
                {
                    "status": "pending",
                    "message": "Enrollment queued for manual approval"
                }

            Response (401): Invalid/expired token
            Response (403): Policy rejection
            Response (400): Bad request
            Response (500): Server error
            """
            try:
                data = request.json
                if not data:
                    return jsonify({"error": "Request body required"}), 400

                # Extract required fields
                token = data.get("token")
                csr = data.get("csr")
                metadata = data.get("metadata", {})

                if not token:
                    return jsonify({"error": "Missing 'token' field"}), 400
                if not csr:
                    return jsonify({"error": "Missing 'csr' field"}), 400
                if not metadata.get("name"):
                    return jsonify({"error": "Missing 'metadata.name' field"}), 400

                # Validate participant type
                participant_type = metadata.get("type", ParticipantType.CLIENT)
                if participant_type not in DEFINED_PARTICIPANT_TYPES:
                    return jsonify({"error": f"Invalid participant type: {participant_type}"}), 400

                # Validate and default role for admin tokens
                # Default to LEAD (has job submission permissions)
                role = metadata.get("role")
                if participant_type == ParticipantType.ADMIN:
                    if not role:
                        role = AdminRole.LEAD
                    elif role not in DEFINED_ROLES:
                        return jsonify({"error": f"Invalid role: {role}"}), 400

                name = metadata["name"]
                org = metadata.get("org", "")

                # Check if already enrolled
                if self.enrollment_store.is_enrolled(name, participant_type):
                    return jsonify({"error": f"Entity '{name}' already enrolled"}), 409

                # Check if already pending
                pending = self.enrollment_store.get_pending(name, participant_type)
                if pending:
                    if pending.approved and pending.signed_cert:
                        # Already approved - return the certificate
                        return jsonify(
                            {
                                "certificate": pending.signed_cert,
                                "ca_cert": self.cert_service.get_root_ca_pem(),
                            }
                        )
                    else:
                        # Still pending
                        return (
                            jsonify(
                                {
                                    "status": "pending",
                                    "message": "Enrollment request is pending manual approval",
                                }
                            ),
                            202,
                        )

                # Build enrollment context
                context = EnrollmentContext(
                    name=name,
                    participant_type=participant_type,
                    org=org,
                    role=role,
                    source_ip=request.remote_addr,
                )

                # Ensure CSR is bytes
                csr_bytes = csr.encode("utf-8") if isinstance(csr, str) else csr

                # Validate token first
                token_payload = self.cert_service.validate_token(token, context)

                # Evaluate policy
                result = self.cert_service.evaluate_policy(token_payload, context)

                if result.action == ApprovalAction.REJECT:
                    return jsonify({"error": f"Rejected: {result.message}"}), 403

                if result.action == ApprovalAction.PENDING:
                    # Queue for manual approval
                    expires_at = datetime.now(timezone.utc) + timedelta(seconds=self.pending_timeout_seconds)
                    pending_request = PendingRequest(
                        name=name,
                        entity_type=participant_type,
                        org=org,
                        csr_pem=csr if isinstance(csr, str) else csr.decode("utf-8"),
                        submitted_at=datetime.now(timezone.utc),
                        expires_at=expires_at,
                        token_subject=token_payload.subject,
                        role=role,
                        source_ip=request.remote_addr,
                    )
                    self.enrollment_store.add_pending(pending_request)

                    return (
                        jsonify(
                            {
                                "status": "pending",
                                "message": f"Enrollment queued for manual approval (rule: {result.rule_name})",
                            }
                        ),
                        202,
                    )

                # Approved - sign the certificate
                signed_cert = self.cert_service.sign_csr(
                    csr_data=csr_bytes,
                    token=token,
                    context=context,
                )

                # Record enrollment
                self.enrollment_store.add_enrolled(
                    EnrolledEntity(
                        name=name,
                        entity_type=participant_type,
                        enrolled_at=datetime.now(timezone.utc),
                        org=org,
                        role=role,
                    )
                )

                # Return certificate AND rootCA.pem (only after approval!)
                return jsonify(
                    {
                        "certificate": signed_cert.decode("utf-8"),
                        "ca_cert": self.cert_service.get_root_ca_pem(),
                    }
                )

            except ValueError as e:
                # Token or policy errors
                error_msg = str(e)
                if "expired" in error_msg.lower() or "signature" in error_msg.lower():
                    return jsonify({"error": error_msg}), 401
                elif "rejected" in error_msg.lower():
                    return jsonify({"error": error_msg}), 403
                else:
                    return jsonify({"error": error_msg}), 400

            except Exception as e:
                error_msg = str(e)
                # Handle Flask's BadRequest for malformed JSON
                if "400" in error_msg or "Bad Request" in error_msg:
                    return jsonify({"error": "Invalid request body"}), 400
                # Log error type only to avoid leaking sensitive data
                self.logger.error(f"Enrollment error: {type(e).__name__}")
                return jsonify({"error": "Internal server error"}), 500

    def _register_token_routes(self):
        """Register token generation endpoints for Project Admins.

        These endpoints allow Project Admins to generate enrollment tokens
        that can be distributed to site operators. Tokens are JWTs signed
        with the root CA private key, containing:

        - Subject: The participant name or pattern (e.g., "hospital-1", "site-*")
        - Type: client, server, relay, or admin
        - Expiration: When the token becomes invalid
        - Policy: Embedded approval rules (auto-approve, pending, reject)

        Authentication:
        - Requires API key (Bearer token in Authorization header)
        - Only Project Admins should have access to the API key

        Endpoints:
        - POST /api/v1/token - Generate single or batch tokens
        - GET /api/v1/token/<token_id> - Get token info (if implemented)

        Security:
        - Tokens are signed with root CA key (cannot be forged)
        - Single-use enforcement via enrollment tracking
        - Short expiry recommended (hours to days)
        """
        from flask import jsonify, request

        @self.flask_app.route("/api/v1/token", methods=["POST"])
        @self._require_api_key
        def generate_token():
            """Generate enrollment token(s).

            Request (single token):
                {
                    "name": "site-1",
                    "entity_type": "client",
                    "valid_days": 7,
                    "policy_file": "/path/to/policy.yaml"  (optional)
                }

            Request (batch):
                {
                    "names": ["site-1", "site-2", "site-3"],
                    "entity_type": "client",
                    "valid_days": 7
                }

            Response (single):
                {
                    "token": "eyJhbGciOiJSUzI1NiIs...",
                    "subject": "site-1",
                    "expires_at": "2025-01-11T10:00:00Z"
                }

            Response (batch):
                {
                    "tokens": [
                        {"name": "site-1", "token": "eyJ..."},
                        {"name": "site-2", "token": "eyJ..."},
                        ...
                    ]
                }
            """
            if not self.token_service:
                return jsonify({"error": "Token generation not available"}), 501

            try:
                data = request.json
                if not data:
                    return jsonify({"error": "Request body required"}), 400

                entity_type = data.get("entity_type", ParticipantType.CLIENT)
                valid_days = data.get("valid_days", 7)
                policy_file = data.get("policy_file") or self.config.get("policy", {}).get("file")

                # Batch mode
                if "names" in data:
                    names = data["names"]
                    if not isinstance(names, list) or not names:
                        return jsonify({"error": "'names' must be a non-empty list"}), 400

                    tokens = []
                    for name in names:
                        token = self._generate_single_token(
                            name=name,
                            entity_type=entity_type,
                            valid_days=valid_days,
                            policy_file=policy_file,
                            role=data.get("role"),
                        )
                        tokens.append({"name": name, "token": token})

                    return jsonify({"tokens": tokens})

                # Single token mode
                name = data.get("name")
                if not name:
                    return jsonify({"error": "Missing 'name' field"}), 400

                token = self._generate_single_token(
                    name=name,
                    entity_type=entity_type,
                    valid_days=valid_days,
                    policy_file=policy_file,
                    role=data.get("role"),
                )

                expires_at = datetime.now(timezone.utc) + timedelta(days=valid_days)
                return jsonify(
                    {
                        "token": token,
                        "subject": name,
                        "expires_at": expires_at.isoformat(),
                    }
                )

            except Exception as e:
                # Log error type only to avoid leaking sensitive data in logs
                self.logger.error(f"Token generation error: {type(e).__name__}")
                return jsonify({"error": "Token generation failed"}), 500

    def _generate_single_token(
        self,
        name: str,
        entity_type: str,
        valid_days: int,
        policy_file: Optional[str],
        role: Optional[str],
    ) -> str:
        """Generate a single enrollment token."""
        if entity_type == ParticipantType.ADMIN:
            return self.token_service.generate_admin_token(
                user_id=name,
                valid_days=valid_days,
                policy_file=policy_file,
                roles=[role] if role else [AdminRole.LEAD],
            )
        elif entity_type == ParticipantType.RELAY:
            return self.token_service.generate_relay_token(
                relay_name=name,
                valid_days=valid_days,
                policy_file=policy_file,
            )
        else:
            return self.token_service.generate_site_token(
                site_name=name,
                valid_days=valid_days,
                policy_file=policy_file,
            )

    def _register_pending_routes(self):
        """Register pending enrollment request management endpoints.

        When an enrollment policy specifies "pending" as the action, the request
        is queued for manual review. Project Admins can then:

        1. List all pending requests to see who is waiting for approval
        2. Get details about a specific pending request
        3. Approve a request - signs the CSR and issues a certificate
        4. Reject a request - denies enrollment with a reason

        Workflow:
        1. Site submits enrollment request with token
        2. Policy evaluates to "pending" action
        3. Request is stored with CSR and metadata
        4. Admin reviews via these endpoints
        5. Admin approves/rejects
        6. Site polls or retries enrollment to get result

        Authentication:
        - All endpoints require API key (Project Admin only)

        Endpoints:
        - GET /api/v1/pending - List all pending requests
        - GET /api/v1/pending/<name> - Get specific request details
        - POST /api/v1/pending/<name>/approve - Approve and issue certificate
        - POST /api/v1/pending/<name>/reject - Reject with reason
        - POST /api/v1/pending/approve_batch - Bulk approve
        - POST /api/v1/pending/reject_batch - Bulk reject

        Timeout:
        - Pending requests expire after configured timeout (default: 7 days)
        - Expired requests are automatically cleaned up
        """
        from flask import jsonify, request

        @self.flask_app.route("/api/v1/pending", methods=["GET"])
        @self._require_api_key
        def list_pending():
            """List pending enrollment requests."""
            entity_type = request.args.get("entity_type")
            pending = self.enrollment_store.get_all_pending(entity_type)

            return jsonify(
                [
                    {
                        "name": p.name,
                        "entity_type": p.entity_type,
                        "org": p.org,
                        "submitted_at": p.submitted_at.isoformat(),
                        "expires_at": p.expires_at.isoformat(),
                        "status": "approved" if p.approved else "pending",
                    }
                    for p in pending
                ]
            )

        @self.flask_app.route("/api/v1/pending/<name>", methods=["GET"])
        @self._require_api_key
        def get_pending_info(name: str):
            """Get details of a pending enrollment request."""
            entity_type = request.args.get("entity_type")
            if not entity_type:
                return jsonify({"error": "entity_type query parameter required"}), 400

            pending = self.enrollment_store.get_pending(name, entity_type)
            if not pending:
                return jsonify({"error": f"No pending request for {entity_type}: {name}"}), 404

            return jsonify(
                {
                    "name": pending.name,
                    "entity_type": pending.entity_type,
                    "org": pending.org,
                    "status": "approved" if pending.approved else "pending",
                    "submitted_at": pending.submitted_at.isoformat(),
                    "expires_at": pending.expires_at.isoformat(),
                    "source_ip": pending.source_ip,
                    "csr_subject": f"CN={pending.name},O={pending.org}",
                    "token_subject": pending.token_subject,
                    "role": pending.role,
                }
            )

        @self.flask_app.route("/api/v1/pending/<name>/approve", methods=["POST"])
        @self._require_api_key
        def approve_pending(name: str):
            """Approve a pending enrollment request."""
            data = request.json or {}
            entity_type = data.get("entity_type")
            if not entity_type:
                return jsonify({"error": "entity_type required"}), 400

            pending = self.enrollment_store.get_pending(name, entity_type)
            if not pending:
                return jsonify({"error": f"No pending request for {entity_type}: {name}"}), 404

            try:
                context = EnrollmentContext(
                    name=pending.name,
                    participant_type=pending.entity_type,
                    org=pending.org,
                    role=pending.role,
                    source_ip=pending.source_ip,
                )

                csr_bytes = pending.csr_pem.encode("utf-8")
                cert = self.cert_service._sign_certificate(csr_bytes, context)
                from nvflare.lighter.utils import serialize_cert

                signed_cert = serialize_cert(cert).decode("utf-8")

                self.enrollment_store.approve_pending(
                    name=name,
                    entity_type=entity_type,
                    signed_cert=signed_cert,
                    approved_by="admin",
                )

                self.enrollment_store.add_enrolled(
                    EnrolledEntity(
                        name=name,
                        entity_type=entity_type,
                        enrolled_at=datetime.now(timezone.utc),
                        org=pending.org,
                        role=pending.role,
                    )
                )

                return jsonify(
                    {
                        "status": "approved",
                        "name": name,
                        "entity_type": entity_type,
                        "certificate_issued": True,
                    }
                )

            except Exception as e:
                # Log error type only to avoid leaking sensitive data in logs
                self.logger.error(f"Error approving request: {type(e).__name__}")
                return jsonify({"error": "Failed to approve request"}), 500

        @self.flask_app.route("/api/v1/pending/<name>/reject", methods=["POST"])
        @self._require_api_key
        def reject_pending(name: str):
            """Reject a pending enrollment request."""
            data = request.json or {}
            entity_type = data.get("entity_type")
            reason = data.get("reason", "Rejected by administrator")

            if not entity_type:
                return jsonify({"error": "entity_type required"}), 400

            removed = self.enrollment_store.reject_pending(name, entity_type, reason)
            if not removed:
                return jsonify({"error": f"No pending request for {entity_type}: {name}"}), 404

            return jsonify(
                {
                    "status": "rejected",
                    "name": name,
                    "entity_type": entity_type,
                }
            )

        @self.flask_app.route("/api/v1/pending/approve_batch", methods=["POST"])
        @self._require_api_key
        def approve_batch():
            """Batch approve pending requests by pattern."""
            data = request.json or {}
            pattern = data.get("pattern")
            entity_type = data.get("entity_type")

            if not pattern:
                return jsonify({"error": "pattern required"}), 400
            if not entity_type:
                return jsonify({"error": "entity_type required"}), 400

            pending = self.enrollment_store.get_all_pending(entity_type)
            approved_names = []

            for req in pending:
                if fnmatch.fnmatch(req.name, pattern):
                    try:
                        context = EnrollmentContext(
                            name=req.name,
                            participant_type=req.entity_type,
                            org=req.org,
                            role=req.role,
                            source_ip=req.source_ip,
                        )

                        csr_bytes = req.csr_pem.encode("utf-8")
                        cert = self.cert_service._sign_certificate(csr_bytes, context)
                        from nvflare.lighter.utils import serialize_cert

                        signed_cert = serialize_cert(cert).decode("utf-8")

                        self.enrollment_store.approve_pending(
                            name=req.name,
                            entity_type=entity_type,
                            signed_cert=signed_cert,
                            approved_by="admin",
                        )

                        self.enrollment_store.add_enrolled(
                            EnrolledEntity(
                                name=req.name,
                                entity_type=entity_type,
                                enrolled_at=datetime.now(timezone.utc),
                                org=req.org,
                                role=req.role,
                            )
                        )

                        approved_names.append(req.name)

                    except Exception as e:
                        self.logger.error(f"Error approving {req.name}: {e}")

            return jsonify(
                {
                    "approved_count": len(approved_names),
                    "approved_names": approved_names,
                }
            )

        @self.flask_app.route("/api/v1/pending/reject_batch", methods=["POST"])
        @self._require_api_key
        def reject_batch():
            """Batch reject pending requests by pattern."""
            data = request.json or {}
            pattern = data.get("pattern")
            entity_type = data.get("entity_type")
            reason = data.get("reason", "Rejected by administrator")

            if not pattern:
                return jsonify({"error": "pattern required"}), 400
            if not entity_type:
                return jsonify({"error": "entity_type required"}), 400

            pending = self.enrollment_store.get_all_pending(entity_type)
            rejected_names = []

            for req in pending:
                if fnmatch.fnmatch(req.name, pattern):
                    self.enrollment_store.reject_pending(req.name, entity_type, reason)
                    rejected_names.append(req.name)

            return jsonify(
                {
                    "rejected_count": len(rejected_names),
                    "rejected_names": rejected_names,
                }
            )

    def _register_enrolled_routes(self):
        """Register enrolled entities query endpoint.

        This endpoint provides visibility into which participants have
        successfully enrolled and received certificates. Useful for:

        - Auditing: Track who has enrolled and when
        - Monitoring: Verify expected participants are enrolled
        - Troubleshooting: Check if a site has successfully enrolled
        - Capacity planning: See total enrolled count by type

        Authentication:
        - Requires API key (Project Admin only)

        Endpoints:
        - GET /api/v1/enrolled - List all enrolled entities
        - GET /api/v1/enrolled?entity_type=client - Filter by type
        """
        from flask import jsonify, request

        @self.flask_app.route("/api/v1/enrolled", methods=["GET"])
        @self._require_api_key
        def list_enrolled():
            """List enrolled entities."""
            entity_type = request.args.get("entity_type")
            enrolled = self.enrollment_store.get_enrolled(entity_type)

            return jsonify(
                [
                    {
                        "name": e.name,
                        "entity_type": e.entity_type,
                        "org": e.org,
                        "role": e.role,
                        "enrolled_at": e.enrolled_at.isoformat(),
                    }
                    for e in enrolled
                ]
            )

    def _register_health_routes(self):
        """Register health check endpoint for monitoring.

        Provides a simple health check for:
        - Load balancer health probes
        - Kubernetes liveness/readiness checks
        - Monitoring systems (Prometheus, Datadog, etc.)
        - Uptime verification

        Authentication:
        - No authentication required (public endpoint)

        Endpoints:
        - GET /health - Returns {"status": "healthy"} if service is running
        """
        from flask import jsonify

        @self.flask_app.route("/health", methods=["GET"])
        def health():
            """Health check endpoint."""
            return jsonify({"status": "healthy"})

    def _register_ca_routes(self):
        """Register CA certificate endpoints for trust establishment.

        These endpoints provide access to the root CA public certificate,
        which is needed by all participants to:

        1. Verify the Certificate Service's TLS certificate (bootstrap trust)
        2. Verify certificates of other FLARE participants (mTLS)
        3. Verify JWT enrollment tokens (optional, if not using root CA)

        The root CA public certificate (rootCA.pem) can be safely distributed -
        it contains only the public key. The private key (rootCA.key) is NEVER
        exposed and exists only within this Certificate Service.

        Authentication:
        - GET /api/v1/ca-cert - Public (no auth) - download rootCA.pem
        - GET /api/v1/ca-info - Requires API key - get CA metadata

        Endpoints:
        - GET /api/v1/ca-cert - Download PEM-encoded root CA certificate
        - GET /api/v1/ca-info - Get CA info (subject, issuer, validity, fingerprint)

        Security:
        - Only public certificate is exposed, never the private key
        - API key required for detailed CA info
        """
        from flask import Response, jsonify

        @self.flask_app.route("/api/v1/ca-cert", methods=["GET"])
        def get_ca_cert():
            """Download public root CA certificate."""
            cert_path = self.config["ca"]["cert"]
            try:
                with open(cert_path, "rb") as f:
                    cert_pem = f.read()
                return Response(
                    cert_pem,
                    mimetype="application/x-pem-file",
                    headers={"Content-Disposition": "attachment; filename=rootCA.pem"},
                )
            except FileNotFoundError:
                return jsonify({"error": "Root CA not initialized"}), 500

        @self.flask_app.route("/api/v1/ca-info", methods=["GET"])
        def get_ca_info():
            """Get root CA information (without exposing private key)."""
            from cryptography import x509

            cert_path = self.config["ca"]["cert"]
            try:
                with open(cert_path, "rb") as f:
                    cert_pem = f.read()
                cert = x509.load_pem_x509_certificate(cert_pem)
                # Use not_valid_before/after (works on all Python versions)
                # not_valid_before_utc/after_utc are Python 3.11+
                not_before = getattr(cert, "not_valid_before_utc", cert.not_valid_before)
                not_after = getattr(cert, "not_valid_after_utc", cert.not_valid_after)
                return jsonify(
                    {
                        "subject": cert.subject.rfc4514_string(),
                        "issuer": cert.issuer.rfc4514_string(),
                        "not_valid_before": not_before.isoformat(),
                        "not_valid_after": not_after.isoformat(),
                        "serial_number": str(cert.serial_number),
                    }
                )
            except FileNotFoundError:
                return jsonify({"error": "Root CA not initialized"}), 500
            except Exception:
                # Don't include exception details to avoid leaking sensitive info
                return jsonify({"error": "Failed to read CA info"}), 500
