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

"""FLARE Certificate Service package.

This package provides a standalone Certificate Service for the Auto-Scale
enrollment workflow. It handles:

- JWT enrollment token generation
- JWT enrollment token validation
- Policy-based approval evaluation
- CSR signing and certificate issuance
- Pending request management
- Enrollment tracking

The service is deployed separately from the FL Server and holds the
root CA private key for signing certificates.

Components:
- CertService: Core logic for token validation, policy, and signing
- CertServiceApp: HTTP wrapper (Flask) for REST API
- EnrollmentStore: Storage for enrolled entities and pending requests

REST API Endpoints:
- POST /api/v1/enroll - Enroll with token and CSR
- POST /api/v1/token - Generate enrollment tokens (API key required)
- GET  /api/v1/pending - List pending requests (API key required)
- POST /api/v1/pending/<name>/approve - Approve request (API key required)
- POST /api/v1/pending/<name>/reject - Reject request (API key required)
- GET  /api/v1/enrolled - List enrolled entities (API key required)

Usage:
    from nvflare.cert_service import CertService, CertServiceApp
"""

from nvflare.cert_service.app import CertServiceApp
from nvflare.cert_service.cert_service import (
    ApprovalAction,
    ApprovalResult,
    CertService,
    EnrollmentContext,
    TokenPayload,
)
from nvflare.cert_service.store import (
    EnrolledEntity,
    EnrollmentStore,
    PendingRequest,
    SQLiteEnrollmentStore,
    create_enrollment_store,
)

__all__ = [
    "CertService",
    "CertServiceApp",
    "TokenPayload",
    "ApprovalResult",
    "ApprovalAction",
    "EnrollmentContext",
    "EnrollmentStore",
    "SQLiteEnrollmentStore",
    "EnrolledEntity",
    "PendingRequest",
    "create_enrollment_store",
]
