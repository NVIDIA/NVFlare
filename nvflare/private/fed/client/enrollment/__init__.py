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

"""FLARE Client Enrollment Package.

This package provides client-side enrollment functionality for FLARE federation.

Uses CellNet for communication (supports grpc, https, tcp protocols).

Supports:
- Site enrollment: For FL clients (hospital-1, site-1, etc.)
- User enrollment: For admin/researcher users with roles
- Relay enrollment: For relay nodes

Example:
    from nvflare.private.fed.client.enrollment import (
        CertRequestor,
        EnrollmentIdentity,
        EnrollmentOptions,
    )
    
    # Create identity
    identity = EnrollmentIdentity.for_site("hospital-1", org_name="Hospital A")
    
    # Create requestor with Cell
    requestor = CertRequestor(
        cell=cell,  # CellNet Cell object
        enrollment_token="eyJ...",
        identity=identity,
    )
    
    # Enroll
    result = requestor.enroll()
"""

from nvflare.private.fed.client.enrollment.cert_requestor import (
    # Main class
    CertRequestor,
    # Configuration models
    EnrollmentIdentity,
    EnrollmentOptions,
    # Request/Response models
    EnrollmentRequest,
    EnrollmentResult,
    # Constants
    UserRole,
)

__all__ = [
    # Main class
    "CertRequestor",
    # Configuration models
    "EnrollmentIdentity",
    "EnrollmentOptions",
    # Request/Response models
    "EnrollmentRequest",
    "EnrollmentResult",
    # Constants
    "UserRole",
]
