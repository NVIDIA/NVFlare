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

This package provides client-side certificate enrollment via CSR workflow.

Supports enrollment types (from nvflare.lighter.constants):
- client: FL client nodes (hospital-1, site-1, etc.)
- admin: Admin/researcher users with roles
- relay: Relay nodes for network topology

Uses CellNet for protocol-agnostic communication (grpc, https, tcp).

Example:
    from nvflare.fuel.f3.cellnet.cell import Cell
    from nvflare.private.fed.client.enrollment import (
        CertRequestor,
        EnrollmentIdentity,
        EnrollmentOptions,
    )

    cell = Cell(fqcn="client_1", root_url="grpc://server:8002", secure=True, credentials={})
    cell.start()

    # Client (site) enrollment
    identity = EnrollmentIdentity.for_client("hospital-1", org="Hospital A")

    # Admin (user) enrollment
    # identity = EnrollmentIdentity.for_admin("admin@example.com", role="org_admin")

    # Relay enrollment
    # identity = EnrollmentIdentity.for_relay("relay-1")

    requestor = CertRequestor(
        cell=cell,
        enrollment_token="eyJ...",
        identity=identity,
    )

    cert_path = requestor.request_certificate()
    cell.stop()
"""

# Re-export constants from lighter for convenience
from nvflare.lighter.constants import DEFINED_PARTICIPANT_TYPES, DEFINED_ROLES, AdminRole, ParticipantType
from nvflare.private.fed.client.enrollment.cert_requestor import CertRequestor, EnrollmentIdentity, EnrollmentOptions

# Environment variable for enrollment token
ENROLLMENT_TOKEN_ENV = "NVFLARE_ENROLLMENT_TOKEN"

__all__ = [
    # Main classes
    "CertRequestor",
    "EnrollmentIdentity",
    "EnrollmentOptions",
    # Constants
    "ENROLLMENT_TOKEN_ENV",
    # Constants (from nvflare.lighter.constants)
    "ParticipantType",
    "AdminRole",
    "DEFINED_PARTICIPANT_TYPES",
    "DEFINED_ROLES",
]
