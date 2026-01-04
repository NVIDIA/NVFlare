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

"""FLARE Security Enrollment Package.

This package provides certificate enrollment via CSR workflow for all
participant types (clients, servers, relays, admins).

The CertRequestor communicates with a Certificate Service (which can be
hosted separately) to obtain signed certificates.

Supports enrollment types:
- client: FL client nodes (hospital-1, site-1, etc.)
- admin: Admin/researcher users with roles
- relay: Relay nodes for network topology
- server: FL server nodes

Example:
    from nvflare.security.enrollment import (
        CertRequestor,
        EnrollmentIdentity,
        EnrollmentOptions,
    )

    # Client (site) enrollment
    identity = EnrollmentIdentity.for_client("hospital-1", org="Hospital A")

    # Admin (user) enrollment
    # identity = EnrollmentIdentity.for_admin("admin@example.com", role="org_admin")

    # Relay enrollment
    # identity = EnrollmentIdentity.for_relay("relay-1")

    requestor = CertRequestor(
        cert_service_url="https://cert-service.example.com",
        enrollment_token="eyJ...",
        identity=identity,
    )

    result = requestor.request_certificate()
    print(f"Certificate: {result.cert_path}")
    print(f"Root CA: {result.ca_path}")
"""

from typing import Optional

# Re-export constants from lighter for convenience
from nvflare.lighter.constants import DEFINED_PARTICIPANT_TYPES, DEFINED_ROLES, AdminRole, ParticipantType
from nvflare.security.enrollment.cert_requestor import (
    CertRequestor,
    EnrollmentIdentity,
    EnrollmentOptions,
    EnrollmentResult,
)

# Environment variable for enrollment token
ENROLLMENT_TOKEN_ENV = "NVFLARE_ENROLLMENT_TOKEN"

# Environment variable for Certificate Service URL
CERT_SERVICE_URL_ENV = "NVFLARE_CERT_SERVICE_URL"


def get_enrollment_token(startup_dir: str = None) -> Optional[str]:
    """Get enrollment token from environment variable or file.

    Looks for token in order:
    1. NVFLARE_ENROLLMENT_TOKEN environment variable
    2. enrollment_token file in startup_dir

    Args:
        startup_dir: Directory to look for token file (optional)

    Returns:
        Token string if found, None otherwise
    """
    import os

    token = os.environ.get(ENROLLMENT_TOKEN_ENV)
    if token:
        return token.strip()

    if startup_dir:
        token_file = os.path.join(startup_dir, "enrollment_token")
        if os.path.exists(token_file):
            with open(token_file, "r") as f:
                return f.read().strip()

    return None


def get_cert_service_url(startup_dir: str = None) -> Optional[str]:
    """Get Certificate Service URL from environment variable or config file.

    Looks for URL in order:
    1. NVFLARE_CERT_SERVICE_URL environment variable
    2. enrollment.json file in startup_dir

    Args:
        startup_dir: Directory to look for config file (optional)

    Returns:
        URL string if found, None otherwise
    """
    import json
    import os

    url = os.environ.get(CERT_SERVICE_URL_ENV)
    if url:
        return url.strip()

    if startup_dir:
        config_file = os.path.join(startup_dir, "enrollment.json")
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)
                return config.get("cert_service_url")

    return None


def enroll(
    cert_service_url: str,
    token: str,
    identity: "EnrollmentIdentity",
    output_dir: str = ".",
) -> "EnrollmentResult":
    """Perform certificate enrollment via Certificate Service.

    Convenience function that creates CertRequestor and performs enrollment.

    Args:
        cert_service_url: URL of the Certificate Service
        token: JWT enrollment token
        identity: EnrollmentIdentity for the participant
        output_dir: Directory to save certificates

    Returns:
        EnrollmentResult with certificate paths and in-memory data

    Raises:
        RuntimeError: If enrollment fails
    """
    options = EnrollmentOptions(output_dir=output_dir)
    requestor = CertRequestor(
        cert_service_url=cert_service_url,
        enrollment_token=token,
        identity=identity,
        options=options,
    )
    return requestor.request_certificate()


__all__ = [
    # Main classes
    "CertRequestor",
    "EnrollmentIdentity",
    "EnrollmentOptions",
    "EnrollmentResult",
    # Helper functions
    "get_enrollment_token",
    "get_cert_service_url",
    "enroll",
    # Environment variables
    "ENROLLMENT_TOKEN_ENV",
    "CERT_SERVICE_URL_ENV",
    # Constants (from nvflare.lighter.constants)
    "ParticipantType",
    "AdminRole",
    "DEFINED_PARTICIPANT_TYPES",
    "DEFINED_ROLES",
]
