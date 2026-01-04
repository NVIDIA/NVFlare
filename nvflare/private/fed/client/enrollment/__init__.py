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

"""FLARE Client Enrollment Package (Deprecated Location).

This package has been moved to nvflare.security.enrollment.
This module provides backward compatibility imports.

New imports should use:
    from nvflare.security.enrollment import (
        CertRequestor,
        EnrollmentIdentity,
        EnrollmentOptions,
        EnrollmentResult,
    )
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "nvflare.private.fed.client.enrollment is deprecated. " "Use nvflare.security.enrollment instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from new location for backward compatibility
from nvflare.security.enrollment import (
    CERT_SERVICE_URL_ENV,
    DEFINED_PARTICIPANT_TYPES,
    DEFINED_ROLES,
    ENROLLMENT_TOKEN_ENV,
    AdminRole,
    CertRequestor,
    EnrollmentIdentity,
    EnrollmentOptions,
    EnrollmentResult,
    ParticipantType,
)

__all__ = [
    # Main classes
    "CertRequestor",
    "EnrollmentIdentity",
    "EnrollmentOptions",
    "EnrollmentResult",
    # Constants
    "ENROLLMENT_TOKEN_ENV",
    "CERT_SERVICE_URL_ENV",
    # Constants (from nvflare.lighter.constants)
    "ParticipantType",
    "AdminRole",
    "DEFINED_PARTICIPANT_TYPES",
    "DEFINED_ROLES",
]
