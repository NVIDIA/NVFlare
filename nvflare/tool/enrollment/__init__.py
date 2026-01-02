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

"""FLARE Enrollment Token Tools.

This package provides tools for generating and managing enrollment tokens.
The TokenService requires PyJWT as an optional dependency.

Usage:
    from nvflare.tool.enrollment import TokenService
    
    service = TokenService("/path/to/ca")
    token = service.generate_token_from_file("policy.yaml", "site-1")

CLI Usage:
    nvflare enrollment token generate -s site-1 -c /path/to/ca
    nvflare enrollment token batch -n 10 --prefix hospital -o tokens.csv
    nvflare enrollment token info -t <jwt_token>
"""

from typing import TYPE_CHECKING

# Static imports for IDE type checking (not executed at runtime)
if TYPE_CHECKING:
    from nvflare.tool.enrollment.enrollment_cli import (
        def_enrollment_parser,
        handle_enrollment_cmd,
    )
    from nvflare.tool.enrollment.token_service import TokenService

__all__ = [
    "TokenService",
    "def_enrollment_parser",
    "handle_enrollment_cmd",
]


def __getattr__(name: str):
    """Lazy import to avoid requiring jwt dependency at import time."""
    if name == "TokenService":
        from nvflare.tool.enrollment.token_service import TokenService
        return TokenService
    if name == "def_enrollment_parser":
        from nvflare.tool.enrollment.enrollment_cli import def_enrollment_parser
        return def_enrollment_parser
    if name == "handle_enrollment_cmd":
        from nvflare.tool.enrollment.enrollment_cli import handle_enrollment_cmd
        return handle_enrollment_cmd
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
