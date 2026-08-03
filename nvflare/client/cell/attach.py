# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Shared validation and naming helpers for Client API attach mode."""

import re
from urllib.parse import urlsplit

from nvflare.apis.fl_constant import ConnectionSecurity
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers.file_driver import SCHEME as SHARED_FILE_SCHEME
from nvflare.fuel.f3.drivers.file_driver import parse_file_url

ATTACH_LEAF_PREFIX = "client_api_"
ATTACH_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
SECURE_NETWORK_SCHEMES = {"https", "grpcs", "agrpcs", "ngrpcs", "stcp", "satcp"}


def validate_attach_id(attach_id: str) -> str:
    """Return a canonical rendezvous ID or raise ValueError."""
    if not isinstance(attach_id, str) or not ATTACH_ID_PATTERN.fullmatch(attach_id):
        raise ValueError(f"attach_id must contain 1-64 ASCII letters, digits, '_' or '-', but got {attach_id!r}")
    return attach_id


def make_attach_trainer_fqcn(cj_fqcn: str, attach_id: str) -> str:
    """Derive the trainer identity below the CJ that owns its Attach listener."""
    if not isinstance(cj_fqcn, str) or not cj_fqcn:
        raise ValueError(f"cj_fqcn must be a non-empty string, but got {cj_fqcn!r}")
    error = FQCN.validate(cj_fqcn)
    if error:
        raise ValueError(f"invalid cj_fqcn {cj_fqcn!r}: {error}")
    attach_id = validate_attach_id(attach_id)
    fqcn = FQCN.join([cj_fqcn, f"-{ATTACH_LEAF_PREFIX}{attach_id}"])
    error = FQCN.validate(fqcn)
    if error:
        raise ValueError(f"invalid attach trainer FQCN {fqcn!r}: {error}")
    return fqcn


def effective_connection_security(connect_url: str, connection_security: str | None) -> str:
    """Normalize the profile's configured transport security."""
    if connection_security is not None:
        if connection_security not in (
            ConnectionSecurity.CLEAR,
            ConnectionSecurity.TLS,
            ConnectionSecurity.MTLS,
        ):
            raise ValueError(
                f"invalid connection_security {connection_security!r}: expected "
                f"{ConnectionSecurity.CLEAR!r}, {ConnectionSecurity.TLS!r}, or {ConnectionSecurity.MTLS!r}"
            )
        return connection_security
    scheme = urlsplit(connect_url).scheme.lower()
    if scheme in SECURE_NETWORK_SCHEMES:
        return ConnectionSecurity.MTLS
    if scheme == SHARED_FILE_SCHEME:
        return ConnectionSecurity.CLEAR
    raise ValueError(
        "network attach without TLS must explicitly set connection_security='clear'; "
        "use 'clear' only on a trusted, isolated network; otherwise use 'mtls'"
    )


def validate_attach_profile(connect_url: str, connection_security: str | None) -> str:
    """Validate trainer-local transport profile syntax and return its security mode.

    This peer-local check is not an authorization decision. The CJ enforces whether
    a non-secure site route is permitted from trusted local runtime state before
    sending SESSION_OPEN.
    """
    if not isinstance(connect_url, str) or not connect_url.strip():
        raise ValueError("connect_url must be a non-empty string")
    parsed = urlsplit(connect_url)
    if not parsed.scheme:
        raise ValueError(f"connect_url must include a driver scheme, but got {connect_url!r}")
    scheme = parsed.scheme.lower()
    if scheme == "file":
        raise ValueError("file:// is not an F3 transport; use shared-file://0/absolute/path")
    if scheme == SHARED_FILE_SCHEME:
        if connection_security not in (None, ConnectionSecurity.CLEAR):
            raise ValueError("shared-file attach supports only connection_security='clear'")
        try:
            parse_file_url(connect_url)
        except CommError as e:
            raise ValueError(str(e)) from None
        return ConnectionSecurity.CLEAR

    security = effective_connection_security(connect_url, connection_security)
    if security == ConnectionSecurity.TLS:
        raise ValueError("bare-CA TLS attach is not supported; use mTLS or a non-network driver")
    return security
