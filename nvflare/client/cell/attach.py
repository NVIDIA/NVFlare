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

import ipaddress
import re
from urllib.parse import urlsplit

from nvflare.apis.fl_constant import ConnectionSecurity
from nvflare.fuel.f3.cellnet.fqcn import FQCN

ATTACH_LEAF_PREFIX = "client_api_"
ATTACH_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
LOCAL_FILE_SCHEMES = {"file", "shared-file"}
SECURE_NETWORK_SCHEMES = {"https", "grpcs", "agrpcs", "ngrpcs", "stcp", "satcp"}


def validate_attach_id(attach_id: str) -> str:
    """Return a canonical rendezvous ID or raise ValueError."""
    if not isinstance(attach_id, str) or not ATTACH_ID_PATTERN.fullmatch(attach_id):
        raise ValueError(f"attach_id must contain 1-64 ASCII letters, digits, '_' or '-', but got {attach_id!r}")
    return attach_id


def make_attach_trainer_fqcn(site_name: str, attach_id: str) -> str:
    """Derive the site-level ad-hoc trainer identity shared by both endpoints."""
    if not isinstance(site_name, str) or not site_name:
        raise ValueError(f"site_name must be a non-empty string, but got {site_name!r}")
    attach_id = validate_attach_id(attach_id)
    fqcn = FQCN.join([site_name, f"-{ATTACH_LEAF_PREFIX}{attach_id}"])
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
    return ConnectionSecurity.MTLS if scheme in SECURE_NETWORK_SCHEMES else ConnectionSecurity.CLEAR


def validate_attach_transport(
    connect_url: str,
    connection_security: str | None,
    allow_insecure_attach: bool,
) -> str:
    """Validate the V1 attach boundary and return normalized connection security.

    File transports are protected by their filesystem policy. Clear network routes
    are local-only unless the executor explicitly opts into an insecure deployment.
    V1 network attach supports mTLS only: one-way TLS needs a separate, explicitly
    gated listener policy and is intentionally deferred.
    """
    if not isinstance(connect_url, str) or not connect_url.strip():
        raise ValueError("connect_url must be a non-empty string")
    parsed = urlsplit(connect_url)
    if not parsed.scheme:
        raise ValueError(f"connect_url must include a driver scheme, but got {connect_url!r}")
    if parsed.scheme.lower() in LOCAL_FILE_SCHEMES:
        return ConnectionSecurity.CLEAR

    security = effective_connection_security(connect_url, connection_security)
    if security == ConnectionSecurity.TLS:
        raise ValueError("bare-CA TLS attach is not supported; use mTLS or a non-network driver")
    if security == ConnectionSecurity.MTLS:
        return security

    try:
        is_loopback = ipaddress.ip_address(parsed.hostname or "").is_loopback
    except ValueError:
        is_loopback = False
    if not is_loopback and not allow_insecure_attach:
        raise ValueError(
            f"cleartext non-loopback attach route {connect_url!r} is rejected; "
            "set allow_insecure_attach=True only for a trusted development network"
        )
    return security
