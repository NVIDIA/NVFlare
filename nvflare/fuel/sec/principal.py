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

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple

AUTH_METHOD_CERT = "cert"
AUTH_METHOD_OIDC = "oidc"
AUTH_METHOD_UNKNOWN = "unknown"


def to_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def to_tuple(values) -> Tuple[str, ...]:
    if not values:
        return ()
    if isinstance(values, str):
        return (values,)
    if isinstance(values, Sequence):
        return tuple(to_str(v) for v in values if to_str(v))
    return (to_str(values),)


def to_optional_float(value) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


@dataclass(frozen=True)
class Principal:
    """Normalized authenticated admin identity.

    Principal is the boundary object between authentication providers and the
    rest of FLARE. Current authorization policy still consumes one user/org/role
    tuple; raw roles and groups are retained as source identity data for audit
    and future policy work.
    """

    subject: str
    username: str = ""
    email: str = ""
    org: str = ""
    raw_roles: Tuple[str, ...] = field(default_factory=tuple)
    groups: Tuple[str, ...] = field(default_factory=tuple)
    issuer: str = ""
    token_id: str = ""
    auth_time: Optional[float] = None
    token_exp: Optional[float] = None
    effective_role: str = ""
    auth_method: str = AUTH_METHOD_UNKNOWN

    def __post_init__(self):
        object.__setattr__(self, "subject", to_str(self.subject))
        object.__setattr__(self, "username", to_str(self.username))
        object.__setattr__(self, "email", to_str(self.email))
        object.__setattr__(self, "org", to_str(self.org))
        object.__setattr__(self, "raw_roles", to_tuple(self.raw_roles))
        object.__setattr__(self, "groups", to_tuple(self.groups))
        object.__setattr__(self, "issuer", to_str(self.issuer))
        object.__setattr__(self, "token_id", to_str(self.token_id))
        object.__setattr__(self, "auth_time", to_optional_float(self.auth_time))
        object.__setattr__(self, "token_exp", to_optional_float(self.token_exp))
        object.__setattr__(self, "effective_role", to_str(self.effective_role))
        object.__setattr__(self, "auth_method", to_str(self.auth_method) or AUTH_METHOD_UNKNOWN)

    @property
    def display_name(self) -> str:
        return self.username or self.subject

    @classmethod
    def from_legacy_admin(
        cls,
        username: str,
        org: str,
        role: str,
        subject: str = "",
        issuer: str = "",
        auth_method: str = AUTH_METHOD_CERT,
    ):
        role = to_str(role)
        return cls(
            subject=subject or username,
            username=username,
            org=org,
            raw_roles=(role,) if role else (),
            issuer=issuer,
            effective_role=role,
            auth_method=auth_method,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        if not isinstance(data, Mapping):
            raise TypeError(f"principal data must be a mapping but got {type(data)}")

        return cls(
            subject=data.get("subject", ""),
            username=data.get("username", ""),
            email=data.get("email", ""),
            org=data.get("org", ""),
            raw_roles=data.get("raw_roles", ()),
            groups=data.get("groups", ()),
            issuer=data.get("issuer", ""),
            token_id=data.get("token_id", ""),
            auth_time=data.get("auth_time"),
            token_exp=data.get("token_exp"),
            effective_role=data.get("effective_role", ""),
            auth_method=data.get("auth_method", AUTH_METHOD_UNKNOWN),
        )

    def to_dict(self) -> dict:
        data = {
            "subject": self.subject,
            "username": self.username,
            "email": self.email,
            "org": self.org,
            "raw_roles": list(self.raw_roles),
            "groups": list(self.groups),
            "issuer": self.issuer,
            "token_id": self.token_id,
            "auth_time": self.auth_time,
            "token_exp": self.token_exp,
            "effective_role": self.effective_role,
            "auth_method": self.auth_method,
        }
        return {k: v for k, v in data.items() if v not in ("", None, [], ())}

    def to_submitter_dict(self) -> dict:
        data = self.to_dict()
        submitter_keys = ("subject", "username", "email", "org", "effective_role", "auth_method", "issuer")
        return {k: data[k] for k in submitter_keys if k in data}

    def policy_name(self) -> str:
        return self.display_name

    def policy_org(self) -> str:
        return self.org

    def policy_role(self) -> str:
        return self.effective_role
