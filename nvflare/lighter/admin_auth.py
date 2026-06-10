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

from copy import deepcopy
from typing import TYPE_CHECKING

from nvflare.fuel.sec import authn

if TYPE_CHECKING:
    from nvflare.lighter.entity import Project

AUTH_KEY = "auth"
AUTH_TYPE_CERT = authn.ADMIN_AUTH_TYPE_CERT
AUTH_TYPE_OIDC = authn.ADMIN_AUTH_TYPE_OIDC
DEFAULT_OIDC_ADMIN_KIT_NAME = "admin"
OIDC_ADMIN_KIT_NAME_KEY = "admin_kit_name"


def get_admin_auth_config(project: "Project") -> dict:
    """Return the project's auth.admin config section.

    Delegates to the server runtime's fail-closed parser (authn.get_admin_auth_config):
    an 'auth' or 'auth.admin' section that is present but not a mapping raises ValueError
    instead of silently provisioning a cert-mode deployment.
    """
    try:
        admin_auth_config = authn.get_admin_auth_config({AUTH_KEY: project.get_prop(AUTH_KEY)})
    except authn.AuthError as ex:
        raise ValueError(f"invalid project config: {ex}")
    return deepcopy(dict(admin_auth_config))


def get_admin_auth_type(project: "Project") -> str:
    try:
        return authn.get_admin_auth_type({AUTH_KEY: project.get_prop(AUTH_KEY)})
    except authn.AuthError as ex:
        raise ValueError(f"invalid project config: {ex}")


def is_oidc_admin_auth(project: "Project") -> bool:
    return get_admin_auth_type(project) == AUTH_TYPE_OIDC


def is_oidc_admin_auth_config(admin_auth_config: dict) -> bool:
    """Check whether an already-extracted admin auth config section is of type 'oidc'."""
    return str(admin_auth_config.get("type", "")).strip().lower() == AUTH_TYPE_OIDC


def get_oidc_admin_kit_name(project: "Project") -> str:
    admin_auth_config = get_admin_auth_config(project)
    kit_name = admin_auth_config.get(OIDC_ADMIN_KIT_NAME_KEY, DEFAULT_OIDC_ADMIN_KIT_NAME)
    kit_name = str(kit_name).strip()
    return kit_name or DEFAULT_OIDC_ADMIN_KIT_NAME
