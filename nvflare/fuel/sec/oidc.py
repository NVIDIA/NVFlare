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

import base64
import hashlib
import json
import os
import secrets
import time
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from ipaddress import ip_address
from socket import timeout as SocketTimeout
from typing import Callable, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

from nvflare.fuel.sec.authn import (
    AUTH_ERROR_CODE_CONFIG_INVALID,
    AUTH_ERROR_CODE_INVALID_TOKEN,
    AUTH_ERROR_CODE_UNAVAILABLE,
    AuthError,
)
from nvflare.fuel.sec.principal import AUTH_METHOD_OIDC, Principal
from nvflare.fuel.sec.principal import to_str as _as_str
from nvflare.fuel.sec.principal import to_tuple as _as_tuple
from nvflare.fuel.utils.argument_utils import str2bool

OIDC_EXTRA_MESSAGE = "OIDC admin authentication requires installing the OIDC extra: nvflare[oidc]"
AUTH_CODE_GRANT_TYPE = "authorization_code"
DEFAULT_OIDC_ALGORITHMS = ("RS256", "ES256")
REQUIRED_ID_TOKEN_CLAIMS = ("iss", "sub", "aud", "exp", "iat")
FLARE_ROLE_PRECEDENCE = ("project_admin", "org_admin", "lead", "member")
FLARE_ROLES = frozenset(FLARE_ROLE_PRECEDENCE)


class OidcConfigError(AuthError, ValueError):
    """Raised when OIDC admin authentication configuration is invalid."""

    def __init__(self, message: str = "", code: str = AUTH_ERROR_CODE_CONFIG_INVALID):
        super().__init__(message, code)


class RoleMappingError(ValueError):
    """Raised when identity-provider role data cannot be mapped to a FLARE role."""


def _normalize_role(value) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return " ".join(value.lower().split())


class RoleMapper:
    """Maps identity-provider roles or groups to one effective FLARE role."""

    def __init__(
        self,
        role_mapping: Optional[Mapping[str, str]] = None,
        group_mapping: Optional[Mapping[str, str]] = None,
        precedence: Optional[Sequence[str]] = None,
        default_exact_name_mapping: bool = False,
    ):
        self.default_exact_name_mapping = bool(default_exact_name_mapping)
        self.precedence = tuple(_normalize_role(r) for r in (precedence or FLARE_ROLE_PRECEDENCE))
        self._validate_precedence()
        self.role_mapping = self._normalize_mapping(role_mapping or {})
        self.group_mapping = self._normalize_mapping(group_mapping or {})

    def _validate_precedence(self):
        if not self.precedence:
            raise RoleMappingError("role precedence must not be empty")
        unknown = [r for r in self.precedence if r not in FLARE_ROLES]
        if unknown:
            raise RoleMappingError(f"invalid FLARE role in precedence: {unknown[0]}")
        if len(set(self.precedence)) != len(self.precedence):
            raise RoleMappingError("role precedence must not contain duplicates")

    @staticmethod
    def _normalize_mapping(mapping: Mapping[str, str]) -> dict:
        result = {}
        for source_role, flare_role in mapping.items():
            if not isinstance(source_role, str):
                raise RoleMappingError(f"mapping source role must be str but got {type(source_role)}")
            mapped_role = _normalize_role(flare_role)
            if mapped_role not in FLARE_ROLES:
                raise RoleMappingError(f"mapping target role must be a FLARE role but got {flare_role}")
            result[source_role] = mapped_role
        return result

    def require_effective_role(self, raw_roles=None, groups=None) -> str:
        mapped_roles = []

        for raw_role in _as_tuple(raw_roles):
            if raw_role in self.role_mapping:
                mapped_roles.append(self.role_mapping[raw_role])
                continue

            normalized_role = _normalize_role(raw_role)
            if self.default_exact_name_mapping and normalized_role in FLARE_ROLES:
                mapped_roles.append(normalized_role)

        for group in _as_tuple(groups):
            if group in self.group_mapping:
                mapped_roles.append(self.group_mapping[group])

        for role in self.precedence:
            if role in mapped_roles:
                return role
        raise RoleMappingError("no valid FLARE role mapped from identity-provider roles or groups")


@dataclass(frozen=True)
class OidcProviderMetadata:
    jwks_url: str = ""
    authorization_url: str = ""
    token_url: str = ""


class OidcProviderMetadataResolver:
    def __init__(self, http_client=None, request_timeout: float = 10.0):
        self.http_client = http_client
        self.request_timeout = request_timeout

    def resolve(self, issuer: str) -> OidcProviderMetadata:
        issuer = _as_str(issuer).rstrip("/")
        if not issuer:
            raise OidcConfigError("OIDC issuer is required for provider discovery")
        _validate_oidc_url(issuer, "issuer")

        metadata = self._get_metadata(f"{issuer}/.well-known/openid-configuration")
        result = OidcProviderMetadata(
            jwks_url=_as_str(metadata.get("jwks_uri")),
            authorization_url=_as_str(metadata.get("authorization_endpoint")),
            token_url=_as_str(metadata.get("token_endpoint")),
        )
        for field_name in (
            "jwks_url",
            "authorization_url",
            "token_url",
        ):
            url = getattr(result, field_name)
            if url:
                _validate_oidc_url(url, field_name)
        return result

    def _get_metadata(self, url: str) -> Mapping:
        http_client = self.http_client or _require_requests()
        response = http_client.get(url, timeout=self.request_timeout)
        response.raise_for_status()
        metadata = response.json()
        if not isinstance(metadata, Mapping):
            raise OidcConfigError("OIDC discovery endpoint returned invalid metadata")
        return metadata


@dataclass(frozen=True)
class OidcAdminAuthConfig:
    issuer: str
    client_id: str
    jwks_url: str
    role_mapper: RoleMapper = field(default_factory=lambda: RoleMapper(default_exact_name_mapping=False))
    group_claims: Tuple[str, ...] = ("groups", "member_of")
    role_claims: Tuple[str, ...] = ("roles", "realm_access.roles", "resource_access.${client_id}.roles")
    jwks_cache_ttl: float = 300.0
    jwks_request_timeout: float = 10.0

    def __post_init__(self):
        if not self.issuer:
            raise OidcConfigError("OIDC issuer is required")
        if not self.client_id:
            raise OidcConfigError("OIDC client_id is required")
        if not self.jwks_url:
            raise OidcConfigError("OIDC jwks_url is required")
        _validate_oidc_url(self.issuer, "issuer")
        _validate_oidc_url(self.jwks_url, "jwks_url")

    @classmethod
    def from_dict(cls, config: Mapping):
        if not isinstance(config, Mapping):
            raise OidcConfigError(f"OIDC admin auth config must be a mapping but got {type(config)}")
        _reject_unsupported_admin_auth_config(config)

        metadata = _discover_metadata_if_needed(
            config,
            required_keys=(("jwks_url", "jwks_uri"),),
        )
        client_id = _as_str(config.get("client_id"))
        role_mapping = config.get("role_mapping") or {}
        return cls(
            # Normalized to match discovery, which also strips the trailing slash.
            issuer=_as_str(config.get("issuer")).rstrip("/"),
            client_id=client_id,
            jwks_url=_config_str(config, "jwks_url", "jwks_uri") or metadata.jwks_url,
            role_mapper=_role_mapper_from_config(role_mapping),
            group_claims=_as_tuple(config.get("group_sources") or ("groups", "member_of")),
            role_claims=_as_tuple(
                config.get("role_sources") or ("roles", "realm_access.roles", "resource_access.${client_id}.roles")
            ),
            jwks_cache_ttl=float(config.get("jwks_cache_ttl", 300.0)),
            jwks_request_timeout=float(config.get("jwks_request_timeout", config.get("request_timeout", 10.0))),
        )


@dataclass(frozen=True)
class OidcAuthorizationCodeConfig:
    client_id: str
    authorization_url: str
    token_url: str
    scope: str = "openid profile email"
    redirect_uri: str = ""
    open_browser: bool = True
    timeout: float = 300.0
    request_timeout: float = 10.0

    def __post_init__(self):
        if not self.client_id:
            raise OidcConfigError("OIDC authorization-code flow client_id is required")
        if not self.authorization_url:
            raise OidcConfigError("OIDC authorization_url is required")
        if not self.token_url:
            raise OidcConfigError("OIDC token_url is required")
        _validate_oidc_url(self.authorization_url, "authorization_url")
        _validate_oidc_url(self.token_url, "token_url")

    @classmethod
    def from_dict(cls, config: Mapping):
        if not isinstance(config, Mapping):
            raise OidcConfigError(f"OIDC authorization-code config must be a mapping but got {type(config)}")
        _reject_unsupported_login_config(config)

        metadata = _discover_metadata_if_needed(
            config,
            required_keys=(("authorization_url", "authorization_endpoint"), ("token_url", "token_endpoint")),
        )
        return cls(
            client_id=_as_str(config.get("client_id")),
            authorization_url=_config_str(config, "authorization_url", "authorization_endpoint")
            or metadata.authorization_url,
            token_url=_config_str(config, "token_url", "token_endpoint") or metadata.token_url,
            scope=_as_str(config.get("scope", "openid profile email")),
            redirect_uri=_as_str(config.get("redirect_uri")),
            open_browser=_as_bool(config.get("open_browser", True)),
            timeout=float(config.get("timeout", 300.0)),
            request_timeout=float(config.get("request_timeout", 10.0)),
        )


class OidcAdminAuthProvider:
    """Validate OIDC login assertions and return normalized FLARE principals."""

    def __init__(
        self,
        config: OidcAdminAuthConfig,
        jwk_client=None,
    ):
        if not isinstance(config, OidcAdminAuthConfig):
            raise TypeError(f"config must be OidcAdminAuthConfig but got {type(config)}")

        self.config = config
        self.jwk_client = jwk_client or self._new_jwk_client(config)

    @staticmethod
    def _new_jwk_client(config: OidcAdminAuthConfig):
        jwt = _require_jwt()
        return jwt.PyJWKClient(
            config.jwks_url,
            cache_keys=True,
            cache_jwk_set=True,
            lifespan=config.jwks_cache_ttl,
            timeout=config.jwks_request_timeout,
        )

    def authenticate(self, id_token: str) -> Principal:
        if not id_token or not isinstance(id_token, str):
            raise AuthError("missing OIDC id_token", code=AUTH_ERROR_CODE_INVALID_TOKEN)
        claims = self._validate_token(id_token)
        raw_roles = self._extract_roles(claims)
        groups = self._extract_groups(claims)

        try:
            effective_role = self.config.role_mapper.require_effective_role(raw_roles=raw_roles, groups=groups)
        except RoleMappingError as ex:
            raise AuthError(str(ex), code=AUTH_ERROR_CODE_INVALID_TOKEN) from ex

        subject = _claim_str(claims, "sub")
        username = self._first_claim(claims, ("preferred_username", "email", "name")) or subject
        email = _claim_str(claims, "email")
        org = _claim_str(claims, "org")

        try:
            return Principal(
                subject=subject,
                username=username,
                email=email,
                org=org,
                raw_roles=raw_roles,
                groups=groups,
                issuer=_claim_str(claims, "iss"),
                token_id=_claim_str(claims, "jti"),
                auth_time=claims.get("auth_time", claims.get("iat")),
                token_exp=claims.get("exp"),
                effective_role=effective_role,
                auth_method=AUTH_METHOD_OIDC,
            )
        except (TypeError, ValueError) as ex:
            raise AuthError(f"OIDC token has invalid claim data: {ex}", code=AUTH_ERROR_CODE_INVALID_TOKEN) from ex

    def _validate_token(self, token: str) -> Mapping:
        jwt = _require_jwt()
        try:
            header = jwt.get_unverified_header(token)
            alg = _as_str(header.get("alg"))
            if alg not in DEFAULT_OIDC_ALGORITHMS:
                raise AuthError(
                    f"OIDC token uses unsupported signing algorithm '{alg}'", code=AUTH_ERROR_CODE_INVALID_TOKEN
                )
            key = self.jwk_client.get_signing_key_from_jwt(token).key
        except AuthError:
            raise
        except Exception as ex:
            raise AuthError(f"OIDC signing key lookup failed: {ex}", code=AUTH_ERROR_CODE_INVALID_TOKEN) from ex

        issuer = self.config.issuer.rstrip("/")
        try:
            claims = jwt.decode(
                token,
                key=key,
                algorithms=list(DEFAULT_OIDC_ALGORITHMS),
                # Accept either form so an OP whose canonical iss ends in '/' still matches.
                issuer=[issuer, issuer + "/"],
                options={
                    "verify_aud": False,
                    "require": list(REQUIRED_ID_TOKEN_CLAIMS),
                },
            )
        except AuthError:
            raise
        except Exception as ex:
            raise AuthError(f"OIDC token validation failed: {ex}", code=AUTH_ERROR_CODE_INVALID_TOKEN) from ex

        if not _token_bound_to_client(claims, client_id=self.config.client_id):
            raise AuthError(
                "OIDC token audience or authorized party does not match the configured FLARE client",
                code=AUTH_ERROR_CODE_INVALID_TOKEN,
            )
        return claims

    def _extract_roles(self, claims: Mapping) -> Tuple[str, ...]:
        roles = []
        for claim_path in self.config.role_claims:
            roles.extend(_claim_values(claims, _expand_claim_path(claim_path, self.config.client_id)))

        return _dedupe(roles)

    def _extract_groups(self, claims: Mapping) -> Tuple[str, ...]:
        groups = []
        for claim_path in self.config.group_claims:
            groups.extend(_claim_values(claims, _expand_claim_path(claim_path, self.config.client_id)))
        return _dedupe(groups)

    @staticmethod
    def _first_claim(claims: Mapping, claim_names: Sequence[str]) -> str:
        for claim_name in claim_names:
            value = _claim_str(claims, claim_name)
            if value:
                return value
        return ""


class OidcAuthorizationCodeClient:
    """Client-side OIDC authorization-code + PKCE helper for browser SSO."""

    def __init__(
        self,
        config: OidcAuthorizationCodeConfig,
        http_client=None,
        code_receiver=None,
    ):
        if not isinstance(config, OidcAuthorizationCodeConfig):
            raise TypeError(f"config must be OidcAuthorizationCodeConfig but got {type(config)}")

        self.config = config
        self.http_client = http_client
        self.code_receiver = code_receiver

    def request_token(self, prompt_cb: Optional[Callable[[str], None]] = None) -> Mapping:
        code_verifier = _new_pkce_verifier()
        nonce = secrets.token_urlsafe(32)
        receiver = self.code_receiver or _LoopbackAuthorizationCodeReceiver(self.config)

        with receiver:
            redirect_uri = receiver.redirect_uri
            oauth_client = self._new_oauth_session(redirect_uri=redirect_uri)
            authorization_url, state = oauth_client.create_authorization_url(
                self.config.authorization_url,
                code_verifier=code_verifier,
                nonce=nonce,
            )
            if self.config.open_browser:
                _open_browser(authorization_url)
            if prompt_cb:
                prompt_cb(authorization_url)
            code = receiver.wait_for_code(state=state, timeout=self.config.timeout)

        token_response = self.exchange_code(code=code, code_verifier=code_verifier, redirect_uri=redirect_uri)
        _verify_id_token_nonce(token_response, nonce)
        return token_response

    def refresh_token(self, refresh_token: str) -> Mapping:
        """Exchange a refresh token for a new token set (public PKCE client, no client secret)."""
        oauth_client = self._new_oauth_session(redirect_uri=self.config.redirect_uri)
        try:
            token_response = oauth_client.refresh_token(
                self.config.token_url,
                refresh_token=refresh_token,
                timeout=self.config.request_timeout,
            )
        except Exception as ex:
            # The error detail comes from the IdP error response; it never contains the refresh token.
            raise AuthError(f"OIDC refresh-token request failed: {_oauth_error_detail(ex)}") from ex

        if not isinstance(token_response, Mapping):
            raise AuthError("OIDC token endpoint returned invalid response data")
        return token_response

    def exchange_code(self, code: str, code_verifier: str, redirect_uri: str) -> Mapping:
        oauth_client = self._new_oauth_session(redirect_uri=redirect_uri)
        try:
            token_response = oauth_client.fetch_token(
                self.config.token_url,
                grant_type=AUTH_CODE_GRANT_TYPE,
                code=code,
                redirect_uri=redirect_uri,
                code_verifier=code_verifier,
                timeout=self.config.request_timeout,
            )
        except Exception as ex:
            raise AuthError(f"OIDC token endpoint request failed: {_oauth_error_detail(ex)}") from ex

        if not isinstance(token_response, Mapping):
            raise AuthError("OIDC token endpoint returned invalid response data")
        return token_response

    def _new_oauth_session(self, redirect_uri: str):
        OAuth2Session = _require_oauth2_session()
        oauth_client = OAuth2Session(
            client_id=self.config.client_id,
            scope=self.config.scope,
            redirect_uri=redirect_uri,
            token_endpoint_auth_method="none",
            code_challenge_method="S256",
            default_timeout=self.config.request_timeout,
        )
        if self.http_client:
            oauth_client.session = self.http_client
        return oauth_client


class OidcTokenSource:
    """Client-side source for OIDC admin token sets.

    The lookup order is deliberately simple: explicit token, configured
    environment variable, cached id_token, silent refresh-grant renewal,
    then interactive SSO through provider metadata.
    """

    def __init__(self, config: Mapping):
        if not isinstance(config, Mapping):
            raise OidcConfigError(f"OIDC token-source config must be a mapping but got {type(config)}")
        self.config = dict(config)
        _reject_unsupported_login_config(self.config)

    def get_id_token(self, prompt_cb: Optional[Callable[[str], None]] = None) -> str:
        id_token = _config_token(self.config, "id_token", "id_token_env")
        if id_token:
            return id_token

        cache = _read_token_cache(self.config)
        id_token = _cached_id_token(cache)
        if id_token:
            return id_token

        # Built lazily after a cache miss, then shared by the refresh and interactive paths;
        # from_dict may perform a discovery HTTP GET, so it must run at most once per call.
        code_config = OidcAuthorizationCodeConfig.from_dict(self.config)

        refresh_token = _as_str(cache.get("refresh_token"))
        if refresh_token:
            id_token = self._refresh_id_token(code_config, refresh_token)
            if id_token:
                return id_token

        token_response = OidcAuthorizationCodeClient(code_config).request_token(prompt_cb=prompt_cb)

        id_token = _as_str(token_response.get("id_token"))
        if not id_token:
            raise AuthError("OIDC token response did not contain id_token", code=AUTH_ERROR_CODE_INVALID_TOKEN)
        _save_cached_id_token(self.config, id_token, refresh_token=_as_str(token_response.get("refresh_token")))
        return id_token

    def _refresh_id_token(self, code_config: OidcAuthorizationCodeConfig, refresh_token: str) -> str:
        """Renew the id_token silently via the refresh grant; return "" to fall back to interactive SSO.

        No nonce check applies here: refresh-grant responses carry no nonce. The server still
        performs full signature/issuer/audience validation of the id_token at login time.
        Any failure (HTTP error, revoked/expired refresh token, missing id_token) drops the
        cached refresh token so the next attempt goes straight to the browser flow.
        """
        try:
            token_response = OidcAuthorizationCodeClient(code_config).refresh_token(refresh_token)
            id_token = _as_str(token_response.get("id_token"))
            if not id_token:
                raise AuthError("OIDC refresh response did not contain id_token")
        except Exception:
            # Never log token material; the cached refresh token is stale or revoked, so drop it.
            clear_cached_id_token(self.config)
            return ""

        # Keep the rotated refresh token if the IdP returned a new one, else keep the old one.
        _save_cached_id_token(
            self.config, id_token, refresh_token=_as_str(token_response.get("refresh_token")) or refresh_token
        )
        return id_token


def _discover_metadata_if_needed(config: Mapping, required_keys: Sequence[Sequence[str]]) -> OidcProviderMetadata:
    """Resolve provider metadata unless every required endpoint (key plus aliases) is configured."""
    if all(_config_str(config, *keys) for keys in required_keys):
        return OidcProviderMetadata()

    issuer = _as_str(config.get("issuer"))
    if not issuer:
        return OidcProviderMetadata()

    return OidcProviderMetadataResolver(request_timeout=float(config.get("request_timeout", 10.0))).resolve(issuer)


# Keys that are not supported in the server-side admin auth config (fed_server.json).
# Provisioning imports this list so provision-time and runtime validation cannot drift.
UNSUPPORTED_SERVER_ADMIN_AUTH_KEYS = (
    "access_token",
    "access_token_env",
    "audience",
    "allowed_algorithms",
    "authorization_code_timeout",
    "authorization_endpoint",
    "authorization_url",
    "callback_host",
    "callback_path",
    "callback_port",
    "client_secret",
    "client_secret_env",
    "device_authorization_url",
    "device_authorization_endpoint",
    "required_claims",
    "email_claim",
    "group_claims",
    "groups_claim",
    "groups_claims",
    "id_token",
    "id_token_env",
    "login_flow",
    "open_browser",
    "org_claim",
    "redirect_uri",
    "role_claims",
    "roles_claim",
    "roles_claims",
    "scope",
    "token_endpoint",
    "token_endpoint_auth_method",
    "token_url",
    "userinfo",
    "use_userinfo",
    "userinfo_required",
    "userinfo_url",
    "userinfo_endpoint",
    "username_claims",
    "default_org",
)

# OIDC config keys meaningful only to the admin client login flow (fed_admin.json).
# Provisioning strips these from the server-side config (fed_server.json); this module owns
# the partition so provision-time and runtime key handling cannot drift.
CLIENT_LOGIN_ONLY_OIDC_KEYS = frozenset(
    {
        "authorization_endpoint",
        "authorization_url",
        "id_token",
        "id_token_env",
        "open_browser",
        "redirect_uri",
        "scope",
        "timeout",
        "token_endpoint",
        "token_url",
    }
)

# Keys that are not supported in the client-side login config (fed_admin.json).
UNSUPPORTED_LOGIN_CONFIG_KEYS = (
    "access_token",
    "access_token_env",
    "authorization_code_timeout",
    "callback_host",
    "callback_path",
    "callback_port",
    "client_secret",
    "client_secret_env",
    "device_authorization_url",
    "device_authorization_endpoint",
    "login_flow",
    "token_endpoint_auth_method",
)


def _reject_unsupported_admin_auth_config(config: Mapping):
    for key in UNSUPPORTED_SERVER_ADMIN_AUTH_KEYS:
        if key in config:
            raise OidcConfigError(f"unsupported OIDC admin auth config key: {key}")


def _reject_unsupported_login_config(config: Mapping):
    for key in UNSUPPORTED_LOGIN_CONFIG_KEYS:
        if key in config:
            raise OidcConfigError(f"unsupported OIDC admin login config key: {key}")


def _role_mapper_from_config(config) -> RoleMapper:
    if not config:
        return RoleMapper(default_exact_name_mapping=False)
    if not isinstance(config, Mapping):
        raise OidcConfigError(f"OIDC role_mapping must be a mapping but got {type(config)}")

    supported_keys = {"roles", "groups", "precedence", "default_exact_name_mapping"}
    for key in config:
        if key not in supported_keys:
            raise OidcConfigError(f"unsupported OIDC role_mapping config key: {key}")

    role_mapping = config.get("roles") or {}
    if not isinstance(role_mapping, Mapping):
        raise OidcConfigError(f"OIDC role_mapping.roles must be a mapping but got {type(role_mapping)}")

    group_mapping = config.get("groups") or {}
    if not isinstance(group_mapping, Mapping):
        raise OidcConfigError(f"OIDC role_mapping.groups must be a mapping but got {type(group_mapping)}")

    return RoleMapper(
        role_mapping=role_mapping,
        group_mapping=group_mapping,
        precedence=config.get("precedence") or FLARE_ROLE_PRECEDENCE,
        default_exact_name_mapping=config.get("default_exact_name_mapping", False),
    )


class _LoopbackAuthorizationCodeReceiver:
    def __init__(self, config: OidcAuthorizationCodeConfig):
        self.config = config
        self.server = None
        self.redirect_uri = ""
        self.callback_path = ""

    def __enter__(self):
        host, port, path = self._callback_binding()
        self.callback_path = path
        self.server = HTTPServer((host, port), _OidcCallbackHandler)
        self.server.timeout = 1.0
        self.server.expected_path = path
        self.server.expected_state = ""
        self.server.callback_result = None
        self.server.callback_error = None
        bound_host, bound_port = self.server.server_address
        redirect_host = host if host != "0.0.0.0" else bound_host
        self.redirect_uri = self.config.redirect_uri or f"http://{redirect_host}:{bound_port}{path}"
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.server:
            self.server.server_close()

    def wait_for_code(self, state: str, timeout: float) -> str:
        if not self.server:
            raise AuthError("OIDC authorization-code receiver is not started")

        self.server.expected_state = state
        expires_at = time.monotonic() + timeout
        while time.monotonic() < expires_at:
            try:
                self.server.handle_request()
            except SocketTimeout:
                pass

            if self.server.callback_error:
                raise AuthError(self.server.callback_error)
            result = self.server.callback_result
            if result:
                if result.get("state") != state:
                    # Defensive: the handler only records matching state; keep waiting for the real redirect.
                    self.server.callback_result = None
                    continue
                code = _as_str(result.get("code"))
                if not code:
                    raise AuthError("OIDC authorization callback did not contain code")
                return code

        raise AuthError("OIDC authorization-code login timed out")

    def _callback_binding(self):
        if self.config.redirect_uri:
            parsed = urlparse(self.config.redirect_uri)
            if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
                raise OidcConfigError("OIDC redirect_uri for CLI login must be a localhost http URL")
            return parsed.hostname, parsed.port or 80, _normalize_callback_path(parsed.path or "/callback")
        return "127.0.0.1", 0, "/callback"


class _OidcCallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        expected_path = getattr(self.server, "expected_path", "")
        if expected_path and parsed.path != expected_path:
            self._write_response(404, "OIDC callback path not found")
            return

        params = parse_qs(parsed.query)
        state = _as_str(_first(params.get("state")))
        expected_state = _as_str(getattr(self.server, "expected_state", ""))
        if not expected_state or state != expected_state:
            # A stray or attacker-initiated request (even one carrying an 'error' param) must not
            # abort the login in progress; record nothing and keep waiting for the real redirect.
            self._write_response(400, "OIDC callback state did not match a login in progress")
            return

        error = _as_str(_first(params.get("error")))
        if error:
            error_description = _as_str(_first(params.get("error_description")))
            self.server.callback_error = f"OIDC authorization failed: {error_description or error}"
            self._write_response(400, "OIDC authorization failed. You can close this browser tab.")
            return

        self.server.callback_result = {
            "code": _as_str(_first(params.get("code"))),
            "state": state,
        }
        self._write_response(200, "OIDC authorization complete. You can close this browser tab.")

    def log_message(self, _format, *_args):
        return

    def _write_response(self, status: int, message: str):
        body = message.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _token_bound_to_client(claims: Mapping, client_id: str) -> bool:
    aud_values = _as_tuple(claims.get("aud"))
    authorized_party = _as_str(claims.get("azp"))
    if not client_id:
        return False
    if client_id not in aud_values:
        return False
    if len(aud_values) > 1 and authorized_party != client_id:
        return False
    return True


def _claim_str(claims: Mapping, claim_path: str) -> str:
    values = _claim_values(claims, claim_path)
    return values[0] if values else ""


def _claim_values(claims: Mapping, claim_path: str) -> Tuple[str, ...]:
    value = _claim_value(claims, claim_path)
    if isinstance(value, Mapping):
        return tuple(_as_str(key) for key in value.keys() if _as_str(key))
    return _as_tuple(value)


def _claim_value(claims: Mapping, claim_path):
    if not claim_path:
        return None

    parts = claim_path.split(".") if isinstance(claim_path, str) else claim_path
    value = claims
    for part in parts:
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def _as_bool(value) -> bool:
    if isinstance(value, str):
        # str2bool has no "on" token and returns None for unrecognized strings; config strings default to False.
        token = value.strip().lower()
        return token == "on" or str2bool(token) is True
    result = str2bool(value)
    return bool(value) if result is None else result


def _validate_oidc_url(url: str, field_name: str):
    parsed = urlparse(_as_str(url))
    if parsed.scheme == "https" and parsed.netloc:
        return
    if parsed.scheme == "http" and _is_loopback_host(parsed.hostname):
        return
    raise OidcConfigError(f"OIDC {field_name} must use https; http is only allowed for loopback local development")


def _is_loopback_host(hostname: str) -> bool:
    if not hostname:
        return False
    if hostname == "localhost":
        return True
    try:
        return ip_address(hostname).is_loopback
    except ValueError:
        return False


def _config_str(config: Mapping, key: str, *aliases: str) -> str:
    for candidate in (key, *aliases):
        value = _as_str(config.get(candidate))
        if value:
            return value
    return ""


def _config_token(config: Mapping, token_key: str, env_key: str) -> str:
    token = _as_str(config.get(token_key))
    token_env = _as_str(config.get(env_key))
    if token_env:
        token = _as_str(os.getenv(token_env)) or token
    return token


def _read_token_cache(config: Mapping) -> Mapping:
    try:
        with open(_token_cache_path(config), "r", encoding="utf-8") as f:
            cache = json.load(f)
    except Exception:
        return {}
    return cache if isinstance(cache, Mapping) else {}


def _cached_id_token(cache: Mapping) -> str:
    id_token = _as_str(cache.get("id_token"))
    expires_at = float(cache.get("expires_at", 0.0) or 0.0)
    if not id_token or expires_at <= time.time() + 60.0:
        return ""
    return id_token


def clear_cached_id_token(config: Mapping):
    """Remove any cached token set (id_token and refresh_token) for this OIDC config.

    Call on logout and when the server rejects the token, so a stale or revoked
    cached token is not silently replayed on the next login.
    """
    try:
        os.remove(_token_cache_path(config))
    except OSError:
        pass


def _save_cached_id_token(config: Mapping, id_token: str, refresh_token: str = ""):
    expires_at = _id_token_exp(id_token)
    if expires_at <= time.time() + 60.0 and not refresh_token:
        return

    cache_path = _token_cache_path(config)
    cache_dir = os.path.dirname(cache_path)
    os.makedirs(cache_dir, mode=0o700, exist_ok=True)
    try:
        os.chmod(cache_dir, 0o700)
    except Exception:
        pass

    cache = {"id_token": id_token, "expires_at": expires_at}
    if refresh_token:
        cache["refresh_token"] = refresh_token

    tmp_path = f"{cache_path}.tmp"
    data = json.dumps(cache, sort_keys=True)
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp_path, cache_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _token_cache_path(config: Mapping) -> str:
    cache_key = "\0".join(
        (
            _as_str(config.get("issuer")).rstrip("/"),
            _as_str(config.get("client_id")),
            _as_str(config.get("scope", "openid profile email")),
        )
    )
    cache_name = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    return os.path.join(os.path.expanduser("~"), ".nvflare", "oidc_tokens", f"{cache_name}.json")


def _id_token_exp(id_token: str) -> float:
    parts = _as_str(id_token).split(".")
    if len(parts) < 2:
        return 0.0
    payload = parts[1]
    payload += "=" * (-len(payload) % 4)
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload.encode("ascii")).decode("utf-8"))
    except Exception:
        return 0.0
    return float(claims.get("exp", 0.0) or 0.0) if isinstance(claims, Mapping) else 0.0


def _verify_id_token_nonce(token_response, nonce: str):
    """Verify the id_token echoes the nonce of this login attempt (token-injection defense).

    Signature and claim validation happen server-side; the client only confirms its own nonce
    round-tripped, so decoding without signature verification is sufficient here.
    """
    id_token = _as_str(token_response.get("id_token")) if isinstance(token_response, Mapping) else ""
    if not id_token:
        return

    jwt = _require_jwt()
    try:
        claims = jwt.decode(id_token, options={"verify_signature": False})
    except Exception as ex:
        raise AuthError(
            f"OIDC id_token could not be parsed for nonce verification: {ex}",
            code=AUTH_ERROR_CODE_INVALID_TOKEN,
        ) from ex

    if _as_str(claims.get("nonce")) != nonce:
        raise AuthError(
            "OIDC id_token nonce does not match this login attempt",
            code=AUTH_ERROR_CODE_INVALID_TOKEN,
        )


def _expand_claim_path(claim_path: str, client_id: str) -> Tuple[str, ...]:
    # Split before substitution so a dotted client_id stays a single path segment.
    return tuple(part.replace("${client_id}", client_id) for part in _as_str(claim_path).split("."))


def _normalize_callback_path(value) -> str:
    path = _as_str(value) or "/callback"
    if not path.startswith("/"):
        path = f"/{path}"
    return path


def _first(values):
    return values[0] if values else None


def _new_pkce_verifier() -> str:
    return _require_authlib_generate_token()(64)


def _oauth_error_detail(ex: Exception) -> str:
    error = _as_str(getattr(ex, "error", ""))
    description = _as_str(getattr(ex, "description", ""))
    if error and description:
        return f"{error}: {description}"
    return error or _as_str(ex)


def _dedupe(values) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(_as_str(v) for v in values if _as_str(v)))


def _open_browser(url: str):
    if url:
        webbrowser.open(url)


def _require_jwt():
    try:
        import jwt
    except ImportError as ex:
        raise AuthError(OIDC_EXTRA_MESSAGE, code=AUTH_ERROR_CODE_UNAVAILABLE) from ex
    return jwt


def _require_oauth2_session():
    try:
        from authlib.integrations.requests_client import OAuth2Session
    except ImportError as ex:
        raise AuthError(OIDC_EXTRA_MESSAGE, code=AUTH_ERROR_CODE_UNAVAILABLE) from ex
    return OAuth2Session


def _require_authlib_generate_token():
    try:
        from authlib.common.security import generate_token
    except ImportError as ex:
        raise AuthError(OIDC_EXTRA_MESSAGE, code=AUTH_ERROR_CODE_UNAVAILABLE) from ex
    return generate_token


def _require_requests():
    try:
        import requests
    except ImportError as ex:
        raise AuthError(OIDC_EXTRA_MESSAGE, code=AUTH_ERROR_CODE_UNAVAILABLE) from ex
    return requests
