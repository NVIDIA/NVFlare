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
import json
import stat
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import parse_qs, urlparse

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from nvflare.fuel.sec.authn import AUTH_ERROR_CODE_CONFIG_INVALID, AUTH_ERROR_CODE_INVALID_TOKEN, AuthError
from nvflare.fuel.sec.oidc import (
    OidcAdminAuthConfig,
    OidcAdminAuthProvider,
    OidcAuthorizationCodeClient,
    OidcAuthorizationCodeConfig,
    OidcConfigError,
    OidcProviderMetadata,
    OidcProviderMetadataResolver,
    OidcTokenSource,
    _LoopbackAuthorizationCodeReceiver,
    clear_cached_id_token,
)
from nvflare.fuel.sec.principal import AUTH_METHOD_OIDC

ISSUER = "https://keycloak.example.com/realms/nvflare"
CLIENT_ID = "nvflare-admin"
JWKS_URL = f"{ISSUER}/protocol/openid-connect/certs"


def _new_key_and_jwks(kid="key-1"):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key()))
    jwk["kid"] = kid
    jwk["alg"] = "RS256"
    jwk["use"] = "sig"
    return private_key, {"keys": [jwk]}


def _token(private_key, kid="key-1", remove_claims=(), **overrides):
    now = int(time.time())
    claims = {
        "iss": ISSUER,
        "sub": "keycloak-subject",
        "aud": CLIENT_ID,
        "azp": CLIENT_ID,
        "exp": now + 300,
        "nbf": now - 10,
        "iat": now,
        "jti": "token-id",
        "preferred_username": "admin@example.com",
        "email": "admin@example.com",
        "org": "example",
        "realm_access": {"roles": ["flare_member", "flare_project_admin"]},
        "resource_access": {CLIENT_ID: {"roles": ["flare_lead"]}},
        "groups": ["/flare/project-admins"],
    }
    claims.update(overrides)
    for claim in remove_claims:
        claims.pop(claim, None)
    return jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": kid})


def _unsigned_id_token(exp, **claims):
    def _segment(data):
        return base64.urlsafe_b64encode(json.dumps(data).encode("utf-8")).rstrip(b"=").decode("ascii")

    return f"{_segment({'alg': 'none'})}.{_segment({'exp': exp, **claims})}."


class _StaticJwkClient:
    def __init__(self, jwks):
        self.jwks = jwks

    def get_signing_key_from_jwt(self, token):
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        keys = self.jwks.get("keys") or []
        if kid:
            for jwk in keys:
                if jwk.get("kid") == kid:
                    return jwt.PyJWK(jwk)
        elif len(keys) == 1:
            return jwt.PyJWK(keys[0])
        raise jwt.PyJWKClientError(f"signing key not found for kid '{kid}'")


def _provider(jwks, **config_overrides):
    config = {
        "issuer": ISSUER,
        "client_id": CLIENT_ID,
        "jwks_url": JWKS_URL,
        "role_mapping": {
            "roles": {
                "flare_project_admin": "project_admin",
                "flare_member": "member",
                "flare_lead": "lead",
            },
            "groups": {
                "/flare/project-admins": "project_admin",
            },
        },
    }
    config.update(config_overrides)
    auth_config = OidcAdminAuthConfig.from_dict(config)
    return OidcAdminAuthProvider(
        config=auth_config,
        jwk_client=_StaticJwkClient(jwks),
    )


def test_oidc_provider_validates_keycloak_token_and_maps_principal():
    private_key, jwks = _new_key_and_jwks()
    provider = _provider(jwks)

    principal = provider.authenticate(_token(private_key))

    assert principal.auth_method == AUTH_METHOD_OIDC
    assert principal.subject == "keycloak-subject"
    assert principal.username == "admin@example.com"
    assert principal.email == "admin@example.com"
    assert principal.org == "example"
    assert principal.raw_roles == ("flare_member", "flare_project_admin", "flare_lead")
    assert principal.groups == ("/flare/project-admins",)
    assert principal.effective_role == "project_admin"
    assert principal.issuer == ISSUER
    assert principal.token_id == "token-id"
    assert principal.token_exp


def test_oidc_admin_auth_config_discovers_jwks_url(monkeypatch):
    def _resolve(self, issuer):
        assert issuer == ISSUER
        return OidcProviderMetadata(jwks_url=JWKS_URL)

    monkeypatch.setattr("nvflare.fuel.sec.oidc.OidcProviderMetadataResolver.resolve", _resolve)

    config = OidcAdminAuthConfig.from_dict(
        {
            "issuer": ISSUER,
            "client_id": CLIENT_ID,
        }
    )

    assert config.jwks_url == JWKS_URL


@pytest.mark.parametrize("unsupported_key", ["audience", "userinfo", "allowed_algorithms"])
def test_oidc_admin_auth_config_rejects_unsupported_security_knobs(unsupported_key):
    with pytest.raises(OidcConfigError, match=f"unsupported OIDC admin auth config key: {unsupported_key}"):
        OidcAdminAuthConfig.from_dict(
            {
                "issuer": ISSUER,
                "client_id": CLIENT_ID,
                "jwks_url": JWKS_URL,
                unsupported_key: {"enabled": True} if unsupported_key == "userinfo" else "value",
            }
        )


def test_oidc_provider_rejects_bad_issuer():
    private_key, jwks = _new_key_and_jwks()
    provider = _provider(jwks)

    with pytest.raises(AuthError, match="token validation failed"):
        provider.authenticate(_token(private_key, iss="https://bad-issuer"))


def test_oidc_provider_rejects_bad_audience_and_azp():
    private_key, jwks = _new_key_and_jwks()
    provider = _provider(jwks)

    with pytest.raises(AuthError, match="audience or authorized party"):
        provider.authenticate(_token(private_key, aud="other", azp="other"))


def test_oidc_provider_rejects_authorized_party_when_audience_differs():
    private_key, jwks = _new_key_and_jwks()
    provider = _provider(jwks)

    with pytest.raises(AuthError, match="audience or authorized party"):
        provider.authenticate(_token(private_key, aud="account", azp=CLIENT_ID))


def test_oidc_provider_accepts_multi_audience_token_when_azp_matches_client():
    private_key, jwks = _new_key_and_jwks()
    provider = _provider(jwks)

    principal = provider.authenticate(_token(private_key, aud=[CLIENT_ID, "account"], azp=CLIENT_ID))

    assert principal.effective_role == "project_admin"


def test_oidc_provider_rejects_expired_token():
    private_key, jwks = _new_key_and_jwks()
    provider = _provider(jwks)

    with pytest.raises(AuthError, match="token validation failed"):
        provider.authenticate(_token(private_key, exp=int(time.time()) - 1))


@pytest.mark.parametrize("missing_claim", ["iss", "sub", "aud", "exp", "iat"])
def test_oidc_provider_rejects_missing_required_id_token_claim(missing_claim):
    private_key, jwks = _new_key_and_jwks()
    provider = _provider(jwks)

    with pytest.raises(AuthError, match="token validation failed"):
        provider.authenticate(_token(private_key, remove_claims=[missing_claim]))


def test_oidc_provider_rejects_unknown_signing_key_after_refresh():
    private_key, jwks = _new_key_and_jwks(kid="known-key")
    provider = _provider(jwks)

    with pytest.raises(AuthError, match="signing key not found"):
        provider.authenticate(_token(private_key, kid="unknown-key"))


def test_oidc_provider_rejects_token_without_mapped_role():
    private_key, jwks = _new_key_and_jwks()
    provider = _provider(
        jwks,
        role_mapping={
            "default_exact_name_mapping": False,
            "roles": {"flare_project_admin": "project_admin"},
        },
    )

    with pytest.raises(AuthError, match="no valid FLARE role"):
        provider.authenticate(
            _token(
                private_key,
                realm_access={"roles": ["unknown"]},
                resource_access={CLIENT_ID: {"roles": ["unknown-client-role"]}},
                groups=[],
            )
        )


def test_oidc_provider_requires_explicit_role_mapping_by_default():
    private_key, jwks = _new_key_and_jwks()
    provider = _provider(jwks, role_mapping={})

    with pytest.raises(AuthError, match="no valid FLARE role"):
        provider.authenticate(
            _token(
                private_key,
                realm_access={"roles": ["project_admin"]},
                resource_access={},
                groups=[],
            )
        )


def test_oidc_provider_can_opt_in_to_exact_name_role_mapping():
    private_key, jwks = _new_key_and_jwks()
    provider = _provider(jwks, role_mapping={"default_exact_name_mapping": True})

    principal = provider.authenticate(
        _token(
            private_key,
            realm_access={"roles": ["project_admin"]},
            resource_access={},
            groups=[],
        )
    )

    assert principal.effective_role == "project_admin"


def test_oidc_provider_maps_client_roles_for_dotted_client_id():
    dotted_client_id = "com.example.flare"
    private_key, jwks = _new_key_and_jwks()
    provider = _provider(
        jwks,
        client_id=dotted_client_id,
        role_mapping={"roles": {"flare_lead": "lead"}},
    )

    principal = provider.authenticate(
        _token(
            private_key,
            aud=dotted_client_id,
            azp=dotted_client_id,
            realm_access={},
            resource_access={dotted_client_id: {"roles": ["flare_lead"]}},
            groups=[],
        )
    )

    assert principal.raw_roles == ("flare_lead",)
    assert principal.effective_role == "lead"


def test_oidc_provider_accepts_configured_issuer_with_trailing_slash():
    private_key, jwks = _new_key_and_jwks()
    provider = _provider(jwks, issuer=f"{ISSUER}/")

    assert provider.config.issuer == ISSUER

    principal = provider.authenticate(_token(private_key))

    assert principal.issuer == ISSUER


def test_oidc_provider_rejects_non_numeric_time_claim():
    private_key, jwks = _new_key_and_jwks()
    provider = _provider(jwks)

    with pytest.raises(AuthError, match="invalid claim data") as exc_info:
        provider.authenticate(_token(private_key, auth_time="2026-06-09T00:00:00Z"))

    assert exc_info.value.code == AUTH_ERROR_CODE_INVALID_TOKEN


def test_auth_errors_carry_machine_readable_codes():
    private_key, jwks = _new_key_and_jwks()
    provider = _provider(jwks)

    with pytest.raises(AuthError) as token_error:
        provider.authenticate(_token(private_key, iss="https://bad-issuer"))
    assert token_error.value.code == AUTH_ERROR_CODE_INVALID_TOKEN

    with pytest.raises(AuthError) as role_error:
        provider.authenticate(_token(private_key, realm_access={}, resource_access={}, groups=[]))
    assert role_error.value.code == AUTH_ERROR_CODE_INVALID_TOKEN

    with pytest.raises(OidcConfigError) as config_error:
        OidcAdminAuthConfig.from_dict({"issuer": ISSUER, "client_id": CLIENT_ID, "audience": CLIENT_ID})
    assert isinstance(config_error.value, AuthError)
    assert config_error.value.code == AUTH_ERROR_CODE_CONFIG_INVALID


class _Response:
    def __init__(self, status_code, data):
        self.status_code = status_code
        self._data = data

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _HttpClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, data, timeout=None, **kwargs):
        self.calls.append((url, data, timeout, kwargs))
        return self.responses.pop(0)


class _CodeReceiver:
    redirect_uri = "http://127.0.0.1:8250/callback"

    def __init__(self):
        self.state = ""
        self.timeout = 0.0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return None

    def wait_for_code(self, state: str, timeout: float) -> str:
        self.state = state
        self.timeout = timeout
        return "authorization-code"


class _MetadataHttpClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, timeout):
        self.calls.append((url, timeout))
        return self.response


def test_provider_metadata_resolver_reads_well_known_configuration():
    http_client = _MetadataHttpClient(
        _Response(
            200,
            {
                "authorization_endpoint": f"{ISSUER}/protocol/openid-connect/auth",
                "jwks_uri": f"{ISSUER}/protocol/openid-connect/certs",
                "token_endpoint": f"{ISSUER}/protocol/openid-connect/token",
            },
        )
    )

    metadata = OidcProviderMetadataResolver(http_client=http_client, request_timeout=3.0).resolve(ISSUER)

    assert metadata.authorization_url == f"{ISSUER}/protocol/openid-connect/auth"
    assert metadata.jwks_url == f"{ISSUER}/protocol/openid-connect/certs"
    assert metadata.token_url == f"{ISSUER}/protocol/openid-connect/token"
    assert http_client.calls == [(f"{ISSUER}/.well-known/openid-configuration", 3.0)]


def test_oidc_config_rejects_non_https_provider_endpoint():
    with pytest.raises(OidcConfigError, match="must use https"):
        OidcAdminAuthConfig.from_dict(
            {
                "issuer": ISSUER,
                "client_id": CLIENT_ID,
                "jwks_url": "http://keycloak.example.com/realms/nvflare/certs",
            }
        )


def test_oidc_config_allows_loopback_http_for_local_development():
    config = OidcAdminAuthConfig.from_dict(
        {
            "issuer": "http://127.0.0.1:8080/realms/nvflare",
            "client_id": CLIENT_ID,
            "jwks_url": "http://127.0.0.1:8080/realms/nvflare/certs",
        }
    )

    assert config.issuer == "http://127.0.0.1:8080/realms/nvflare"


def test_provider_metadata_resolver_rejects_non_https_issuer():
    with pytest.raises(OidcConfigError, match="must use https"):
        OidcProviderMetadataResolver(http_client=_MetadataHttpClient(_Response(200, {}))).resolve(
            "http://keycloak.example.com/realms/nvflare"
        )


def test_provider_metadata_resolver_rejects_non_https_discovered_endpoint():
    http_client = _MetadataHttpClient(
        _Response(
            200,
            {
                "jwks_uri": "http://keycloak.example.com/realms/nvflare/certs",
            },
        )
    )

    with pytest.raises(OidcConfigError, match="must use https"):
        OidcProviderMetadataResolver(http_client=http_client).resolve(ISSUER)


def test_authorization_code_config_discovers_sso_endpoints(monkeypatch):
    def _resolve(self, issuer):
        assert issuer == ISSUER
        return OidcProviderMetadata(
            authorization_url=f"{ISSUER}/protocol/openid-connect/auth",
            token_url=f"{ISSUER}/protocol/openid-connect/token",
        )

    monkeypatch.setattr("nvflare.fuel.sec.oidc.OidcProviderMetadataResolver.resolve", _resolve)

    config = OidcAuthorizationCodeConfig.from_dict({"issuer": ISSUER, "client_id": CLIENT_ID})

    assert config.authorization_url == f"{ISSUER}/protocol/openid-connect/auth"
    assert config.token_url == f"{ISSUER}/protocol/openid-connect/token"


def test_authorization_code_config_rejects_confidential_client_options():
    with pytest.raises(OidcConfigError, match="unsupported OIDC admin login config key: client_secret_env"):
        OidcAuthorizationCodeConfig.from_dict(
            {
                "client_id": CLIENT_ID,
                "authorization_url": f"{ISSUER}/protocol/openid-connect/auth",
                "token_url": f"{ISSUER}/protocol/openid-connect/token",
                "client_secret_env": "UNSUPPORTED_SECRET_ENV",
            }
        )


def test_token_source_rejects_removed_access_token_config():
    with pytest.raises(OidcConfigError, match="unsupported OIDC admin login config key: access_token_env"):
        OidcTokenSource({"id_token_env": "NVFLARE_OIDC_ID_TOKEN", "access_token_env": "NVFLARE_OIDC_ACCESS_TOKEN"})


def test_authorization_code_client_exchanges_pkce_code_for_token_set():
    http_client = _HttpClient([_Response(200, {"id_token": "auth-code-id-token"})])
    client = OidcAuthorizationCodeClient(
        OidcAuthorizationCodeConfig(
            client_id=CLIENT_ID,
            authorization_url=f"{ISSUER}/protocol/openid-connect/auth",
            token_url=f"{ISSUER}/protocol/openid-connect/token",
        ),
        http_client=http_client,
    )

    token_response = client.exchange_code(
        code="authorization-code",
        code_verifier="pkce-verifier",
        redirect_uri="http://127.0.0.1:8250/callback",
    )

    assert token_response["id_token"] == "auth-code-id-token"
    assert http_client.calls[0][0] == f"{ISSUER}/protocol/openid-connect/token"
    assert http_client.calls[0][1]["grant_type"] == "authorization_code"
    assert http_client.calls[0][1]["code_verifier"] == "pkce-verifier"


class _NonceEchoHttpClient:
    """Returns an id_token echoing the nonce sent in the authorization request, like a real IdP."""

    def __init__(self, prompts):
        self.prompts = prompts
        self.calls = []

    def post(self, url, data, timeout=None, **kwargs):
        self.calls.append((url, data, timeout, kwargs))
        nonce = parse_qs(urlparse(self.prompts[0]).query)["nonce"][0]
        return _Response(200, {"id_token": _unsigned_id_token(int(time.time()) + 300, nonce=nonce)})


def test_authorization_code_client_builds_pkce_authorization_url():
    prompts = []
    http_client = _NonceEchoHttpClient(prompts)
    receiver = _CodeReceiver()
    client = OidcAuthorizationCodeClient(
        OidcAuthorizationCodeConfig(
            client_id=CLIENT_ID,
            authorization_url=f"{ISSUER}/protocol/openid-connect/auth",
            token_url=f"{ISSUER}/protocol/openid-connect/token",
            open_browser=False,
        ),
        http_client=http_client,
        code_receiver=receiver,
    )

    token_response = client.request_token(prompt_cb=lambda request: prompts.append(request))

    assert len(prompts) == 1
    query = parse_qs(urlparse(prompts[0]).query)
    assert query["client_id"] == [CLIENT_ID]
    assert query["redirect_uri"] == [receiver.redirect_uri]
    assert query["code_challenge_method"] == ["S256"]
    assert query["code_challenge"]
    assert query["state"] == [receiver.state]
    assert query["nonce"]
    assert receiver.timeout == 300.0
    claims = jwt.decode(token_response["id_token"], options={"verify_signature": False})
    assert claims["nonce"] == query["nonce"][0]


def _auth_code_client(http_client):
    return OidcAuthorizationCodeClient(
        OidcAuthorizationCodeConfig(
            client_id=CLIENT_ID,
            authorization_url=f"{ISSUER}/protocol/openid-connect/auth",
            token_url=f"{ISSUER}/protocol/openid-connect/token",
            open_browser=False,
        ),
        http_client=http_client,
        code_receiver=_CodeReceiver(),
    )


def test_authorization_code_client_rejects_id_token_with_wrong_nonce():
    http_client = _HttpClient(
        [_Response(200, {"id_token": _unsigned_id_token(int(time.time()) + 300, nonce="injected-nonce")})]
    )

    with pytest.raises(AuthError, match="nonce does not match"):
        _auth_code_client(http_client).request_token()


def test_authorization_code_client_rejects_id_token_without_nonce_claim():
    http_client = _HttpClient([_Response(200, {"id_token": _unsigned_id_token(int(time.time()) + 300)})])

    with pytest.raises(AuthError, match="nonce does not match"):
        _auth_code_client(http_client).request_token()


def test_loopback_receiver_ignores_stray_callback_requests():
    config = OidcAuthorizationCodeConfig(
        client_id=CLIENT_ID,
        authorization_url=f"{ISSUER}/protocol/openid-connect/auth",
        token_url=f"{ISSUER}/protocol/openid-connect/token",
        open_browser=False,
    )
    state = "expected-state"
    statuses = []

    def _get_status(url):
        try:
            with urllib.request.urlopen(url) as response:
                return response.status
        except urllib.error.HTTPError as ex:
            return ex.code

    def _browse(callback_url):
        # A stray parameterless GET and a stray error redirect with the wrong state must not
        # abort the login; the real redirect afterwards must still succeed.
        statuses.append(_get_status(callback_url))
        statuses.append(_get_status(f"{callback_url}?error=access_denied&state=wrong-state"))
        statuses.append(_get_status(f"{callback_url}?code=real-code&state={state}"))

    with _LoopbackAuthorizationCodeReceiver(config) as receiver:
        browser = threading.Thread(target=_browse, args=(receiver.redirect_uri,), daemon=True)
        browser.start()
        code = receiver.wait_for_code(state=state, timeout=10.0)
        browser.join(timeout=10.0)

    assert code == "real-code"
    assert statuses == [400, 400, 200]


def test_token_source_uses_authorization_code_sso_when_discovered(monkeypatch):
    def _resolve(self, issuer):
        assert issuer == ISSUER
        return OidcProviderMetadata(
            authorization_url=f"{ISSUER}/protocol/openid-connect/auth",
            token_url=f"{ISSUER}/protocol/openid-connect/token",
        )

    def _request_token(self, prompt_cb=None):
        prompt_cb(f"{ISSUER}/protocol/openid-connect/auth?client_id={CLIENT_ID}")
        return {"id_token": "auth-code-id-token"}

    prompts = []
    monkeypatch.setattr("nvflare.fuel.sec.oidc.OidcProviderMetadataResolver.resolve", _resolve)
    monkeypatch.setattr("nvflare.fuel.sec.oidc.OidcAuthorizationCodeClient.request_token", _request_token)

    id_token = OidcTokenSource(
        {
            "issuer": ISSUER,
            "client_id": CLIENT_ID,
        }
    ).get_id_token(prompt_cb=lambda authorization_url: prompts.append(authorization_url))

    assert id_token == "auth-code-id-token"
    assert prompts == [f"{ISSUER}/protocol/openid-connect/auth?client_id={CLIENT_ID}"]


def test_token_source_reuses_cached_id_token_without_browser_login(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    id_token = _unsigned_id_token(int(time.time()) + 300)
    request_calls = []

    def _request_token(self, prompt_cb=None):
        request_calls.append(self.config.client_id)
        return {"id_token": id_token}

    config = {
        "issuer": ISSUER,
        "client_id": CLIENT_ID,
        "authorization_url": f"{ISSUER}/protocol/openid-connect/auth",
        "token_url": f"{ISSUER}/protocol/openid-connect/token",
    }
    monkeypatch.setattr("nvflare.fuel.sec.oidc.OidcAuthorizationCodeClient.request_token", _request_token)

    assert OidcTokenSource(config).get_id_token() == id_token
    cache_files = list((home / ".nvflare" / "oidc_tokens").glob("*.json"))
    assert len(cache_files) == 1
    assert stat.S_IMODE(cache_files[0].stat().st_mode) == 0o600

    def _unexpected_request_token(self, prompt_cb=None):
        raise AssertionError("cached token should avoid authorization-code login")

    monkeypatch.setattr("nvflare.fuel.sec.oidc.OidcAuthorizationCodeClient.request_token", _unexpected_request_token)

    assert OidcTokenSource(config).get_id_token() == id_token
    assert request_calls == [CLIENT_ID]


def test_token_source_does_not_cache_near_expiry_id_token(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    id_tokens = [
        _unsigned_id_token(int(time.time()) + 30),
        _unsigned_id_token(int(time.time()) + 300),
    ]
    request_calls = []

    def _request_token(self, prompt_cb=None):
        request_calls.append(self.config.client_id)
        return {"id_token": id_tokens.pop(0)}

    config = {
        "issuer": ISSUER,
        "client_id": CLIENT_ID,
        "authorization_url": f"{ISSUER}/protocol/openid-connect/auth",
        "token_url": f"{ISSUER}/protocol/openid-connect/token",
    }
    monkeypatch.setattr("nvflare.fuel.sec.oidc.OidcAuthorizationCodeClient.request_token", _request_token)

    OidcTokenSource(config).get_id_token()
    OidcTokenSource(config).get_id_token()

    assert request_calls == [CLIENT_ID, CLIENT_ID]


def _login_config():
    return {
        "issuer": ISSUER,
        "client_id": CLIENT_ID,
        "authorization_url": f"{ISSUER}/protocol/openid-connect/auth",
        "token_url": f"{ISSUER}/protocol/openid-connect/token",
    }


def _seed_token_cache(monkeypatch, config, id_token, refresh_token):
    def _request_token(self, prompt_cb=None):
        return {"id_token": id_token, "refresh_token": refresh_token}

    monkeypatch.setattr("nvflare.fuel.sec.oidc.OidcAuthorizationCodeClient.request_token", _request_token)
    assert OidcTokenSource(config).get_id_token() == id_token

    def _unexpected_request_token(self, prompt_cb=None):
        raise AssertionError("refresh grant should avoid the interactive browser flow")

    monkeypatch.setattr("nvflare.fuel.sec.oidc.OidcAuthorizationCodeClient.request_token", _unexpected_request_token)


def _read_cache_file(home):
    cache_files = list((home / ".nvflare" / "oidc_tokens").glob("*.json"))
    assert len(cache_files) == 1
    return json.loads(cache_files[0].read_text())


def test_authorization_code_client_refresh_grant_posts_refresh_token_without_client_secret():
    fresh_id_token = _unsigned_id_token(int(time.time()) + 300)
    http_client = _HttpClient([_Response(200, {"id_token": fresh_id_token, "refresh_token": "rotated-rt"})])
    client = OidcAuthorizationCodeClient(
        OidcAuthorizationCodeConfig(
            client_id=CLIENT_ID,
            authorization_url=f"{ISSUER}/protocol/openid-connect/auth",
            token_url=f"{ISSUER}/protocol/openid-connect/token",
        ),
        http_client=http_client,
    )

    token_response = client.refresh_token("stored-refresh-token")

    assert token_response["id_token"] == fresh_id_token
    assert token_response["refresh_token"] == "rotated-rt"
    url, data, timeout, _kwargs = http_client.calls[0]
    assert url == f"{ISSUER}/protocol/openid-connect/token"
    assert data["grant_type"] == "refresh_token"
    assert data["refresh_token"] == "stored-refresh-token"
    assert "client_secret" not in data
    assert timeout == 10.0


def test_token_source_uses_refresh_grant_for_expired_id_token_without_browser(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    config = _login_config()
    _seed_token_cache(monkeypatch, config, _unsigned_id_token(int(time.time()) + 30), "initial-rt")

    # The renewed id_token carries no nonce claim: refresh-grant responses are accepted without one.
    renewed_id_token = _unsigned_id_token(int(time.time()) + 300)
    refresh_calls = []

    def _refresh_token(self, refresh_token):
        refresh_calls.append(refresh_token)
        return {"id_token": renewed_id_token, "refresh_token": "rotated-rt"}

    monkeypatch.setattr("nvflare.fuel.sec.oidc.OidcAuthorizationCodeClient.refresh_token", _refresh_token)

    assert OidcTokenSource(config).get_id_token() == renewed_id_token
    assert refresh_calls == ["initial-rt"]

    cache = _read_cache_file(home)
    assert cache["id_token"] == renewed_id_token
    assert cache["refresh_token"] == "rotated-rt"

    # The renewed id_token is now cached: the next login uses it directly, no refresh and no browser.
    def _unexpected_refresh_token(self, refresh_token):
        raise AssertionError("valid cached id_token should avoid the refresh grant")

    monkeypatch.setattr("nvflare.fuel.sec.oidc.OidcAuthorizationCodeClient.refresh_token", _unexpected_refresh_token)
    assert OidcTokenSource(config).get_id_token() == renewed_id_token


def test_token_source_keeps_old_refresh_token_when_idp_does_not_rotate(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    config = _login_config()
    _seed_token_cache(monkeypatch, config, _unsigned_id_token(int(time.time()) + 30), "initial-rt")

    renewed_id_token = _unsigned_id_token(int(time.time()) + 300)

    def _refresh_token(self, refresh_token):
        return {"id_token": renewed_id_token}

    monkeypatch.setattr("nvflare.fuel.sec.oidc.OidcAuthorizationCodeClient.refresh_token", _refresh_token)

    assert OidcTokenSource(config).get_id_token() == renewed_id_token
    assert _read_cache_file(home)["refresh_token"] == "initial-rt"


def test_token_source_drops_refresh_token_and_falls_back_to_browser_on_refresh_failure(
    tmp_path, monkeypatch, caplog, capsys
):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    config = _login_config()
    _seed_token_cache(monkeypatch, config, _unsigned_id_token(int(time.time()) + 30), "revoked-rt")

    def _refresh_token(self, refresh_token):
        raise AuthError("OIDC refresh-token request failed: invalid_grant: Token is not active")

    interactive_id_token = _unsigned_id_token(int(time.time()) + 300)
    request_calls = []

    def _request_token(self, prompt_cb=None):
        request_calls.append(self.config.client_id)
        return {"id_token": interactive_id_token}

    monkeypatch.setattr("nvflare.fuel.sec.oidc.OidcAuthorizationCodeClient.refresh_token", _refresh_token)
    monkeypatch.setattr("nvflare.fuel.sec.oidc.OidcAuthorizationCodeClient.request_token", _request_token)

    assert OidcTokenSource(config).get_id_token() == interactive_id_token
    assert request_calls == [CLIENT_ID]

    # The failed refresh token is dropped: the new cache holds only the interactive token set.
    cache = _read_cache_file(home)
    assert cache["id_token"] == interactive_id_token
    assert "refresh_token" not in cache

    # The refresh token must never appear in log or console output.
    assert "revoked-rt" not in caplog.text
    captured = capsys.readouterr()
    assert "revoked-rt" not in captured.out
    assert "revoked-rt" not in captured.err


def test_clear_cached_id_token_removes_refresh_token(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    config = _login_config()
    _seed_token_cache(monkeypatch, config, _unsigned_id_token(int(time.time()) + 300), "long-lived-rt")
    assert _read_cache_file(home)["refresh_token"] == "long-lived-rt"

    clear_cached_id_token(config)

    assert not list((home / ".nvflare" / "oidc_tokens").glob("*.json"))


def test_clear_cached_id_token_forces_new_login(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    request_calls = []

    def _request_token(self, prompt_cb=None):
        request_calls.append(self.config.client_id)
        return {"id_token": _unsigned_id_token(int(time.time()) + 300)}

    config = {
        "issuer": ISSUER,
        "client_id": CLIENT_ID,
        "authorization_url": f"{ISSUER}/protocol/openid-connect/auth",
        "token_url": f"{ISSUER}/protocol/openid-connect/token",
    }
    monkeypatch.setattr("nvflare.fuel.sec.oidc.OidcAuthorizationCodeClient.request_token", _request_token)

    OidcTokenSource(config).get_id_token()
    assert list((home / ".nvflare" / "oidc_tokens").glob("*.json"))

    clear_cached_id_token(config)
    assert not list((home / ".nvflare" / "oidc_tokens").glob("*.json"))
    clear_cached_id_token(config)  # clearing an already-empty cache must not raise

    OidcTokenSource(config).get_id_token()
    assert request_calls == [CLIENT_ID, CLIENT_ID]
