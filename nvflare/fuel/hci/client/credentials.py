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

"""Credential strategies for admin client authentication.

Every behavioral difference between admin authentication methods (cert-based vs OIDC) is
localized here: AdminAPI selects one AdminCredentials object at construction time (via
make_admin_credentials) and delegates to it at every point where the auth methods differ —
config validation, user name resolution, identity assertion/verification, login, and
logout/auth-reject cleanup.

To add a new admin authentication method, implement one new AdminCredentials subclass and
map it in make_admin_credentials(). Do NOT add auth-type checks to api.py or cli.py.
"""

from abc import ABC, abstractmethod

from nvflare.apis.fl_constant import ConnectionSecurity
from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.hci.proto import InternalCommands, ProtoKey
from nvflare.fuel.sec.authn import ADMIN_AUTH_TYPE_CERT, ADMIN_AUTH_TYPE_OIDC, SUPPORTED_ADMIN_AUTH_TYPES
from nvflare.private.fed.utils.identity_utils import IdentityAsserter, get_cn_from_cert, load_cert_file
from nvflare.security.logging import secure_format_exception

from .api_spec import UidSource
from .api_status import APIStatus

OIDC_ADMIN_PLACEHOLDER_NAME = "oidc-admin"


class AdminCredentials(ABC):
    """Strategy that encapsulates how an admin client authenticates to the FL server.

    The methods take the AdminAPI object ("api") so the strategies can stay stateless;
    all configured values (certs, oidc_config, study, ...) live on the api.
    """

    # whether interactive clients (cli.py) should prompt the user for a user name
    prompts_for_username = True

    @abstractmethod
    def validate_config(self, api):
        """Validate the auth-method specific parts of the admin config.

        Raises:
            ConfigError: if the config is not valid for this auth method.
        """
        pass

    def resolve_user_name(self, configured_user_name: str, uid_source: str, client_cert) -> str:
        """Determine the user name to use, based on the configured name and uid_source."""
        if uid_source == UidSource.CERT:
            if not client_cert:
                raise ConfigError("uid_source 'cert' requires Client Cert file name")
            # We'll find the username from the client cert
            cert = load_cert_file(client_cert)
            return get_cn_from_cert(cert)
        return configured_user_name

    @abstractmethod
    def assert_client_identity(self, api) -> bool:
        """Whether to assert app-level client identity (cert CN) when authenticating the cell."""
        pass

    @abstractmethod
    def verify_server_identity(self, api) -> bool:
        """Whether to verify the server endpoint identity when authenticating the cell."""
        pass

    def prepare_login(self, api):
        """Per-login setup, done once before the (possibly retried) login attempts start."""
        pass

    def end_login(self, api):
        """Per-login cleanup, done once after the (possibly retried) login attempts end."""
        pass

    @abstractmethod
    def login(self, api) -> dict:
        """Perform one login attempt with the server.

        Returns:
            A dict of login status and details.
        """
        pass

    def on_logout(self, api):
        """Cleanup on logout. Must never raise."""
        pass


class CertAdminCredentials(AdminCredentials):
    """Admin authentication with the user's client certificate (signed common name)."""

    def validate_config(self, api):
        if not api.client_cert:
            raise ConfigError("missing Client Cert file name")
        if not api.client_key:
            raise ConfigError("missing Client Key file name")

    def assert_client_identity(self, api) -> bool:
        return bool(api.client_cert and api.client_key)

    def verify_server_identity(self, api) -> bool:
        return self.assert_client_identity(api)

    def login(self, api) -> dict:
        command = f"{InternalCommands.CERT_LOGIN} {api.user_name}"

        id_asserter = IdentityAsserter(private_key_file=api.client_key, cert_file=api.client_cert)
        cn_signature = id_asserter.sign_common_name(nonce="")

        headers = {
            "user_name": api.user_name,
            "cert": id_asserter.cert_data,
            "signature": cn_signature,
            "study": api.study,
        }
        return api._execute_login_command(command, headers)


class OidcAdminCredentials(AdminCredentials):
    """Admin authentication with an OIDC id_token from the configured identity provider.

    The user identity is established by the OIDC login command, not by a client cert:
    client certs (if any) remain in use for mTLS transport only.
    """

    # the user identity comes from the id_token; never ask the user for a name
    prompts_for_username = False

    def validate_config(self, api):
        conn_sec = str(api.conn_sec or "").lower()
        if bool(api.client_cert) != bool(api.client_key):
            raise ConfigError("OIDC admin config must specify both client_cert and client_key, or neither")
        if conn_sec == ConnectionSecurity.CLEAR:
            raise ConfigError("OIDC admin authentication requires TLS or mTLS admin transport")
        if not api.client_cert and not api.client_key and conn_sec != ConnectionSecurity.TLS:
            raise ConfigError(
                "OIDC admin authentication without a client private key requires connection_security 'tls'"
            )

    def resolve_user_name(self, configured_user_name: str, uid_source: str, client_cert) -> str:
        if uid_source != UidSource.CERT and not configured_user_name:
            return OIDC_ADMIN_PLACEHOLDER_NAME
        return super().resolve_user_name(configured_user_name, uid_source, client_cert)

    def assert_client_identity(self, api) -> bool:
        # The placeholder/user name cannot match the cert CN, so never assert app-level
        # client identity, even when client certs are configured (mTLS transport).
        return False

    def verify_server_identity(self, api) -> bool:
        # server endpoint identity must always be verified, even without client certs
        return True

    def prepare_login(self, api):
        # Obtain the id_token once and reuse it across retries: the token source may
        # involve interactive SSO, which must not be re-triggered by every retry.
        api._oidc_login_token = self.get_id_token(api)

    def end_login(self, api):
        # the id_token is not kept beyond the login attempt
        api._oidc_login_token = None

    def login(self, api) -> dict:
        id_token = getattr(api, "_oidc_login_token", None) or self.get_id_token(api)
        headers = {
            "id_token": id_token,
            "study": api.study,
        }

        resp = api._execute_login_command(InternalCommands.OIDC_LOGIN, headers)
        if resp.get(ProtoKey.STATUS) == APIStatus.ERROR_AUTHENTICATION:
            # the server rejected this token: don't replay it on the next login attempt
            api._oidc_login_token = None
            self.clear_cached_token(api)
        return resp

    def get_id_token(self, api):
        from nvflare.fuel.sec.oidc import OidcTokenSource

        def _prompt(authorization_url):
            api._print_hci(f"Open this URL to authenticate with SSO: {authorization_url}")

        return OidcTokenSource(api.oidc_config).get_id_token(prompt_cb=_prompt)

    def clear_cached_token(self, api):
        """Remove the cached OIDC id_token so the next login fetches a fresh one.

        Called only when the server REJECTS the token. A normal logout must keep the
        cache: every one-shot CLI command (e.g. ``nvflare system ...``) closes its
        session, and clearing here would force a fresh browser SSO on each command.
        Must never raise.
        """
        try:
            from nvflare.fuel.sec.oidc import clear_cached_id_token

            clear_cached_id_token(api.oidc_config)
        except Exception as ex:
            try:
                api.debug(f"failed to clear cached OIDC id_token: {secure_format_exception(ex)}")
            except Exception:
                pass


def make_admin_credentials(auth_type: str) -> AdminCredentials:
    """The single place where an admin auth type is mapped to its credential strategy.

    An empty/missing auth_type defaults to cert; any other unrecognized value is a config
    error (fail closed) rather than silently falling back to cert authentication.
    """
    auth_type = str(auth_type or "").strip().lower() or ADMIN_AUTH_TYPE_CERT
    if auth_type not in SUPPORTED_ADMIN_AUTH_TYPES:
        raise ConfigError(f"unsupported admin authentication type '{auth_type}'")
    if auth_type == ADMIN_AUTH_TYPE_OIDC:
        return OidcAdminCredentials()
    return CertAdminCredentials()
