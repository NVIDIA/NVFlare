# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
import threading
from typing import List, Mapping, Optional

from nvflare.apis.fl_constant import SystemConfigs
from nvflare.apis.job_def import DEFAULT_STUDY
from nvflare.apis.utils.format_check import name_check
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import InternalCommands, ReplyKeyword
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandSpec
from nvflare.fuel.hci.security import IdentityKey, get_identity_info
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.sec.authn import (
    ADMIN_AUTH_TYPE_CERT,
    ADMIN_AUTH_TYPE_OIDC,
    AUTH_ERROR_CODE_CONFIG_INVALID,
    AUTH_ERROR_CODE_NOT_CONFIGURED,
    AUTH_ERROR_CODE_UNAVAILABLE,
    SUPPORTED_ADMIN_AUTH_TYPES,
    AuthError,
    get_admin_auth_config,
)
from nvflare.fuel.sec.principal import AUTH_METHOD_CERT, Principal
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.lighter.utils import cert_to_dict, load_crt_bytes
from nvflare.security.logging import secure_format_exception
from nvflare.security.study_registry import StudyRegistryService

from .reg import CommandFilter
from .sess import Session, SessionManager


class LoginModule(CommandModule, CommandFilter):
    def __init__(self, sess_mgr: SessionManager, oidc_auth_provider=None, admin_auth_config: Optional[Mapping] = None):
        """Login module.

        CommandModule containing the login commands to handle login and logout of admin clients, as well as the
        CommandFilter pre_command to check that a client is logged in with a valid session.

        Args:
            sess_mgr: SessionManager
        """
        if not isinstance(sess_mgr, SessionManager):
            raise TypeError("sess_mgr must be SessionManager but got {}.".format(type(sess_mgr)))

        self.session_mgr = sess_mgr
        self.oidc_auth_provider = oidc_auth_provider
        self.oidc_auth_provider_lock = threading.Lock()
        self.admin_auth_config = admin_auth_config
        self.logger = get_obj_logger(self)

    def get_spec(self):
        return CommandModuleSpec(
            name="login",
            cmd_specs=[
                CommandSpec(
                    name=InternalCommands.CERT_LOGIN,
                    description="login to server with SSL cert",
                    usage="login userName",
                    handler_func=self.handle_cert_login,
                    visible=False,
                ),
                CommandSpec(
                    name=InternalCommands.OIDC_LOGIN,
                    description="login to server with OIDC token set",
                    usage="oidc_login",
                    handler_func=self.handle_oidc_login,
                    visible=False,
                ),
                CommandSpec(
                    name=InternalCommands.LOGOUT,
                    description="logout from server",
                    usage="logout",
                    handler_func=self.handle_logout,
                    visible=False,
                ),
            ],
        )

    @staticmethod
    def _reject(conn: Connection, reason: str = "", code: str = ""):
        if code:
            conn.append_string(f"REJECT: {code}: {reason}" if reason else f"REJECT: {code}")
        else:
            conn.append_string(f"REJECT: {reason}" if reason else "REJECT")

    def _validate_login_study(self, conn: Connection, principal: Principal, study) -> bool:
        user_name = principal.policy_name()
        if not isinstance(study, str):
            self._reject(conn, "study must be a string", code="AUTH_INVALID_STUDY")
            return False

        invalid, _ = name_check(study, "study")
        if invalid:
            self._reject(conn, f"invalid study name '{study}'", code="AUTH_INVALID_STUDY_NAME")
            return False

        registry = StudyRegistryService.get_registry()
        if study != DEFAULT_STUDY:
            if not registry:
                self.logger.warning(f"rejecting login for user '{user_name}': no study registry for study '{study}'")
                self._reject(conn, f"study '{study}' is not configured on the server", code="AUTH_STUDY_NOT_CONFIGURED")
                return False
            if not registry.has_study(study):
                self.logger.warning(f"rejecting login for user '{user_name}': unknown study '{study}'")
                self._reject(conn, f"unknown study '{study}'", code="AUTH_UNKNOWN_STUDY")
                return False
            if not registry.has_user(user_name, study):
                self.logger.warning(f"rejecting login for user '{user_name}': no mapping for study '{study}'")
                self._reject(
                    conn,
                    f"user '{user_name}' is not mapped to study '{study}'",
                    code="AUTH_STUDY_USER_NOT_MAPPED",
                )
                return False

        return True

    def _create_session_and_reply(self, conn: Connection, principal, study: str):
        hci = conn.get_prop(ConnProps.HCI_SERVER)
        id_asserter = hci.get_id_asserter()

        request = conn.get_prop(ConnProps.REQUEST)
        assert isinstance(request, CellMessage)
        origin = request.get_header(MessageHeaderKey.ORIGIN)

        session = self.session_mgr.create_session_from_principal(
            principal=principal,
            origin_fqcn=origin,
            active_study=study,
        )
        token = session.make_token(id_asserter)
        self.logger.info(f"Created user session for {principal.policy_name()}")
        conn.append_string("OK")
        conn.append_token(token)

    def _get_oidc_admin_config(self):
        if self._get_admin_auth_type() != ADMIN_AUTH_TYPE_OIDC:
            return None

        oidc_config = self._get_admin_auth_config().get("oidc")
        if isinstance(oidc_config, Mapping):
            return dict(oidc_config)

        return None

    def _get_admin_auth_config(self) -> Mapping:
        if self.admin_auth_config is not None:
            # injected config is the already-extracted auth.admin section; fail closed like
            # the shared startup-config parser when it is not a mapping
            if not isinstance(self.admin_auth_config, Mapping):
                raise AuthError(
                    "invalid admin auth config: must be a mapping",
                    code=AUTH_ERROR_CODE_CONFIG_INVALID,
                )
            return self.admin_auth_config
        return get_admin_auth_config(ConfigService.get_section(SystemConfigs.STARTUP_CONF) or {})

    def _get_admin_auth_type(self) -> str:
        admin_auth_config = self._get_admin_auth_config()
        auth_type = str(admin_auth_config.get("type", ADMIN_AUTH_TYPE_CERT) or ADMIN_AUTH_TYPE_CERT).strip().lower()
        if auth_type not in SUPPORTED_ADMIN_AUTH_TYPES:
            raise AuthError(
                f"unsupported admin authentication type '{auth_type}'",
                code=AUTH_ERROR_CODE_CONFIG_INVALID,
            )
        return auth_type

    def _get_oidc_auth_provider(self):
        if self.oidc_auth_provider:
            return self.oidc_auth_provider

        with self.oidc_auth_provider_lock:
            if self.oidc_auth_provider:
                return self.oidc_auth_provider

            oidc_config = self._get_oidc_admin_config()
            if not oidc_config:
                raise AuthError("OIDC admin authentication is not configured", code=AUTH_ERROR_CODE_NOT_CONFIGURED)

            try:
                from nvflare.fuel.sec.oidc import OidcAdminAuthConfig, OidcAdminAuthProvider
            except ImportError as ex:
                raise AuthError(
                    "OIDC admin authentication provider is unavailable", code=AUTH_ERROR_CODE_UNAVAILABLE
                ) from ex

            try:
                self.oidc_auth_provider = OidcAdminAuthProvider(OidcAdminAuthConfig.from_dict(oidc_config))
            except AuthError:
                raise
            except Exception as ex:
                raise AuthError(
                    f"invalid OIDC admin authentication configuration: {ex}", code=AUTH_ERROR_CODE_CONFIG_INVALID
                ) from ex
            return self.oidc_auth_provider

    def handle_cert_login(self, conn: Connection, args: List[str]):
        try:
            auth_type = self._get_admin_auth_type()
        except AuthError as ex:
            self._reject(conn, str(ex), code="AUTH_CONFIG_INVALID")
            return

        if auth_type == ADMIN_AUTH_TYPE_OIDC:
            self._reject(conn, "certificate admin login is disabled", code="AUTH_CERT_DISABLED")
            return

        if len(args) != 2:
            self._reject(conn)
            return

        user_name = args[1]
        headers = conn.get_prop(ConnProps.CMD_HEADERS)
        cert_data = headers.get("cert")
        signature = headers.get("signature")
        study = headers.get("study", DEFAULT_STUDY)

        self.logger.debug(f"got cert login headers: {headers=}")
        hci = conn.get_prop(ConnProps.HCI_SERVER)
        identity_verifier = hci.get_id_verifier()

        cert = load_crt_bytes(cert_data)

        try:
            ok = identity_verifier.verify_common_name(
                asserter_cert=cert,
                asserted_cn=user_name,
                signature=signature,
                nonce="",
            )
            if not ok:
                raise AuthError("certificate identity verification failed")

            cert_dict = cert_to_dict(cert)
            self.logger.debug(f"got cert dict: {cert_dict}")
            identity = get_identity_info(cert_dict)
            principal = Principal.from_legacy_admin(
                username=user_name,
                subject=identity.get(IdentityKey.NAME) or user_name,
                org=identity.get(IdentityKey.ORG) or "",
                role=identity.get(IdentityKey.ROLE) or "",
                issuer="certificate",
                auth_method=AUTH_METHOD_CERT,
            )
            self.logger.debug("verified certificate identity for admin user '%s'", user_name)
        except AuthError as ex:
            self.logger.error(f"cert admin authentication failed: {secure_format_exception(ex)}")
            self._reject(conn)
            return
        except Exception as ex:
            self.logger.error(f"cert admin authentication failed: {secure_format_exception(ex)}")
            self.logger.debug("cert admin authentication failure cause", exc_info=ex)
            self._reject(conn)
            return

        if not self._validate_login_study(conn, principal, study):
            return

        self._create_session_and_reply(conn, principal, study)

    def handle_oidc_login(self, conn: Connection, args: List[str]):
        if len(args) != 1:
            self._reject(conn)
            return

        headers = conn.get_prop(ConnProps.CMD_HEADERS) or {}
        id_token = headers.get("id_token")
        if not id_token:
            self._reject(conn, "missing OIDC id_token", code="AUTH_OIDC_MISSING_TOKEN")
            return

        try:
            principal = self._get_oidc_auth_provider().authenticate(id_token)
            self.logger.info(
                "verified OIDC identity for admin user '%s' with effective role '%s'",
                principal.policy_name(),
                principal.policy_role(),
            )
        except AuthError as ex:
            self.logger.error(f"OIDC admin authentication failed: {secure_format_exception(ex)}")
            if ex.__cause__:
                self.logger.debug("OIDC admin authentication failure cause", exc_info=ex.__cause__)
            message = str(ex)
            error_code = getattr(ex, "code", "")
            if error_code == AUTH_ERROR_CODE_CONFIG_INVALID:
                self._reject(conn, message, code="AUTH_CONFIG_INVALID")
            elif error_code == AUTH_ERROR_CODE_NOT_CONFIGURED:
                self._reject(conn, "OIDC admin authentication is not configured", code="AUTH_OIDC_NOT_CONFIGURED")
            elif error_code == AUTH_ERROR_CODE_UNAVAILABLE:
                self._reject(conn, message, code="AUTH_OIDC_UNAVAILABLE")
            else:
                # every other AuthError on this path carries AUTH_ERROR_CODE_INVALID_TOKEN
                self._reject(conn, "OIDC token rejected", code="AUTH_OIDC_INVALID_TOKEN")
            return

        study = headers.get("study", DEFAULT_STUDY)
        if not self._validate_login_study(conn, principal, study):
            return

        self._create_session_and_reply(conn, principal, study)

    def handle_logout(self, conn: Connection, args: List[str]):
        if self.session_mgr:
            token = conn.get_prop(ConnProps.TOKEN)
            if token:
                self.session_mgr.end_session_by_token(token)
        conn.append_string("OK")

    def pre_command(self, conn: Connection, args: List[str]):
        if args[0] in [InternalCommands.CERT_LOGIN, InternalCommands.OIDC_LOGIN, InternalCommands.CHECK_SESSION]:
            # skip login and check session commands
            return True

        # validate token
        token = conn.get_token()
        if token is None:
            conn.append_error("not authenticated - no token")
            return False

        hci = conn.get_prop(ConnProps.HCI_SERVER)
        id_asserter = hci.get_id_asserter()

        sess = self.session_mgr.get_session(token, id_asserter)
        if not sess:
            # try to recreate the session
            request = conn.get_prop(ConnProps.REQUEST)
            assert isinstance(request, CellMessage)
            origin = request.get_header(MessageHeaderKey.ORIGIN)

            try:
                sess = self.session_mgr.recreate_session(token, origin, id_asserter)
                self.logger.info(f"recreated admin session for {sess.user_name}")
            except Exception as ex:
                self.logger.error(f"cannot recreate admin session: {secure_format_exception(ex)}")
                conn.append_error(ReplyKeyword.SESSION_INACTIVE)
                conn.append_string(
                    "user not authenticated or session timed out after {} seconds of inactivity - logged out".format(
                        self.session_mgr.idle_timeout
                    )
                )
                return False

        assert isinstance(sess, Session)
        sess.mark_active()
        conn.set_prop(ConnProps.SESSION, sess)
        conn.set_prop(ConnProps.USER_NAME, sess.user_name)
        conn.set_prop(ConnProps.USER_ORG, sess.user_org)
        conn.set_prop(ConnProps.USER_ROLE, sess.user_role)
        conn.set_prop(ConnProps.USER_PRINCIPAL, sess.principal)
        conn.set_prop(ConnProps.ACTIVE_STUDY, sess.active_study)
        conn.set_prop(ConnProps.TOKEN, token)
        return True

    def close(self):
        self.session_mgr.shutdown()
