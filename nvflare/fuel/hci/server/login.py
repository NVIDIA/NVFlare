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
import traceback
from typing import List

from cryptography.hazmat.primitives import hashes

from nvflare.apis.job_def import DEFAULT_STUDY
from nvflare.apis.utils.format_check import name_check
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import InternalCommands, ReplyKeyword
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandSpec
from nvflare.fuel.hci.security import IdentityKey, get_identity_info
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.sec.admin_audit import audit_context, record, reject, update_context
from nvflare.fuel.sec.admin_cert import get_admin_study_entitlements, validate_admin_leaf_cert
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.lighter.utils import cert_to_dict, load_crt_chain_bytes
from nvflare.security.logging import secure_format_exception
from nvflare.security.study_registry import StudyRegistryService

from .reg import CommandFilter
from .sess import Session, SessionManager


class LoginModule(CommandModule, CommandFilter):
    def __init__(self, sess_mgr: SessionManager):
        """Login module.

        CommandModule containing the login commands to handle login and logout of admin clients, as well as the
        CommandFilter pre_command to check that a client is logged in with a valid session.

        Args:
            sess_mgr: SessionManager
        """
        if not isinstance(sess_mgr, SessionManager):
            raise TypeError("sess_mgr must be SessionManager but got {}.".format(type(sess_mgr)))

        self.session_mgr = sess_mgr
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
                    name=InternalCommands.LOGOUT,
                    description="logout from server",
                    usage="logout",
                    handler_func=self.handle_logout,
                    visible=False,
                ),
            ],
        )

    def handle_cert_login(self, conn: Connection, args: List[str]):
        with audit_context(user="unknown"):
            try:
                self._handle_cert_login(conn, args)
            except Exception:
                record("admin.login.error", reason="processing_error")
                raise

    def _handle_cert_login(self, conn: Connection, args: List[str]):
        def _reject(reason: str = "", code: str = "", audit_reason: str = "verification_failed"):
            record("admin.login.rejected", reason=audit_reason)
            if code:
                conn.append_string(f"REJECT: {code}: {reason}" if reason else f"REJECT: {code}")
            else:
                conn.append_string(f"REJECT: {reason}" if reason else "REJECT")

        if len(args) != 2:
            _reject(audit_reason="invalid_arguments")
            return

        user_name = args[1]
        headers = conn.get_prop(ConnProps.CMD_HEADERS)
        cert_data = headers.get("cert")
        signature = headers.get("signature")
        study = headers.get("study", DEFAULT_STUDY)

        hci = conn.get_prop(ConnProps.HCI_SERVER)
        identity_verifier = hci.get_id_verifier()
        id_asserter = hci.get_id_asserter()
        if not identity_verifier or not id_asserter:
            self.logger.error("rejecting admin login because the server signing identity is not configured")
            _reject(
                "server signing identity is not configured",
                code="AUTH_SERVER_IDENTITY_UNAVAILABLE",
                audit_reason="server_identity_unavailable",
            )
            return

        cert_chain = load_crt_chain_bytes(cert_data)
        cert = cert_chain[0]
        try:
            ok = identity_verifier.verify_common_name(
                asserter_cert=cert,
                asserted_cn=user_name,
                signature=signature,
                nonce="",
                intermediate_certs=cert_chain[1:],
            )
            self.logger.debug(f"verify common name: {ok=}")
        except Exception as ex:
            self.logger.error(f"identity_verifier.verify_common_name got exception: {ex}")
            traceback.print_exc()
            record("admin.login.error", reason="verification_error")
            verification_error = True
            ok = False
        else:
            verification_error = False

        if not ok:
            if verification_error:
                conn.append_string("REJECT")
            else:
                _reject()
            return

        try:
            validate_admin_leaf_cert(cert)
            cert_studies = get_admin_study_entitlements(cert)
        except Exception as ex:
            self.logger.error(f"admin certificate validation failed: {secure_format_exception(ex)}")
            _reject(audit_reason="invalid_admin_certificate")
            return

        update_context(user=user_name, identity_source="locally_verified")
        if not isinstance(study, str):
            _reject("study must be a string", code="AUTH_INVALID_STUDY", audit_reason="invalid_study")
            return

        invalid, _ = name_check(study, "study")
        if invalid:
            _reject(f"invalid study name '{study}'", code="AUTH_INVALID_STUDY_NAME", audit_reason="invalid_study")
            return

        registry = StudyRegistryService.get_registry()
        if study != DEFAULT_STUDY:
            if not registry:
                self.logger.warning(f"rejecting login for user '{user_name}': no study registry for study '{study}'")
                _reject(
                    f"study '{study}' is not configured on the server",
                    code="AUTH_STUDY_NOT_CONFIGURED",
                    audit_reason="study_not_configured",
                )
                return
            if not registry.has_study(study):
                self.logger.warning(f"rejecting login for user '{user_name}': unknown study '{study}'")
                _reject(f"unknown study '{study}'", code="AUTH_UNKNOWN_STUDY", audit_reason="unknown_study")
                return
            if not registry.has_user(user_name, study) and study not in cert_studies:
                self.logger.warning(f"rejecting login for user '{user_name}': no mapping for study '{study}'")
                _reject(
                    f"user '{user_name}' is not mapped to study '{study}'",
                    code="AUTH_STUDY_USER_NOT_MAPPED",
                    audit_reason="study_membership_denied",
                )
                return

        cert_dict = cert_to_dict(cert)
        self.logger.debug(f"got cert dict: {cert_dict}")
        identity = get_identity_info(cert_dict)

        grant = conn.get_prop(ConnProps.ADMIN_TRANSPORT_GRANT)
        if not grant or grant.user_name != user_name or grant.cert_fingerprint != cert.fingerprint(hashes.SHA256()):
            _reject(audit_reason="admin_transport_identity_mismatch")
            return

        session, token = self.session_mgr.create_session(
            grant,
            user_org=identity.get(IdentityKey.ORG) or "",
            user_role=identity.get(IdentityKey.ROLE) or "",
            active_study=study,
            cert_studies=cert_studies,
        )
        if not session:
            _reject(audit_reason="inactive_admin_transport")
            return
        self.logger.info(f"Created user session for {user_name}")
        conn.append_string("OK")
        conn.append_token(token)
        record("admin.login.succeeded", study=study)

    def handle_logout(self, conn: Connection, args: List[str]):
        if self.session_mgr:
            session = conn.get_prop(ConnProps.SESSION)
            if session:
                self.session_mgr.end_session_by_id(session.sess_id)
        conn.append_string("OK")

    def pre_command(self, conn: Connection, args: List[str]):
        if args[0] in [InternalCommands.CERT_LOGIN, InternalCommands.CHECK_SESSION]:
            # skip login and check session commands
            return True

        # validate token
        token = conn.get_token()
        if token is None:
            reject("admin.session.rejected", user="unknown", reason="missing_token")
            conn.append_error("not authenticated - no token")
            return False

        sess = self.session_mgr.get_session(token, conn.get_prop(ConnProps.ADMIN_TRANSPORT_GRANT), touch=True)
        if not sess:
            reject("admin.session.rejected", user="unknown", reason="invalid_or_expired_token")
            conn.append_error(ReplyKeyword.SESSION_INACTIVE)
            return False

        assert isinstance(sess, Session)
        update_context(user=sess.user_name, identity_source="locally_verified")
        conn.set_prop(ConnProps.SESSION, sess)
        conn.set_prop(ConnProps.USER_NAME, sess.user_name)
        conn.set_prop(ConnProps.USER_ORG, sess.user_org)
        conn.set_prop(ConnProps.USER_ROLE, sess.user_role)
        conn.set_prop(ConnProps.ACTIVE_STUDY, sess.active_study)
        conn.set_prop(ConnProps.CERT_STUDIES, sess.cert_studies)
        conn.set_prop(ConnProps.TOKEN, token)
        return True

    def close(self):
        self.session_mgr.shutdown()
