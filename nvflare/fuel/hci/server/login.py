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

from nvflare.apis.job_def import DEFAULT_STUDY
from nvflare.apis.utils.format_check import name_check
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import InternalCommands, ReplyKeyword
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandSpec
from nvflare.fuel.hci.security import IdentityKey, get_identity_info
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.lighter.utils import cert_to_dict, load_crt_bytes
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
            raise TypeError(f"sess_mgr must be SessionManager but got {type(sess_mgr)}.")

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

    def handle_cert_login(self, conn: Connection, args: list[str]):
        def _reject(reason: str = "", code: str = ""):
            if code:
                conn.append_string(f"REJECT: {code}: {reason}" if reason else f"REJECT: {code}")
            else:
                conn.append_string(f"REJECT: {reason}" if reason else "REJECT")

        if len(args) != 2:
            _reject()
            return

        user_name = args[1]
        headers = conn.get_prop(ConnProps.CMD_HEADERS)
        cert_data = headers.get("cert")
        signature = headers.get("signature")
        study = headers.get("study", DEFAULT_STUDY)

        self.logger.debug(f"got cert login headers: {headers=}")
        hci = conn.get_prop(ConnProps.HCI_SERVER)
        identity_verifier = hci.get_id_verifier()
        id_asserter = hci.get_id_asserter()

        cert = load_crt_bytes(cert_data)
        try:
            ok = identity_verifier.verify_common_name(
                asserter_cert=cert,
                asserted_cn=user_name,
                signature=signature,
                nonce="",
            )
            self.logger.debug(f"verify common name: {ok=}")
        except Exception as ex:
            self.logger.error(f"identity_verifier.verify_common_name got exception: {ex}")
            traceback.print_exc()
            ok = False

        if not ok:
            _reject()
            return

        if not isinstance(study, str):
            _reject("study must be a string", code="AUTH_INVALID_STUDY")
            return

        invalid, _ = name_check(study, "study")
        if invalid:
            _reject(f"invalid study name '{study}'", code="AUTH_INVALID_STUDY_NAME")
            return

        registry = StudyRegistryService.get_registry()
        if study != DEFAULT_STUDY:
            if not registry:
                self.logger.warning(f"rejecting login for user '{user_name}': no study registry for study '{study}'")
                _reject(f"study '{study}' is not configured on the server", code="AUTH_STUDY_NOT_CONFIGURED")
                return
            if not registry.has_study(study):
                self.logger.warning(f"rejecting login for user '{user_name}': unknown study '{study}'")
                _reject(f"unknown study '{study}'", code="AUTH_UNKNOWN_STUDY")
                return
            if not registry.has_user(user_name, study):
                self.logger.warning(f"rejecting login for user '{user_name}': no mapping for study '{study}'")
                _reject(
                    f"user '{user_name}' is not mapped to study '{study}'",
                    code="AUTH_STUDY_USER_NOT_MAPPED",
                )
                return

        cert_dict = cert_to_dict(cert)
        self.logger.debug(f"got cert dict: {cert_dict}")
        identity = get_identity_info(cert_dict)

        request = conn.get_prop(ConnProps.REQUEST)
        assert isinstance(request, CellMessage)
        origin = request.get_header(MessageHeaderKey.ORIGIN)

        session = self.session_mgr.create_session(
            user_name=user_name,
            user_org=identity.get(IdentityKey.ORG) or "",
            user_role=identity.get(IdentityKey.ROLE) or "",
            origin_fqcn=origin,
            active_study=study,
        )
        token = session.make_token(id_asserter)
        self.logger.info(f"Created user session for {user_name}")
        conn.append_string("OK")
        conn.append_token(token)

    def handle_logout(self, conn: Connection, args: list[str]):
        if self.session_mgr:
            token = conn.get_prop(ConnProps.TOKEN)
            if token:
                self.session_mgr.end_session_by_token(token)
        conn.append_string("OK")

    def pre_command(self, conn: Connection, args: list[str]):
        if args[0] in [InternalCommands.CERT_LOGIN, InternalCommands.CHECK_SESSION]:
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
        conn.set_prop(ConnProps.ACTIVE_STUDY, sess.active_study)
        conn.set_prop(ConnProps.TOKEN, token)
        return True

    def close(self):
        self.session_mgr.shutdown()
