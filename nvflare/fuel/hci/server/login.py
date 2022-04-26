# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

from abc import ABC, abstractmethod
from typing import List

from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandSpec
from nvflare.fuel.hci.security import verify_password
from nvflare.fuel.hci.server.constants import ConnProps

from .reg import CommandFilter
from .sess import CHECK_SESSION_CMD_NAME, SessionManager

LOGIN_CMD_NAME = "_login"
CERT_LOGIN_CMD_NAME = "_cert_login"


class Authenticator(ABC):
    """Base class for authenticating credentials."""

    @abstractmethod
    def authenticate(self, user_name: str, credential: str, credential_type: str) -> bool:
        """Authenticate a specified user with the provided credential.

        Args:
            user_name: user login name
            credential: provided credential
            credential_type: type of credential

        Returns: True if successful, False otherwise

        """
        pass


class SimpleAuthenticator(Authenticator):
    def __init__(self, users):
        """Authenticator to use in the LoginModule for authenticating admin clients for login.

        Args:
            users: user information
        """
        self.users = users

    def authenticate_password(self, user_name: str, pwd: str):
        pwd_hash = self.users.get(user_name)
        if pwd_hash is None:
            return False

        return verify_password(pwd_hash, pwd)

    def authenticate_cn(self, user_name: str, cn):
        return user_name == cn

    def authenticate(self, user_name: str, credential, credential_type):
        if credential_type == "password":
            return self.authenticate_password(user_name, credential)
        elif credential_type == "cn":
            return self.authenticate_cn(user_name, credential)
        else:
            return False


class LoginModule(CommandModule, CommandFilter):
    def __init__(self, authenticator: Authenticator, sess_mgr: SessionManager):
        """Login module.

        CommandModule containing the login commands to handle login and logout of admin clients, as well as the
        CommandFilter pre_command to check that a client is logged in with a valid session.

        Args:
            authenticator: Authenticator
            sess_mgr: SessionManager
        """
        if authenticator:
            if not isinstance(authenticator, Authenticator):
                raise TypeError("authenticator must be Authenticator but got {}.".format(type(authenticator)))

        if not isinstance(sess_mgr, SessionManager):
            raise TypeError("sess_mgr must be SessionManager but got {}.".format(type(sess_mgr)))

        self.authenticator = authenticator
        self.session_mgr = sess_mgr

    def get_spec(self):
        return CommandModuleSpec(
            name="login",
            cmd_specs=[
                CommandSpec(
                    name=LOGIN_CMD_NAME,
                    description="login to server",
                    usage="login userName password",
                    handler_func=self.handle_login,
                    visible=False,
                ),
                CommandSpec(
                    name=CERT_LOGIN_CMD_NAME,
                    description="login to server with SSL cert",
                    usage="login userName",
                    handler_func=self.handle_cert_login,
                    visible=False,
                ),
                CommandSpec(
                    name="_logout",
                    description="logout from server",
                    usage="logout",
                    handler_func=self.handle_logout,
                    visible=False,
                ),
            ],
        )

    def handle_login(self, conn: Connection, args: List[str]):
        if not self.authenticator:
            conn.append_string("OK")
            return

        if len(args) != 3:
            conn.append_string("REJECT")
            return

        user_name = args[1]
        pwd = args[2]

        ok = self.authenticator.authenticate(user_name, pwd, "password")
        if not ok:
            conn.append_string("REJECT")
            return

        session = self.session_mgr.create_session(user_name)
        conn.append_string("OK")
        conn.append_token(session.token)

    def handle_cert_login(self, conn: Connection, args: List[str]):
        if not self.authenticator:
            conn.append_string("OK")
            return

        if len(args) != 2:
            conn.append_string("REJECT")
            return

        cn = conn.get_prop("_client_cn", None)
        if cn is None:
            conn.append_string("REJECT")
            return

        user_name = args[1]

        ok = self.authenticator.authenticate(user_name, cn, "cn")
        if not ok:
            conn.append_string("REJECT")
            return

        session = self.session_mgr.create_session(user_name)
        conn.append_string("OK")
        conn.append_token(session.token)

    def handle_logout(self, conn: Connection, args: List[str]):
        if self.authenticator and self.session_mgr:
            token = conn.get_prop(ConnProps.TOKEN)
            if token:
                self.session_mgr.end_session(token)
        conn.append_string("OK")

    def pre_command(self, conn: Connection, args: List[str]):
        if args[0] in [LOGIN_CMD_NAME, CERT_LOGIN_CMD_NAME, CHECK_SESSION_CMD_NAME]:
            # skip login and check session commands
            return True

        # validate token
        req_json = conn.request
        token = None
        data = req_json["data"]
        for item in data:
            it = item["type"]
            if it == "token":
                token = item["data"]
                break

        if token is None:
            conn.append_error("not authenticated - no token")
            return False

        sess = self.session_mgr.get_session(token)
        if sess:
            sess.mark_active()
            conn.set_prop(ConnProps.SESSION, sess)
            conn.set_prop(ConnProps.USER_NAME, sess.user_name)
            conn.set_prop(ConnProps.TOKEN, token)
            return True
        else:
            conn.append_error("session_inactive")
            conn.append_string(
                "user not authenticated or session timed out after {} seconds of inactivity - logged out".format(
                    self.session_mgr.idle_timeout
                )
            )
            return False
