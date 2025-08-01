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
import json
import threading
import time
import uuid
from typing import List

from nvflare.fuel.f3.cellnet.defs import CellChannel
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.fuel.hci.base64_utils import b64str_to_str, str_to_b64str
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import InternalCommands, ReplyKeyword
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandSpec
from nvflare.fuel.utils.time_utils import time_to_string
from nvflare.private.fed.utils.identity_utils import IdentityAsserter, TokenVerifier

LIST_SESSIONS_CMD_NAME = InternalCommands.LIST_SESSIONS
CHECK_SESSION_CMD_NAME = InternalCommands.CHECK_SESSION


class Session(object):
    def __init__(self, sess_id, user_name, org, role, origin_fqcn):
        """Object keeping track of an admin client session with token and time data."""
        self.sess_id = sess_id
        self.user_name = user_name
        self.user_org = org
        self.user_role = role
        self.origin_fqcn = origin_fqcn
        self.start_time = time.time()
        self.last_active_time = time.time()

    def mark_active(self):
        self.last_active_time = time.time()

    def make_token(self, id_asserter: IdentityAsserter):
        user = {
            "n": self.user_name,
            "r": self.user_role,
            "o": self.user_org,
            "s": self.sess_id,
        }
        ds = json.dumps(user)
        bds = str_to_b64str(ds)
        signature = id_asserter.sign(ds, return_str=True)

        # both bds and signature are b64 str
        return f"{bds}:{signature}"

    @staticmethod
    def decode_token(token: str, id_asserter: IdentityAsserter = None):
        if not isinstance(token, str):
            raise ValueError(f"token must be str but got {type(token)}")

        parts = token.split(":")
        if len(parts) != 2:
            raise ValueError(f"invalid token {token}: expects 2 parts but got {len(parts)}")

        bds = parts[0]
        signature = parts[1]
        ds = b64str_to_str(bds)
        if id_asserter:
            token_verifier = TokenVerifier(id_asserter.cert)
            is_valid = token_verifier.verify("", ds, signature)
            if not is_valid:
                return None

        user = json.loads(ds)
        return Session(
            user_name=user.get("n"),
            role=user.get("r"),
            org=user.get("o"),
            sess_id=user.get("s"),
            origin_fqcn="",
        )


class SessionManager(CommandModule):
    def __init__(self, cell, idle_timeout=1800, monitor_interval=5):
        """Session manager.

        Args:
            idle_timeout: session idle timeout
            monitor_interval: interval for obtaining updates when monitoring
        """
        if monitor_interval <= 0:
            monitor_interval = 5

        self.cell = cell
        self.sess_update_lock = threading.Lock()
        self.sessions = {}  # token => Session
        self.idle_timeout = idle_timeout
        self.monitor_interval = monitor_interval
        self.asked_to_stop = False
        self.monitor = threading.Thread(target=self.monitor_sessions)
        self.monitor.daemon = True
        self.monitor.start()

    def monitor_sessions(self):
        """Runs loop in a thread to end sessions that time out."""
        while True:
            # print('checking for dead sessions ...')
            if self.asked_to_stop:
                break

            dead_sess = None
            for _, sess in self.sessions.items():
                time_passed = time.time() - sess.last_active_time
                # print('time passed: {} secs'.format(time_passed))
                if time_passed > self.idle_timeout:
                    dead_sess = sess
                    break

            if dead_sess:
                # print('ending dead session {}'.format(dead_sess.token))
                self.end_session_by_id(dead_sess.sess_id, "Your session is closed due to inactivity.")
            else:
                # print('no dead sessions found')
                pass

            time.sleep(self.monitor_interval)

    def shutdown(self):
        self.asked_to_stop = True

    def create_session(self, user_name, user_org, user_role, origin_fqcn):
        """Creates new session with a new session token.

        Args:
            user_name: username for session
            user_org: org of the user
            user_role: user's role
            origin_fqcn: request origin FQCN
            id_asserter: used to sign session token

        Returns: Session

        """
        sess_id = str(uuid.uuid4())
        sess = Session(
            sess_id=sess_id,
            user_name=user_name,
            org=user_org,
            role=user_role,
            origin_fqcn=origin_fqcn,
        )
        with self.sess_update_lock:
            self.sessions[sess_id] = sess
        return sess

    def recreate_session(self, token: str, origin_fqcn, id_asserter: IdentityAsserter):
        sess = Session.decode_token(token, id_asserter)
        sess.origin_fqcn = origin_fqcn
        with self.sess_update_lock:
            self.sessions[sess.sess_id] = sess
        return sess

    def get_session(self, token: str):
        try:
            sess = Session.decode_token(token)
        except:
            return None

        with self.sess_update_lock:
            return self.sessions.get(sess.sess_id)

    def get_sessions(self):
        result = []
        with self.sess_update_lock:
            for _, s in self.sessions.items():
                result.append(s)
        return result

    def end_session_by_token(self, token, reason=None):
        try:
            sess = Session.decode_token(token)
        except:
            return
        self.end_session_by_id(sess.sess_id, reason)

    def end_session_by_id(self, sess_id: str, reason=None):
        with self.sess_update_lock:
            sess = self.sessions.pop(sess_id, None)
            if sess and reason:
                self.cell.fire_and_forget(
                    channel=CellChannel.HCI,
                    topic="SESSION_EXPIRED",
                    targets=sess.origin_fqcn,
                    message=CellMessage(payload=reason),
                    optional=True,
                )

    def get_spec(self):
        return CommandModuleSpec(
            name="sess",
            cmd_specs=[
                CommandSpec(
                    name=LIST_SESSIONS_CMD_NAME,
                    description="list user sessions",
                    usage=LIST_SESSIONS_CMD_NAME,
                    handler_func=self.handle_list_sessions,
                    visible=False,
                    enabled=True,
                ),
                CommandSpec(
                    name=CHECK_SESSION_CMD_NAME,
                    description="check if session is active",
                    usage=CHECK_SESSION_CMD_NAME,
                    handler_func=self.handle_check_session,
                    visible=False,
                ),
            ],
        )

    def handle_list_sessions(self, conn: Connection, args: List[str]):
        """Lists sessions and the details in a table.

        Registered in the FedAdminServer with ``cmd_reg.register_module(sess_mgr)``.
        """
        with self.sess_update_lock:
            sess_list = list(self.sessions.values())
        sess_list.sort(key=lambda x: x.user_name, reverse=False)
        table = conn.append_table(["User", "Org", "Role", "Session ID", "Start", "Last Active", "Idle"])
        for s in sess_list:
            table.add_row(
                [
                    s.user_name,
                    s.user_org,
                    s.user_role,
                    s.sess_id,
                    time_to_string(s.start_time),
                    time_to_string(s.last_active_time),
                    f"{(time.time() - s.last_active_time)}",
                ]
            )

    def handle_check_session(self, conn: Connection, args: List[str]):
        token = conn.get_token()
        if not token:
            conn.append_error("invalid_session")
            return

        sess = self.get_session(token)
        if sess:
            conn.append_string("OK")
        else:
            conn.append_error(ReplyKeyword.SESSION_INACTIVE)
            conn.append_string(
                "admin client session timed out after {} seconds of inactivity - logging out".format(self.idle_timeout)
            )
