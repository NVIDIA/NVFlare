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
import time
from typing import List

from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import InternalCommands
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandSpec
from nvflare.fuel.hci.security import make_session_token
from nvflare.fuel.utils.time_utils import time_to_string

LIST_SESSIONS_CMD_NAME = InternalCommands.LIST_SESSIONS
CHECK_SESSION_CMD_NAME = InternalCommands.CHECK_SESSION


class Session(object):
    def __init__(self):
        """Object keeping track of an admin client session with token and time data."""
        self.user_name = None
        self.user_org = None
        self.user_role = None
        self.start_time = None
        self.last_active_time = None
        self.token = None

    def mark_active(self):
        self.last_active_time = time.time()


class SessionManager(CommandModule):
    def __init__(self, idle_timeout=3600, monitor_interval=5):
        """Session manager.

        Args:
            idle_timeout: session idle timeout
            monitor_interval: interval for obtaining updates when monitoring
        """
        if monitor_interval <= 0:
            monitor_interval = 5

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
                self.end_session(dead_sess.token)
            else:
                # print('no dead sessions found')
                pass

            time.sleep(self.monitor_interval)

    def shutdown(self):
        self.asked_to_stop = True
        # self.monitor.join(timeout=10)

    def create_session(self, user_name, user_org, user_role):
        """Creates new session with a new session token.

        Args:
            user_name: user name for session
            user_org: org of the user
            user_role: user's role

        Returns: Session

        """
        token = make_session_token()
        sess = Session()
        sess.user_name = user_name
        sess.user_role = user_role
        sess.user_org = user_org
        sess.start_time = time.time()
        sess.last_active_time = sess.start_time
        sess.token = token
        with self.sess_update_lock:
            self.sessions[token] = sess
        return sess

    def get_session(self, token: str):
        with self.sess_update_lock:
            return self.sessions.get(token)

    def get_sessions(self):
        result = []
        with self.sess_update_lock:
            for _, s in self.sessions.items():
                result.append(s)
        return result

    def end_session(self, token):
        with self.sess_update_lock:
            self.sessions.pop(token, None)

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
                    enabled=False,
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
                    "{}".format(s.token),
                    time_to_string(s.start_time),
                    time_to_string(s.last_active_time),
                    "{}".format(time.time() - s.last_active_time),
                ]
            )

    def handle_check_session(self, conn: Connection, args: List[str]):
        token = None
        data = conn.request["data"]
        for item in data:
            it = item["type"]
            if it == "token":
                token = item["data"]
                break

        sess = self.get_session(token)
        if sess:
            conn.append_string("OK")
        else:
            conn.append_error("session_inactive")
            conn.append_string(
                "admin client session timed out after {} seconds of inactivity - logging out".format(self.idle_timeout)
            )
