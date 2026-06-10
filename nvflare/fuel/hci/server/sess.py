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
from typing import List, Optional

from nvflare.apis.job_def import DEFAULT_STUDY
from nvflare.fuel.f3.cellnet.defs import CellChannel
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.fuel.hci.base64_utils import b64str_to_str, str_to_b64str
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import InternalCommands, ReplyKeyword
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandSpec
from nvflare.fuel.sec.principal import AUTH_METHOD_CERT, AUTH_METHOD_OIDC, Principal, to_optional_float
from nvflare.fuel.utils.time_utils import time_to_string
from nvflare.private.fed.utils.identity_utils import IdentityAsserter, TokenVerifier

LIST_SESSIONS_CMD_NAME = InternalCommands.LIST_SESSIONS
CHECK_SESSION_CMD_NAME = InternalCommands.CHECK_SESSION


DEFAULT_MAX_SESSION_LIFETIME = 8 * 60 * 60

# keep ended-session ids blocklisted slightly past token expiration to absorb clock skew
ENDED_SESSION_BLOCKLIST_MARGIN = 60.0


class Session(object):
    def __init__(
        self,
        sess_id,
        user_name=None,
        org=None,
        role=None,
        origin_fqcn="",
        active_study=DEFAULT_STUDY,
        principal=None,
        start_time: Optional[float] = None,
        expires_at: Optional[float] = None,
        session_epoch: str = "",
    ):
        """Object keeping track of an admin client session with token and time data.

        The principal is the single source of truth for the session identity. The legacy
        user_name/org/role args remain only for decode_token, which must rebuild a cert-admin
        principal from the legacy n/o/r token fields when the token carries no principal data.
        """
        self.sess_id = sess_id
        self.active_study = active_study
        self.origin_fqcn = origin_fqcn
        self.principal = principal or Principal.from_legacy_admin(
            username=user_name,
            org=org,
            role=role,
            auth_method=AUTH_METHOD_CERT,
        )
        self.start_time = float(start_time or time.time())
        self.last_active_time = time.time()
        self.expires_at = to_optional_float(expires_at)
        if not self.expires_at and self.principal.auth_method == AUTH_METHOD_OIDC:
            self.expires_at = to_optional_float(self.principal.token_exp)
        self.session_epoch = session_epoch or ""

    @property
    def user_name(self) -> str:
        return self.principal.policy_name()

    @property
    def user_org(self) -> str:
        return self.principal.policy_org()

    @property
    def user_role(self) -> str:
        return self.principal.policy_role()

    def mark_active(self):
        self.last_active_time = time.time()

    def is_expired(self, now: Optional[float] = None) -> bool:
        return bool(self.expires_at and (now or time.time()) >= self.expires_at)

    def make_token(self, id_asserter: IdentityAsserter):
        user = {
            "n": self.user_name,
            "r": self.user_role,
            "o": self.user_org,
            "s": self.sess_id,
            "study": self.active_study,
            "iat": int(self.start_time),
        }
        if self.expires_at:
            user["exp"] = int(self.expires_at)
        if self.session_epoch:
            user["epoch"] = self.session_epoch
        # Cert-admin principal data is fully derivable from legacy n/o/r fields.
        # Keep cert session tokens compatible and only add extended principal
        # metadata for non-cert auth methods such as OIDC.
        if self.principal and self.principal.auth_method != AUTH_METHOD_CERT:
            user["p"] = self.principal.to_dict()
        ds = json.dumps(user)
        bds = str_to_b64str(ds)
        signature = id_asserter.sign(ds, return_str=True)

        # both bds and signature are b64 str
        return f"{bds}:{signature}"

    @staticmethod
    def decode_token(token: str, id_asserter: IdentityAsserter = None, now: Optional[float] = None):
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
        principal_data = user.get("p")
        if principal_data is not None and not isinstance(principal_data, dict):
            raise ValueError(f"invalid principal data in token: expected dict but got {type(principal_data)}")
        principal = Principal.from_dict(principal_data) if principal_data else None
        expires_at = to_optional_float(user.get("exp"))
        if principal and principal.auth_method == AUTH_METHOD_OIDC and not expires_at:
            raise ValueError("OIDC session token is missing expiration")
        if expires_at and (now or time.time()) >= expires_at:
            raise ValueError("session token expired")

        return Session(
            user_name=user.get("n"),
            role=user.get("r"),
            org=user.get("o"),
            sess_id=user.get("s"),
            origin_fqcn="",
            active_study=user.get("study", user.get("t", DEFAULT_STUDY)),
            principal=principal,
            start_time=user.get("iat"),
            expires_at=expires_at,
            session_epoch=user.get("epoch", ""),
        )


class SessionManager(CommandModule):
    def __init__(self, cell, idle_timeout=1800, monitor_interval=5, max_session_lifetime=DEFAULT_MAX_SESSION_LIFETIME):
        """Session manager.

        Args:
            idle_timeout: session idle timeout
            monitor_interval: interval for obtaining updates when monitoring
            max_session_lifetime: absolute session token lifetime cap in seconds
        """
        if monitor_interval <= 0:
            monitor_interval = 5

        self.cell = cell
        self.sess_update_lock = threading.Lock()
        self.sessions = {}  # token => Session
        self.ended_sessions = {}  # sess_id => blocklist expiry (token exp + margin)
        self.idle_timeout = idle_timeout
        self.max_session_lifetime = max_session_lifetime
        self.session_epoch = str(uuid.uuid4())
        self.monitor_interval = monitor_interval
        self.asked_to_stop = False
        self.monitor = threading.Thread(target=self.monitor_sessions)
        self.monitor.daemon = True
        self.monitor.start()

    def monitor_sessions(self):
        """Runs loop in a thread to end sessions that time out."""
        while True:
            if self.asked_to_stop:
                break

            now = time.time()
            with self.sess_update_lock:
                # prune blocklist entries whose tokens can no longer validate anyway
                for sess_id in [i for i, expiry in self.ended_sessions.items() if expiry <= now]:
                    del self.ended_sessions[sess_id]
                sessions = list(self.sessions.values())

            for sess in sessions:
                if sess.is_expired(now):
                    self.end_session_by_id(
                        sess.sess_id, "Your session has reached its maximum lifetime - session token expired."
                    )
                elif now - sess.last_active_time > self.idle_timeout:
                    self.end_session_by_id(sess.sess_id, "Your session is closed due to inactivity.")

            time.sleep(self.monitor_interval)

    def shutdown(self):
        self.asked_to_stop = True

    def create_session_from_principal(self, principal: Principal, origin_fqcn, active_study=DEFAULT_STUDY):
        if not isinstance(principal, Principal):
            raise TypeError(f"principal must be Principal but got {type(principal)}")
        sess_id = str(uuid.uuid4())
        sess = Session(
            sess_id=sess_id,
            origin_fqcn=origin_fqcn,
            active_study=active_study,
            principal=principal,
            expires_at=self._session_expires_at(principal),
            session_epoch=self.session_epoch,
        )
        with self.sess_update_lock:
            self.sessions[sess_id] = sess
        return sess

    def recreate_session(self, token: str, origin_fqcn, id_asserter: IdentityAsserter):
        sess = Session.decode_token(token, id_asserter)
        self._validate_session_epoch(sess)
        sess.origin_fqcn = origin_fqcn
        with self.sess_update_lock:
            if sess.sess_id in self.ended_sessions:
                raise ValueError("session has ended")
            self.sessions[sess.sess_id] = sess
        return sess

    def get_session(self, token: str, id_asserter=None):
        try:
            sess = Session.decode_token(token, id_asserter)
            if sess is None:
                return None
            self._validate_session_epoch(sess)
        except Exception:
            return None

        with self.sess_update_lock:
            if sess.sess_id in self.ended_sessions:
                return None
            active_sess = self.sessions.get(sess.sess_id)
            if active_sess and active_sess.is_expired():
                self.ended_sessions[sess.sess_id] = self._blocklist_expiry(active_sess.expires_at)
                self.sessions.pop(sess.sess_id, None)
                return None
            return active_sess

    def _blocklist_expiry(self, expires_at: Optional[float]) -> float:
        if not expires_at:
            if self.max_session_lifetime and self.max_session_lifetime > 0:
                expires_at = time.time() + self.max_session_lifetime
            else:
                return float("inf")
        return expires_at + ENDED_SESSION_BLOCKLIST_MARGIN

    def _session_expires_at(self, principal: Optional[Principal]) -> Optional[float]:
        now = time.time()
        expires_at = (
            now + self.max_session_lifetime if self.max_session_lifetime and self.max_session_lifetime > 0 else None
        )
        if principal and principal.auth_method == AUTH_METHOD_OIDC:
            token_exp = to_optional_float(principal.token_exp)
            if token_exp:
                expires_at = min(expires_at, token_exp) if expires_at else token_exp
        return expires_at

    def _validate_session_epoch(self, sess: Session):
        if sess.session_epoch != self.session_epoch:
            raise ValueError("session token was issued by a previous server session")

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
        self.end_session_by_id(sess.sess_id, reason, expires_at=sess.expires_at)

    def end_session_by_id(self, sess_id: str, reason=None, expires_at: Optional[float] = None):
        with self.sess_update_lock:
            sess = self.sessions.pop(sess_id, None)
            self.ended_sessions[sess_id] = self._blocklist_expiry(sess.expires_at if sess else expires_at)
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
