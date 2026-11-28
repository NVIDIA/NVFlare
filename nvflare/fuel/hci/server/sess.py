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
import secrets
import threading
import time
import uuid
from dataclasses import dataclass
from hashlib import sha256
from typing import Callable, List

from nvflare.apis.job_def import DEFAULT_STUDY
from nvflare.fuel.f3.cellnet.defs import CellChannel, MessageHeaderKey, MessagePropKey
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import InternalCommands, ReplyKeyword
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandSpec
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.sec.admin_audit import record, reject, snapshot
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.fuel.utils.time_utils import time_to_string
from nvflare.private.defs import CellMessageHeaderKeys
from nvflare.security.logging import secure_format_exception
from nvflare.security.study_registry import StudyRegistryService

LIST_SESSIONS_CMD_NAME = InternalCommands.LIST_SESSIONS
CHECK_SESSION_CMD_NAME = InternalCommands.CHECK_SESSION


_EXPIRED_MESSAGE = "Your session is closed due to inactivity or cert expiry."


def _token_digest(token):
    if isinstance(token, str) and 0 < len(token) <= 128 and token.isascii():
        return sha256(token.encode("ascii")).digest()
    return None


def _transport_peer(request):
    origin = request.get_header(MessageHeaderKey.ORIGIN)
    peer = request.get_prop(MessagePropKey.ENDPOINT)
    if not origin or origin != getattr(peer, "name", None):
        return None
    return request.get_prop(MessagePropKey.PEER_CERT_FINGERPRINT) or request.get_prop(MessagePropKey.CONNECTION)


@dataclass(eq=False)
class AdminTransportGrant:
    """Server-held registration for an admin cell that proved certificate possession."""

    token_digest: bytes
    user_name: str
    origin: str
    cert_fingerprint: bytes
    cert_exp: float
    peer: object
    connection: object
    created: float
    session_id: str = ""


class Session:
    def __init__(self, grant, org, role, active_study, cert_studies, token_digest):
        """Server-side identity and activity for one successful admin login."""
        self.sess_id = str(uuid.uuid4())
        self.user_name = grant.user_name
        self.user_org = org
        self.user_role = role
        self.active_study = active_study
        self.cert_exp = grant.cert_exp
        self.cert_studies = tuple(cert_studies)
        self.origin_fqcn = grant.origin
        self.start_time = time.time()
        self.last_active_time = self.start_time
        self.audit_reference = ""
        self.grant = grant
        self.registry_membership = False
        self.token_digest = token_digest

    def mark_active(self):
        self.last_active_time = time.time()

    def is_cert_expired(self, now=None):
        if not self.cert_exp:
            return False
        now = time.time() if now is None else now
        return now >= self.cert_exp

    def study_allowed(self, registry):
        if self.active_study == DEFAULT_STUDY:
            return True
        if not registry or not registry.has_study(self.active_study):
            return False
        mapped = registry.has_user(self.user_name, self.active_study)
        return mapped or (not self.registry_membership and self.active_study in self.cert_studies)


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
        self.logger = get_obj_logger(self)
        self.sess_update_lock = threading.Lock()
        self.sessions = {}  # session ID => Session
        self._session_tokens = {}  # token digest => Session
        self._transport_grants = {}  # registration token digest => AdminTransportGrant
        self._origin_grants = {}
        self.downloads = {}  # session ID => {transaction ID: cancel callback}
        self.uploads = {}
        self.idle_timeout = idle_timeout
        self.monitor_interval = monitor_interval
        self._stop_event = threading.Event()
        self._monitor_wake = threading.Event()
        self._study_revocations = []
        StudyRegistryService.add_listener(self._study_registry_changed)
        self.monitor = threading.Thread(target=self.monitor_sessions, daemon=True)
        self.monitor.start()

    def _expiry_reason(self, sess, now):
        if sess.is_cert_expired(now):
            return "certificate_expired"
        if now - sess.last_active_time >= self.idle_timeout:
            return "inactivity"
        if not sess.study_allowed(StudyRegistryService.get_registry()):
            return "study_membership_revoked"
        return None

    def _study_registry_changed(self, registry):
        # Invalidate both credentials before the study mutation returns. The
        # monitor cancels downloads without delaying that mutation's reply.
        with self.sess_update_lock:
            for sess in list(self.sessions.values()):
                if not sess.study_allowed(registry):
                    self._study_revocations.append(self._detach_session(sess.sess_id))
                elif sess.active_study != DEFAULT_STUDY and registry.has_user(sess.user_name, sess.active_study):
                    sess.registry_membership = True
            if self._study_revocations:
                self._monitor_wake.set()

    def _finish_study_revocations(self, revoked):
        for sess, downloads in revoked:
            self._finish_session(
                sess, downloads, "Your study access changed. Please log in again.", "study_membership_revoked"
            )

    def check_sessions(self):
        """Revoke expired sessions and registrations that did not complete login."""
        now = time.time()
        with self.sess_update_lock:
            revoked, self._study_revocations = self._study_revocations, []
            expired = []
            for sess in list(self.sessions.values()):
                reason = self._expiry_reason(sess, now)
                if reason:
                    expired.append((*self._detach_session(sess.sess_id), reason))
            for grant in list(self._transport_grants.values()):
                if not grant.session_id and self._pending_grant_expired(grant, now):
                    self._detach_grant(grant)
        self._finish_study_revocations(revoked)
        for sess, downloads, reason in expired:
            self._finish_session(sess, downloads, _EXPIRED_MESSAGE, reason)

    def monitor_sessions(self):
        self._monitor_wake.clear()
        while not self._stop_event.is_set():
            self.check_sessions()
            self._monitor_wake.wait(self.monitor_interval)
            self._monitor_wake.clear()

    def shutdown(self):
        self._stop_event.set()
        self._monitor_wake.set()
        StudyRegistryService.remove_listener(self._study_registry_changed)
        with self.sess_update_lock:
            revoked, self._study_revocations = self._study_revocations, []
            ended = [self._detach_session(sess_id) for sess_id in list(self.sessions)]
            self._transport_grants.clear()
            self._origin_grants.clear()
        self._finish_study_revocations(revoked)
        for sess, downloads in ended:
            self._finish_session(sess, downloads, None, "server_shutdown")

    def _pending_grant_expired(self, grant, now):
        return now >= grant.cert_exp or now - grant.created >= self.idle_timeout

    def register_transport(self, request, user_name, token, cert_fingerprint, cert_exp):
        """Bind a verified registration to its origin and actual transport peer."""
        peer = _transport_peer(request)
        peer_fingerprint = request.get_prop(MessagePropKey.PEER_CERT_FINGERPRINT)
        digest = _token_digest(token)
        now = time.time()
        if not peer or not digest or now >= cert_exp or (peer_fingerprint and peer_fingerprint != cert_fingerprint):
            return False
        origin = request.get_header(MessageHeaderKey.ORIGIN)
        connection = request.get_prop(MessagePropKey.CONNECTION)
        grant = AdminTransportGrant(digest, user_name, origin, cert_fingerprint, cert_exp, peer, connection, now)
        previous_session = None
        with self.sess_update_lock:
            if self._stop_event.is_set():
                return False
            previous = self._origin_grants.get(origin)
            if previous:
                if previous.user_name != user_name:
                    return False
                if previous.session_id:
                    previous_session = self._detach_session(previous.session_id)
                else:
                    self._detach_grant(previous)
            self._transport_grants[digest] = grant
            self._origin_grants[origin] = grant
        if previous_session:
            self._finish_session(*previous_session, None, "transport_replaced")
        return True

    def close_transport(self, request):
        """Revoke a grant on a direct CellNet goodbye from its original connection."""
        peer = _transport_peer(request)
        connection = request.get_prop(MessagePropKey.CONNECTION)
        if not peer or not connection:
            return
        origin = request.get_header(MessageHeaderKey.ORIGIN)
        with self.sess_update_lock:
            grant = self._origin_grants.get(origin)
            if not grant or grant.peer != peer or grant.connection is not connection:
                return
            if not grant.session_id:
                self._detach_grant(grant)
                return
            sess, downloads = self._detach_session(grant.session_id)
        self._finish_session(sess, downloads, None, "transport_closed")

    def get_transport(self, request, *, require_session=False):
        peer = _transport_peer(request)
        if not peer:
            return None
        digest = _token_digest(request.get_header(CellMessageHeaderKeys.TOKEN))
        origin = request.get_header(MessageHeaderKey.ORIGIN)
        user = request.get_header(CellMessageHeaderKeys.CLIENT_NAME)
        with self.sess_update_lock:
            grant = self._transport_grants.get(digest)
            if not grant or grant.origin != origin or grant.user_name != user or grant.peer != peer:
                return None
            now = time.time()
            if not grant.session_id:
                if self._pending_grant_expired(grant, now):
                    self._detach_grant(grant)
                    return None
                return None if require_session else grant
            sess = self.sessions[grant.session_id]
            reason = self._expiry_reason(sess, now)
            if not reason:
                grant.connection = request.get_prop(MessagePropKey.CONNECTION)
                return grant
            sess, downloads = self._detach_session(sess.sess_id)
        self._finish_session(sess, downloads, _EXPIRED_MESSAGE, reason)
        return None

    def create_session(
        self, grant: AdminTransportGrant, user_org: str, user_role: str, active_study=DEFAULT_STUDY, cert_studies=()
    ):
        """Record a certificate login and return its session and opaque token."""
        token = secrets.token_urlsafe(32)
        sess = Session(grant, user_org, user_role, active_study, cert_studies, _token_digest(token))
        replaced = None
        with self.sess_update_lock:
            if self._stop_event.is_set() or self._transport_grants.get(grant.token_digest) is not grant:
                return None, None
            now = time.time()
            current = self.sessions.get(grant.session_id)
            if current:
                if self._expiry_reason(current, now):
                    return None, None
            elif self._pending_grant_expired(grant, now):
                return None, None

            registry = StudyRegistryService.get_registry()
            if not sess.study_allowed(registry):
                return None, None
            if current:
                # A verified certificate login can be retried after a lost reply.
                replaced = self._detach_session(current.sess_id, revoke_transport=False)
            sess.registry_membership = active_study != DEFAULT_STUDY and registry.has_user(
                grant.user_name, active_study
            )
            grant.session_id = sess.sess_id
            self.sessions[sess.sess_id] = sess
            self._session_tokens[sess.token_digest] = sess
        if replaced:
            self._finish_session(*replaced, None, "session_replaced")
        sess.audit_reference = record("admin.session.created", user=grant.user_name, study=active_study)
        return sess, token

    def get_session(self, token: str, grant: AdminTransportGrant, *, touch=False):
        digest = _token_digest(token)
        with self.sess_update_lock:
            sess = self._session_tokens.get(digest)
            if not grant or not sess or sess.grant is not grant:
                return None
            reason = self._expiry_reason(sess, time.time())
            if not reason:
                if touch:
                    sess.mark_active()
                return sess
            sess, downloads = self._detach_session(sess.sess_id)
        self._finish_session(sess, downloads, _EXPIRED_MESSAGE, reason)
        return None

    def get_transport_session(self, grant: AdminTransportGrant):
        with self.sess_update_lock:
            sess = self.sessions.get(grant.session_id)
            if sess and sess.grant is grant and not self._expiry_reason(sess, time.time()):
                return sess
        return None

    def bind_upload(self, sess: Session, tx_id: str, transfer, cancel_cb: Callable[[], None]) -> bool:
        """Register cleanup for this upload, even if its transaction ID is reused."""
        with self.sess_update_lock:
            if self.sessions.get(sess.sess_id) is not sess or self._expiry_reason(sess, time.time()):
                return False
            self.uploads.setdefault(sess.sess_id, {})[(tx_id, transfer)] = cancel_cb
        return True

    def end_upload(self, sess: Session, tx_id: str, transfer):
        with self.sess_update_lock:
            uploads = self.uploads.get(sess.sess_id)
            if uploads:
                uploads.pop((tx_id, transfer), None)
                if not uploads:
                    self.uploads.pop(sess.sess_id, None)

    def bind_download(self, sess_id: str, tx_id: str, cancel_cb: Callable[[], None]) -> bool:
        """Bind an authorized download transaction to its current session."""
        with self.sess_update_lock:
            sess = self.sessions.get(sess_id)
            if not sess or self._expiry_reason(sess, time.time()):
                return False
            self.downloads.setdefault(sess_id, {})[tx_id] = cancel_cb
        return True

    def mark_download_active(self, sess_id: str, tx_id: str) -> bool:
        """Refresh a session only for verified progress from one of its bound downloads."""
        with self.sess_update_lock:
            sess = self.sessions.get(sess_id)
            session_downloads = self.downloads.get(sess_id)
            if (
                not sess
                or not session_downloads
                or tx_id not in session_downloads
                or self._expiry_reason(sess, time.time())
            ):
                return False
            sess.mark_active()
        return True

    def end_download(self, sess_id: str, tx_id: str):
        with self.sess_update_lock:
            session_downloads = self.downloads.get(sess_id)
            if not session_downloads:
                return
            session_downloads.pop(tx_id, None)
            if not session_downloads:
                self.downloads.pop(sess_id, None)

    def get_sessions(self):
        result = []
        with self.sess_update_lock:
            for _, s in self.sessions.items():
                result.append(s)
        return result

    def _detach_grant(self, grant):
        self._transport_grants.pop(grant.token_digest, None)
        self._origin_grants.pop(grant.origin, None)

    def _detach_session(self, sess_id, *, revoke_transport=True):
        sess = self.sessions.pop(sess_id, None)
        if sess:
            self._session_tokens.pop(sess.token_digest, None)
            if revoke_transport:
                self._detach_grant(sess.grant)
        downloads = self.downloads.pop(sess_id, {})
        uploads = self.uploads.pop(sess_id, {})
        transfers = list(downloads.items())
        transfers.extend((tx_id, cancel) for (tx_id, _), cancel in uploads.items())
        return sess, transfers

    def end_session_by_id(self, sess_id: str, reason=None, *, audit_reason="logout"):
        with self.sess_update_lock:
            sess, downloads = self._detach_session(sess_id)
        self._finish_session(sess, downloads, reason, audit_reason)

    def _finish_session(self, sess, transfers, reason, audit_reason):
        if sess:
            record(
                "admin.session.ended",
                user=sess.user_name,
                ref=snapshot().request_id or sess.audit_reference,
                reason=audit_reason,
                study=sess.active_study,
            )
        if sess and reason:
            try:
                notification = CellMessage(payload=reason)
                notification.set_prop(MessagePropKey.TARGET_AUTH_PEER, sess.grant.peer)
                self.cell.fire_and_forget(
                    channel=CellChannel.HCI,
                    topic="SESSION_EXPIRED",
                    targets=sess.origin_fqcn,
                    message=notification,
                    optional=True,
                )
            except Exception as ex:
                self.logger.error(f"failed to notify expired session {sess.sess_id}: {secure_format_exception(ex)}")
        for tx_id, cancel_cb in transfers:
            try:
                cancel_cb()
            except Exception as ex:
                self.logger.error(
                    f"failed to cancel transfer {tx_id} for session {getattr(sess, 'sess_id', None)}: {secure_format_exception(ex)}"
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
            reject("admin.session.rejected", user="unknown", reason="missing_token")
            conn.append_error("invalid_session")
            return

        sess = self.get_session(token, conn.get_prop(ConnProps.ADMIN_TRANSPORT_GRANT))
        if sess:
            conn.append_string("OK")
        else:
            reject("admin.session.rejected", user="unknown", reason="invalid_or_expired_token")
            conn.append_error(ReplyKeyword.SESSION_INACTIVE)
            conn.append_string(
                "admin client session timed out after {} seconds of inactivity - logging out".format(self.idle_timeout)
            )
