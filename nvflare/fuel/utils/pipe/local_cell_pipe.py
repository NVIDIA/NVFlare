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

"""LocalCellPipe: A Pipe implementation using a direct local Cellnet connection.

Unlike CellPipe, which routes messages through the external FL network (server / CP),
LocalCellPipe establishes a direct TCP connection on localhost between the CJ (Client
Job) process and the trainer subprocess.  No FL-network credentials, TLS certificates,
or round-trips through a relay are required for the CJ ↔ subprocess channel.

Architecture
------------
PASSIVE side (CJ process)
  Creates a Cellnet Cell whose FQCN is ``server.<site_name>_<token>`` and which binds
  a *internal* listener on ``tcp://127.0.0.1:<random-port>``.  The bound port is
  dynamically assigned by the OS at ``Cell.start()`` time and retrieved via
  ``CoreCell.get_internal_listener_url()``.

ACTIVE side (trainer subprocess)
  Creates a Cellnet Cell whose FQCN is ``server.<site_name>_<token>.client`` and which
  connects to the PASSIVE cell via ``parent_url=<internal-listener-url>``.  The URL is
  written to the client-api config file by the PASSIVE side's ``export()`` call (which
  happens *after* ``open()`` has already started the Cell and bound the port).

Shared Cell
  Both the task pipe and the metric pipe share a **single** Cell on each side via a
  class-level ``_cells_info`` cache, keyed by ``(mode, site_name, token)``.  Different
  channels (``task``, ``metric``) are multiplexed over the same Cell using distinct
  Cellnet *channel* strings.  This exactly mirrors the sharing pattern of ``CellPipe``.

Timing guarantee
  ``Pipe.open()`` is always called before ``Pipe.export()`` (``START_RUN`` /
  ``ABOUT_TO_START_RUN`` event ordering in the FL engine), so the dynamically bound
  port is always known by the time ``export()`` is invoked.
"""

import queue
import threading
import time
from typing import Tuple, Union

from nvflare.apis.fl_constant import SystemVarName
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.cell import Message as CellMessage
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply
from nvflare.fuel.utils.attributes_exportable import ExportMode
from nvflare.fuel.utils.constants import Mode
from nvflare.fuel.utils.log_utils import get_obj_logger

from .pipe import Message, Pipe, Topic

# ──────────────────────────────────────────────────────────────────────────────
# Message header keys (private to LocalCellPipe)
# ──────────────────────────────────────────────────────────────────────────────
_PREFIX = "local_cell_pipe."
_HEADER_MSG_TYPE = _PREFIX + "msg_type"
_HEADER_MSG_ID = _PREFIX + "msg_id"
_HEADER_REQ_ID = _PREFIX + "req_id"
_HEADER_START_TIME = _PREFIX + "start"
_HEADER_HB_SEQ = _PREFIX + "hb_seq"

# Dummy root_url used to satisfy CoreCell's requirement for at least one of
# root_url / parent_url.  For a server-child cell this value is stored but
# never used to create any listener or external connector.
_DUMMY_ROOT_URL = "tcp://127.0.0.1:0"


def _to_cell_message(msg: Message, extra: dict = None) -> CellMessage:
    headers = {
        _HEADER_MSG_TYPE: msg.msg_type,
        _HEADER_MSG_ID: msg.msg_id,
        _HEADER_START_TIME: time.time(),
    }
    if extra:
        headers.update(extra)
    if msg.req_id:
        headers[_HEADER_REQ_ID] = msg.req_id
    return CellMessage(headers=headers, payload=msg.data)


def _from_cell_message(cm: CellMessage) -> Message:
    return Message(
        msg_id=cm.get_header(_HEADER_MSG_ID),
        msg_type=cm.get_header(_HEADER_MSG_TYPE),
        topic=cm.get_header(MessageHeaderKey.TOPIC),
        req_id=cm.get_header(_HEADER_REQ_ID),
        data=cm.payload,
    )


def _make_passive_fqcn(site_name: str, token: str) -> str:
    """FQCN for the PASSIVE (CJ-side) Cell.

    The ``server.`` prefix makes CoreCell treat this as a server-family cell.
    Being non-root (gen=2) it calls ``_set_bb_for_server_child()``, which
    creates only the internal listener (no external connectors).
    """
    return f"server.{site_name}_{token}"


def _make_active_fqcn(site_name: str, token: str) -> str:
    """FQCN for the ACTIVE (subprocess-side) Cell.

    Child of the PASSIVE cell in the FQCN tree; uses internal connector to
    reach the PASSIVE cell's internal listener.
    """
    return f"server.{site_name}_{token}.client"


# ──────────────────────────────────────────────────────────────────────────────
# Internal cell-info / lifecycle helper
# ──────────────────────────────────────────────────────────────────────────────
class _LocalCellInfo:
    """Holds the shared Cell and reference-counts the pipes that use it."""

    def __init__(self, cell: Cell):
        self.cell = cell
        self.started = False
        self.pipes = []
        self.lock = threading.Lock()

    def start(self):
        with self.lock:
            if not self.started:
                self.cell.start()
                self.started = True

    def add_pipe(self, p: "LocalCellPipe"):
        with self.lock:
            self.pipes.append(p)

    def close_pipe(self, p: "LocalCellPipe"):
        with self.lock:
            try:
                self.pipes.remove(p)
                if not self.pipes:
                    self.cell.stop()
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────────────────────
# LocalCellPipe
# ──────────────────────────────────────────────────────────────────────────────
class LocalCellPipe(Pipe):
    """Pipe that uses a direct local Cellnet connection between the CJ process
    (PASSIVE) and the trainer subprocess (ACTIVE).

    Args:
        mode (Mode): ``Mode.PASSIVE`` for the CJ side, ``Mode.ACTIVE`` for the
            subprocess side.
        site_name (str): FLARE site name.  Used to build a unique FQCN.
        token (str): Job ID.  Used to build a unique FQCN.
        parent_url (str): **ACTIVE side only** – the internal-listener URL
            exported by the PASSIVE side.  Leave empty (``""``) for the PASSIVE
            side or during job-config serialisation (placeholder args).
    """

    _lock = threading.Lock()
    # (mode_str, site_name, token) -> _LocalCellInfo
    _cells_info: dict = {}

    def __init__(
        self,
        mode: Mode,
        site_name: str,
        token: str,
        parent_url: str = "",
    ):
        super().__init__(mode)
        self.logger = get_obj_logger(self)
        self.site_name = site_name
        self.token = token
        self._parent_url = parent_url

        # ── placeholder guard (job-config serialisation) ──────────────────────
        # The job-config system passes SystemVarName placeholders (e.g.
        # "{SITE_NAME}") when building the component graph.  Detect them and
        # skip the real initialisation; the object will be re-instantiated at
        # runtime with concrete values.
        sysvar_placeholders = ["{" + varname + "}" for varname in dir(SystemVarName)]
        if any(arg in sysvar_placeholders for arg in [site_name, token, parent_url]):
            self._ci = None
            self._peer_fqcn = None
            self._channel = None
            self._received_msgs = None
            self._pipe_lock = None
            self._closed = False
            return

        mode_str = f"{mode}".strip().lower()
        if mode_str not in ("passive", "active"):
            raise ValueError(f"invalid mode '{mode}' – must be 'passive' or 'active'")

        passive_fqcn = _make_passive_fqcn(site_name, token)
        active_fqcn = _make_active_fqcn(site_name, token)

        if mode_str == "passive":
            self._peer_fqcn = active_fqcn
            ci = self._get_or_create_cell("passive", site_name, token, passive_fqcn, parent_url=None)
        else:
            if not parent_url:
                raise ValueError("ACTIVE LocalCellPipe requires 'parent_url' (the PASSIVE cell's listener URL)")
            self._peer_fqcn = passive_fqcn
            ci = self._get_or_create_cell("active", site_name, token, active_fqcn, parent_url=parent_url)

        self._ci = ci
        self._ci.add_pipe(self)
        self._channel: str = ""
        self._received_msgs: queue.Queue = queue.Queue()
        self._pipe_lock = threading.Lock()
        self._closed = False
        self._hb_seq = 1

    # ── Cell creation / caching ───────────────────────────────────────────────

    @classmethod
    def _get_or_create_cell(
        cls,
        mode_str: str,
        site_name: str,
        token: str,
        fqcn: str,
        parent_url,
    ) -> _LocalCellInfo:
        cache_key = (mode_str, site_name, token)
        with cls._lock:
            ci = cls._cells_info.get(cache_key)
            if ci is None:
                cell = cls._build_cell(mode_str, fqcn, parent_url)
                ci = _LocalCellInfo(cell)
                cls._cells_info[cache_key] = ci
            return ci

    @staticmethod
    def _build_cell(mode_str: str, fqcn: str, parent_url) -> Cell:
        """Construct the underlying Cellnet Cell.

        PASSIVE side
            FQCN  ``server.<site>_<token>``  (server family, gen=2, non-root)
            CoreCell calls ``_set_bb_for_server_child(None, True)`` which
            creates the internal listener and nothing else.
            ``root_url`` is set to the dummy value to satisfy CoreCell's
            requirement that at least one of root_url / parent_url be provided.

        ACTIVE side
            FQCN  ``server.<site>_<token>.client``  (server family, gen=3)
            CoreCell calls ``_set_bb_for_server_child(parent_url, False)``
            which creates the internal connector to the PASSIVE listener.
        """
        if mode_str == "passive":
            return Cell(
                fqcn=fqcn,
                root_url=_DUMMY_ROOT_URL,  # stored but not used for server-child
                secure=False,
                credentials={},
                create_internal_listener=True,
                parent_url=None,
            )
        else:  # active
            return Cell(
                fqcn=fqcn,
                root_url=None,
                secure=False,
                credentials={},
                create_internal_listener=False,
                parent_url=parent_url,
            )

    # ── Pipe interface ────────────────────────────────────────────────────────

    def open(self, name: str):
        if self._ci is None:
            raise RuntimeError("LocalCellPipe was not fully initialised (placeholder args?)")
        with self._pipe_lock:
            if self._closed:
                raise BrokenPipeError("pipe already closed")
            self._ci.start()
            channel = f"{_PREFIX}{name}"
            self._channel = channel
            self._ci.cell.register_request_cb(
                channel=channel,
                topic="*",
                cb=self._receive_message,
            )
        self.logger.info(f"LocalCellPipe opened channel '{channel}' peer={self._peer_fqcn}")

    def send(self, msg: Message, timeout=None) -> bool:
        with self._pipe_lock:
            if self._closed:
                raise BrokenPipeError("pipe closed")

        optional = msg.topic in (Topic.END, Topic.ABORT, Topic.HEARTBEAT)

        if msg.topic == Topic.HEARTBEAT:
            extra = {_HEADER_HB_SEQ: self._hb_seq}
            self._hb_seq += 1
            self._ci.cell.fire_and_forget(
                channel=self._channel,
                topic=msg.topic,
                targets=[self._peer_fqcn],
                message=_to_cell_message(msg, extra),
                optional=optional,
            )
            return True

        if not timeout and msg.topic in (Topic.END, Topic.ABORT):
            timeout = 5.0  # keep connection alive long enough for message to go out

        request = _to_cell_message(msg)
        request.set_header(MessageHeaderKey.MSG_ROOT_ID, msg.msg_id)

        reply = self._ci.cell.send_request(
            channel=self._channel,
            topic=msg.topic,
            target=self._peer_fqcn,
            request=request,
            timeout=timeout,
            optional=optional,
        )
        if reply is not None:
            rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
            if rc == ReturnCode.OK:
                return True
            err = f"failed to send '{msg.topic}' to '{self._peer_fqcn}' on '{self._channel}': {rc}"
            if optional:
                self.logger.debug(err)
            else:
                self.logger.error(err)
        return False

    def _receive_message(self, request: CellMessage) -> Union[None, CellMessage]:
        sender = request.get_header(MessageHeaderKey.ORIGIN)
        if self._peer_fqcn != sender:
            raise RuntimeError(f"LocalCellPipe peer FQCN mismatch: expected {self._peer_fqcn!r} but got {sender!r}")
        msg = _from_cell_message(request)
        self._received_msgs.put_nowait(msg)
        return make_reply(ReturnCode.OK)

    def receive(self, timeout=None) -> Union[None, Message]:
        try:
            if timeout:
                return self._received_msgs.get(block=True, timeout=timeout)
            else:
                return self._received_msgs.get_nowait()
        except queue.Empty:
            return None

    def clear(self):
        while not self._received_msgs.empty():
            try:
                self._received_msgs.get_nowait()
            except queue.Empty:
                break

    def can_resend(self) -> bool:
        return True

    def close(self):
        with self._pipe_lock:
            if self._closed:
                return
            self._closed = True
        if self._ci is not None:
            self._ci.close_pipe(self)

    # ── Export / import for config-file hand-off ──────────────────────────────

    def export(self, export_mode: str) -> Tuple[str, dict]:
        """Return ``(class_path, constructor_args)`` for this pipe.

        ExportMode.PEER
            PASSIVE side: return args so the ACTIVE side can connect.
            Retrieves the dynamically bound internal-listener URL via
            ``CoreCell.get_internal_listener_url()`` (always available after
            ``open()`` has been called).

            ACTIVE side: return args for the PASSIVE side (mode flipped,
            parent_url left empty because PASSIVE doesn't need it).

        ExportMode.SELF
            Return args that recreate *this* side of the pipe.
        """
        cls_path = f"{self.__module__}.{self.__class__.__name__}"
        mode_str = f"{self.mode}".strip().lower()

        if export_mode == ExportMode.SELF:
            return cls_path, {
                "mode": self.mode,
                "site_name": self.site_name,
                "token": self.token,
                "parent_url": self._parent_url,
            }

        # ExportMode.PEER
        if mode_str == "passive":
            internal_url = self._ci.cell.core_cell.get_internal_listener_url()
            if not internal_url:
                raise RuntimeError(
                    "LocalCellPipe PASSIVE has no internal listener URL. "
                    "Ensure open() has been called before export()."
                )
            return cls_path, {
                "mode": "ACTIVE",
                "site_name": self.site_name,
                "token": self.token,
                "parent_url": internal_url,
            }
        else:  # active → peer is passive
            return cls_path, {
                "mode": "PASSIVE",
                "site_name": self.site_name,
                "token": self.token,
                "parent_url": "",
            }

    def get_last_peer_active_time(self) -> float:
        """Compatibility shim; LocalCellPipe does not track peer-active times."""
        return 0.0
