# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import queue
import threading
import time
from typing import Tuple, Union

from nvflare.apis.fl_constant import ConnPropKey, FLMetaKey, SystemVarName
from nvflare.fuel.data_event.utils import get_scope_property
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.cell import Message as CellMessage
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.net_agent import NetAgent
from nvflare.fuel.f3.cellnet.utils import make_reply
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.sec.authn import set_add_auth_headers_filters
from nvflare.fuel.utils.attributes_exportable import ExportMode
from nvflare.fuel.utils.config_service import search_file
from nvflare.fuel.utils.constants import Mode
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.fuel.utils.validation_utils import check_object_type, check_str

from .pipe import Message, Pipe, Topic

SSL_ROOT_CERT = "rootCA.pem"
_PREFIX = "cell_pipe."

_HEADER_MSG_TYPE = _PREFIX + "msg_type"
_HEADER_MSG_ID = _PREFIX + "msg_id"
_HEADER_REQ_ID = _PREFIX + "req_id"
_HEADER_START_TIME = _PREFIX + "start"
_HEADER_HB_SEQ = _PREFIX + "hb_seq"


def _cell_fqcn(mode, site_name, token, parent_fqcn):
    # The FQCN of the cell must be unique in the whole cellnet.
    # We use the combination of mode, site_name, and token to derive the value of FQCN
    # Since the token is usually used across all sites, the "site_name" differentiate cell on one site from another.
    # The two peer pipes on the same site share the same site_name and token, but are differentiated by their modes.
    base = f"{site_name}_{token}_{mode}"
    if parent_fqcn == FQCN.ROOT_SERVER:
        return base
    else:
        return FQCN.join([parent_fqcn, base])


def _to_cell_message(msg: Message, extra=None) -> CellMessage:
    headers = {_HEADER_MSG_TYPE: msg.msg_type, _HEADER_MSG_ID: msg.msg_id, _HEADER_START_TIME: time.time()}
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


class _CellInfo:
    """
    A cell could be used by multiple pipes (e.g. one pipe for task interaction, another for metrics logging).
    """

    def __init__(self, site_name, cell, net_agent, auth_token, token_signature):
        self.site_name = site_name
        self.cell = cell
        self.auth_token = auth_token
        self.token_signature = token_signature
        self.net_agent = net_agent
        self.started = False
        self.pipes = []
        self.lock = threading.Lock()

    def start(self):
        with self.lock:
            if not self.started:
                self.cell.start()
                self.started = True

    def add_pipe(self, p):
        with self.lock:
            self.pipes.append(p)

    def close_pipe(self, p):
        with self.lock:
            try:
                self.pipes.remove(p)
                if len(self.pipes) == 0:
                    # all pipes are closed - close cell and agent
                    self.net_agent.close()
                    self.cell.stop()
            except:
                pass


class CellPipe(Pipe):
    """
    CellPipe is an implementation of `Pipe` that utilizes the `Cell` from NVFlare's foundation layer (f3) to
    do the communication.
    """

    _lock = threading.Lock()
    _cells_info = {}  # (root_url, site_name, token) => _CellInfo

    @classmethod
    def _build_cell(cls, site_name, fqcn, parent_conn_props, secure_mode, workspace_dir, logger):
        """Build a cell if necessary.
        The combination of (root_url, site_name, token) uniquely determine one cell.
        There can be multiple pipes on the same cell.

        Args:
            parent_conn_props: parent for this cell
            secure_mode: whether cellnet is in secure mode
            workspace_dir: workspace that contains startup kit for connecting to server. Needed only if secure_mode

        Returns:

        """
        with cls._lock:
            ci = cls._cells_info.get(fqcn)
            if not ci:
                if secure_mode:
                    root_cert_path = search_file(SSL_ROOT_CERT, workspace_dir)
                    if not root_cert_path:
                        raise ValueError(f"cannot find {SSL_ROOT_CERT} from config path {workspace_dir}")

                    credentials = {
                        DriverParams.CA_CERT.value: root_cert_path,
                    }
                else:
                    credentials = {}

                conn_sec = parent_conn_props.get(ConnPropKey.CONNECTION_SECURITY)
                if conn_sec:
                    credentials[DriverParams.CONNECTION_SECURITY.value] = conn_sec

                parent_url = parent_conn_props.get(ConnPropKey.URL)

                if FQCN.get_parent(fqcn):
                    # the cell has a parent: connect to the parent
                    cell_root = None
                    cell_parent_url = parent_url
                else:
                    # the cell has no parent: the parent_url is the root of the cellnet
                    cell_root = parent_url
                    cell_parent_url = None

                cell = Cell(
                    fqcn=fqcn,
                    root_url=cell_root,
                    secure=secure_mode,
                    credentials=credentials,
                    parent_url=cell_parent_url,
                    create_internal_listener=False,
                )

                auth_token = get_scope_property(scope_name=site_name, key=FLMetaKey.AUTH_TOKEN, default="NA")
                token_signature = get_scope_property(site_name, FLMetaKey.AUTH_TOKEN_SIGNATURE, default="NA")

                net_agent = NetAgent(cell)
                ci = _CellInfo(site_name, cell, net_agent, auth_token, token_signature)
                cls._cells_info[fqcn] = ci

                set_add_auth_headers_filters(cell, ci.site_name, ci.auth_token, ci.token_signature)
            return ci

    def __init__(
        self,
        mode: Mode,
        site_name: str,
        token: str,
        root_url: str = "",
        secure_mode: bool = True,
        workspace_dir: str = "",
    ):
        """The constructor of the CellPipe.

        Args:
            mode: passive or active mode
            site_name (str): name of the FLARE site
            token (str): unique id to guarantee the uniqueness of cell's FQCN.
            root_url (str): the root url of the cellnet that the pipe's cell will join
            secure_mode (bool): whether connection to the root is secure (TLS)
            workspace_dir (str): the directory that contains startup for joining the cellnet. Required only in secure_mode
        """
        super().__init__(mode)
        self.logger = get_obj_logger(self)

        self.site_name = site_name
        self.token = token
        self.secure_mode = secure_mode
        self.workspace_dir = workspace_dir
        self.root_url = root_url

        # this section is needed by job config to prevent building cell when using SystemVarName arguments
        # TODO: enhance this part
        sysvarname_placeholders = ["{" + varname + "}" for varname in dir(SystemVarName)]
        if any([arg in sysvarname_placeholders for arg in [site_name, token, root_url, secure_mode, workspace_dir]]):
            return

        check_str("root_url", root_url)
        check_object_type("secure_mode", secure_mode, bool)
        check_str("token", token)
        check_str("site_name", site_name)
        check_str("workspace_dir", workspace_dir)

        # determine the endpoint for this pipe to connect to
        root_conn_props = get_scope_property(site_name, ConnPropKey.ROOT_CONN_PROPS)

        if root_conn_props:
            # Not in simulator
            if not isinstance(root_conn_props, dict):
                raise RuntimeError(f"expect root_conn_props for {site_name} to be dict but got {type(root_conn_props)}")

            cp_conn_props = get_scope_property(site_name, ConnPropKey.CP_CONN_PROPS)
            if cp_conn_props:
                if not isinstance(cp_conn_props, dict):
                    raise RuntimeError(f"expect cp_conn_props to be dict but got {type(cp_conn_props)}")

            url_to_conns = {
                root_conn_props.get(ConnPropKey.URL): root_conn_props,
                cp_conn_props.get(ConnPropKey.URL): cp_conn_props,
            }

            relay_conn_props = get_scope_property(site_name, ConnPropKey.RELAY_CONN_PROPS)
            if relay_conn_props:
                if not isinstance(relay_conn_props, dict):
                    raise RuntimeError(f"expect relay_conn_props to be dict but got {type(relay_conn_props)}")
                url_to_conns[relay_conn_props.get(ConnPropKey.URL)] = relay_conn_props

            if not root_url:
                # root_url not specified - use CP!
                root_url = cp_conn_props.get(ConnPropKey.URL)
                self.root_url = root_url

            conn_props = url_to_conns.get(self.root_url)
            if not conn_props:
                raise RuntimeError(f"cannot determine conn props for '{root_url}'")
        else:
            # this is running in simulator
            conn_props = {
                ConnPropKey.URL: root_url,
                ConnPropKey.FQCN: FQCN.ROOT_SERVER,
            }

        mode = f"{mode}".strip().lower()  # convert to lower case string
        fqcn = _cell_fqcn(mode, site_name, token, conn_props.get(ConnPropKey.FQCN))

        self.ci = self._build_cell(site_name, fqcn, conn_props, secure_mode, workspace_dir, self.logger)
        self.cell = self.ci.cell
        self.ci.add_pipe(self)

        if mode == "active":
            peer_mode = "passive"
        elif mode == "passive":
            peer_mode = "active"
        else:
            raise ValueError(f"invalid mode {mode} - must be 'active' or 'passive'")

        self.peer_fqcn = _cell_fqcn(peer_mode, site_name, token, conn_props.get(ConnPropKey.FQCN))
        self.received_msgs = queue.Queue()  # contains Message(s), not CellMessage(s)!
        self.channel = None  # the cellnet message channel
        self.pipe_lock = threading.Lock()  # used to ensure no msg to be sent after closed
        self.closed = False
        self.last_peer_active_time = 0.0
        self.hb_seq = 1

    def _update_peer_active_time(self, msg: CellMessage, ch_name: str, msg_type: str):
        origin = msg.get_header(MessageHeaderKey.ORIGIN)
        if origin == self.peer_fqcn:
            self.logger.debug(f"{time.time()}: _update_peer_active_time: {ch_name=} {msg_type=} {msg.headers}")
            self.last_peer_active_time = time.time()

    def get_last_peer_active_time(self):
        return self.last_peer_active_time

    def set_cell_cb(self, channel_name: str):
        # This allows multiple pipes over the same cell (e.g. one channel for tasks, another for metrics),
        # as long as different pipes use different cell message channels
        self.channel = f"{_PREFIX}{channel_name}"
        self.cell.register_request_cb(channel=self.channel, topic="*", cb=self._receive_message)
        self.cell.core_cell.add_incoming_request_filter(
            channel="*", topic="*", cb=self._update_peer_active_time, ch_name=channel_name, msg_type="req"
        )
        self.cell.core_cell.add_incoming_reply_filter(
            channel="*", topic="*", cb=self._update_peer_active_time, ch_name=channel_name, msg_type="reply"
        )
        self.logger.info(f"registered CellPipe request CB for {self.channel}")

    def send(self, msg: Message, timeout=None) -> bool:
        """Sends the specified message to the peer.

        Args:
            msg: the message to be sent
            timeout: if specified, number of secs to wait for the peer to read the message.
                If not specified, wait indefinitely.

        Returns:
            Whether the message is read by the peer.
        """
        with self.pipe_lock:
            if self.closed:
                raise BrokenPipeError("pipe closed")

        # Note: the following code must not be within the lock scope
        # Otherwise only one message can be sent at a time!
        optional = False
        if msg.topic in [Topic.END, Topic.ABORT, Topic.HEARTBEAT]:
            optional = True

        if not timeout and msg.topic in [Topic.END, Topic.ABORT]:
            timeout = 5.0  # need to keep the connection for some time; otherwise the msg may not go out

        if msg.topic == Topic.HEARTBEAT:
            # for debugging purpose
            extra_headers = {_HEADER_HB_SEQ: self.hb_seq}
            self.hb_seq += 1

            # don't need to wait for reply!
            self.cell.fire_and_forget(
                channel=self.channel,
                topic=msg.topic,
                targets=[self.peer_fqcn],
                message=_to_cell_message(msg, extra_headers),
                optional=optional,
            )
            return True

        reply = self.cell.send_request(
            channel=self.channel,
            topic=msg.topic,
            target=self.peer_fqcn,
            request=_to_cell_message(msg),
            timeout=timeout,
            optional=optional,
        )
        if reply:
            rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
            if rc == ReturnCode.OK:
                return True
            else:
                err = f"failed to send '{msg.topic}' to '{self.peer_fqcn}' in channel '{self.channel}': {rc}"
                if optional:
                    self.logger.debug(err)
                else:
                    self.logger.error(err)
                return False
        else:
            return False

    def _receive_message(self, request: CellMessage) -> Union[None, CellMessage]:
        sender = request.get_header(MessageHeaderKey.ORIGIN)
        topic = request.get_header(MessageHeaderKey.TOPIC)
        self.logger.debug(f"got msg from peer {sender}: {topic}")

        if self.peer_fqcn != sender:
            raise RuntimeError(f"peer FQCN mismatch: expect {self.peer_fqcn} but got {sender}")
        msg = _from_cell_message(request)
        self.received_msgs.put_nowait(msg)
        return make_reply(ReturnCode.OK)

    def receive(self, timeout=None) -> Union[None, Message]:
        try:
            if timeout:
                return self.received_msgs.get(block=True, timeout=timeout)
            else:
                return self.received_msgs.get_nowait()
        except queue.Empty:
            return None

    def clear(self):
        while not self.received_msgs.empty():
            self.received_msgs.get_nowait()

    def can_resend(self) -> bool:
        return True

    def open(self, name: str):
        with self.pipe_lock:
            if self.closed:
                raise BrokenPipeError("pipe already closed")
            self.ci.start()
            self.set_cell_cb(name)

    def close(self):
        with self.pipe_lock:
            if self.closed:
                return
            self.ci.close_pipe(self)
            self.closed = True

    def export(self, export_mode: str) -> Tuple[str, dict]:
        if export_mode == ExportMode.SELF:
            mode = self.mode
        else:
            mode = Mode.ACTIVE if self.mode == Mode.PASSIVE else Mode.PASSIVE

        export_args = {
            "mode": mode,
            "site_name": self.site_name,
            "token": self.token,
            "root_url": self.root_url,
            "secure_mode": self.cell.core_cell.secure,
            "workspace_dir": self.workspace_dir,
        }
        return f"{self.__module__}.{self.__class__.__name__}", export_args
