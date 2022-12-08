#  Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import copy
import logging
import threading
import time
import traceback
import uuid

from typing import List, Union, Dict
from .conn_state import ConnState
from .message import Message
from .endpoint import Endpoint, EndpointMonitor
from .communicator import Communicator
from .headers import Headers
from .receiver import Receiver
from .driver_manager import DriverManager, DriverRequirementKey, DriverUse, Visibility


class TargetCellUnreachable(Exception):
    pass


class MessageHeaderKey:

    MSG_TYPE = "cellnet.msg_type"
    REQ_ID = "cellnet.req_id"
    REPLY_EXPECTED = "cellnet.reply_expected"
    TOPIC = "cellnet.topic"
    WAIT_UNTIL = "cellnet.wait_until"
    ORIGIN = "cellnet.origin"
    DESTINATION = "cellnet.destination"
    FROM_CELL = "cellnet.from"
    TO_CELL = "cellnet.to"
    CONN_URL = "cellnet.conn_url"
    CHANNEL = "cellnet.channel"
    RETURN_CODE = "cellnet.return_code"
    ERROR = "cellnet.error"


class ReturnCode:

    OK = "ok"
    TIMEOUT = "timeout"
    COMM_ERROR = "comm_error"
    PROCESS_EXCEPTION = "process_exception"   # receiver error processing request


class MessageType:

    REQ = "req"
    REPLY = "reply"
    RETURN = "return"   # return to sender due to forward error


class CellPropertyKey:

    FQCN = "fqcn"
    ROLES = "roles"


class CellRole:

    SERVER = "server"
    ROOT = "root"
    CLIENT = "client"


class CellAgent:
    """
    A CellAgent represents a cell in another cell.
    """

    def __init__(
            self,
            fqcn: str,
            roles: List[str],
            endpoint: Endpoint
    ):
        """

        Args:
            fqcn: FQCN of the cell represented
        """
        if not FQCN.is_valid(fqcn):
            raise ValueError(f"invalid FQCN '{fqcn}'")

        self.fqcn = fqcn
        self.roles = roles
        self.endpoint = endpoint

    def has_role(self, role: str) -> bool:
        return role in self.roles


class _CB:

    def __init__(self, cb, args, kwargs):
        self.cb = cb
        self.args = args
        self.kwargs = kwargs


class _Registry:

    def __init__(self):
        self.reg = {}  # channel/topic => _CB

    @staticmethod
    def _item_key(channel: str, topic: str) -> str:
        return f"{channel}:{topic}"

    def set(self, channel: str, topic: str, item):
        key = self._item_key(channel, topic)
        self.reg[key] = item

    def append(self, channel: str, topic: str, item):
        key = self._item_key(channel, topic)
        item_list = self.reg.get(key)
        if not item_list:
            item_list = []
            self.reg[key] = item_list
        item_list.append(item)

    def find(self, channel: str, topic: str):
        item = self.reg.get(self._item_key(channel, topic))
        if not item:
            # try topic * in channel
            item = self.reg.get(self._item_key(channel, "*"))

        if not item:
            # try topic * in channel *
            item = self.reg.get(self._item_key("*", "*"))

        return item


class _Waiter(threading.Event):

    def __init__(self, targets: List[str]):
        super().__init__()
        self.replies = {}        # target_id => reply
        self.reply_time = {}     # target_id => reply recv timestamp
        timeout_msg = make_reply(ReturnCode.TIMEOUT)
        for t in targets:
            self.replies[t] = timeout_msg
        self.send_time = time.time()
        self.id = str(uuid.uuid4())


class FQCN:

    SEPARATOR = "."
    ROOT_SERVER = "server"

    @staticmethod
    def split(fqcn: str) -> List[str]:
        return fqcn.split(FQCN.SEPARATOR)

    @staticmethod
    def join(path: List[str]) -> str:
        return FQCN.SEPARATOR.join(path)

    @staticmethod
    def is_valid(fqcn: str) -> bool:
        if not isinstance(fqcn, str):
            return False
        if not fqcn:
            return False

    @staticmethod
    def get_roles(fqcn: str) -> (int, List[str]):
        roles = []
        path = FQCN.split(fqcn)
        gen = len(path)
        if gen == 1:
            roles.append(CellRole.ROOT)
        if path[0] == FQCN.ROOT_SERVER:
            roles.append(CellRole.SERVER)
        else:
            roles.append(CellRole.CLIENT)
        return gen, roles

    @staticmethod
    def get_root(fqcn: str) -> str:
        path = FQCN.split(fqcn)
        return path[0]

    @staticmethod
    def same_family(fqcn1: str, fqcn2: str):
        return FQCN.get_root(fqcn1) == FQCN.get_root(fqcn2)


class Cell(Receiver, EndpointMonitor):

    APP_ID = 1

    def __init__(
            self,
            fqcn: str,
            root_url: str,
            secure: bool,
            credentials: dict,
            parent_url: str=None,
            max_timeout=3600,
    ):
        """

        Args:
            fqcn: the Cell's FQCN (Fully Qualified Cell Name)
            roles: roles of the cell
            credentials: credentials for secure connections
            root_url: the URL for backbone external connection
            max_timeout: default timeout for send_and_receive
            parent_url: url for connecting to parent cell

        FQCN is the name of all ancestor names, concatenated with colons.

        Example:
            server.J12345       (the cell for job J12345 on the server)
            server              (the root cell of server)
            nih_1.J12345     (the cell for job J12345 on client_1's site)
            client_1.J12345.R0  (the cell for rank R0 of J12345 on client_1 site)
            client_1            (he root cell of client_1)

        """
        self.my_fqcn = fqcn
        self.secure = secure
        self.roles = FQCN.get_roles(fqcn)
        self.generation = 0
        self.root_url = root_url
        self.agents = {}  # cell_fqcn => CellAgent
        self.agent_lock = threading.Lock()
        self.generation, self.roles = FQCN.get_roles(self.my_fqcn)

        ep = Endpoint(
            name=fqcn,
            url=root_url,
            properties={
                CellPropertyKey.FQCN: self.my_fqcn,
            })
        ep.conn_props = credentials

        self.communicator = Communicator(
            local_endpoint=ep
        )

        self.communicator.register_receiver(endpoint=None, app=self.APP_ID, receiver=self)
        self.communicator.register_monitor(monitor=self)
        self.req_reg = _Registry()
        self.in_req_filter_reg = _Registry()  # for request received
        self.out_reply_filter_reg = _Registry()  # for reply going out
        self.out_req_filter_reg = _Registry()  # for request sent
        self.in_reply_filter_reg = _Registry()  # for reply received
        self.error_handler_reg = _Registry()
        self.cell_connected_cb = None
        self.cell_connected_cb_args = None
        self.cell_connected_cb_kwargs = None
        self.cell_disconnected_cb = None
        self.cell_disconnected_cb_args = None
        self.cell_disconnected_cb_kwargs = None

        self.waiters = {}  # req_id => req
        self.stats_lock = threading.Lock()
        self.req_hw = 0
        self.num_sar_reqs = 0  # send-and-receive
        self.num_faf_reqs = 0
        self.num_timeout_reqs = 0

        # req_expiry specifies how long we keep requests in "reqs" table if they are
        # not answered or picked up
        if not max_timeout or max_timeout <= 0:
            max_timeout = 3600  # one hour
        self.max_timeout = max_timeout
        self.asked_to_stop = False
        self.running = False

        self._name = self.__class__.__name__
        self.logger = logging.getLogger(self._name)

        # add appropriate drivers based on roles of the cell
        # a cell can have at most two listeners: one for external, one for internal
        self.driver_manager = DriverManager()
        self.ext_listener = None        # external listener
        self.ext_listener_lock = threading.Lock()
        self.ext_listener_impossible = False

        self.int_listener = None        # backbone internal listener - only for cells with child cells

        # a cell could have any number of connectors: some for backbone, some for ad-hoc
        self.bb_ext_connector = None    # backbone external connector - only for Client cells
        self.bb_int_connector = None    # backbone internal connector - only for non-root cells

        # ad-hoc connectors
        self.adhoc_connectors = {}              # target cell fqcn => connector
        self.adhoc_connector_lock = threading.Lock()
        self.root_change_lock = threading.Lock()

        if CellRole.SERVER in self.roles:
            if CellRole.ROOT in self.roles:
                self._set_bb_for_server_root()
            else:
                self._set_bb_for_server_child(parent_url)
        else:
            # client side
            if CellRole.ROOT in self.roles:
                self._set_bb_for_client_root()
            else:
                self._set_bb_for_client_child(parent_url)

    def change_root(self, to_url: str):
        """
        Change to a different server url

        Args:
            to_url: the new url of the server root

        Returns:

        """
        with self.root_change_lock:
            if to_url == self.root_url:
                # already changed
                return

            self.root_url = to_url

            if CellRole.CLIENT not in self.roles:
                # only affect clients
                return

            # drop connections to all cells on server and their agents
            # drop the backbone connector
            if self.bb_ext_connector:
                self.communicator.delete_driver(self.bb_ext_connector)
                self.bb_ext_connector = None

            # drop ad-hoc connectors to cells on server
            with self.adhoc_connector_lock:
                cells_to_delete = []
                for to_cell in self.adhoc_connectors.keys():
                    if FQCN.get_root(to_cell) == FQCN.ROOT_SERVER:
                        cells_to_delete.append(to_cell)
                for c in cells_to_delete:
                    connector = self.adhoc_connectors.pop(c, None)
                    if connector:
                        self.communicator.delete_driver(connector)

            # drop agents
            with self.agent_lock:
                agents_to_delete = []
                for cell_fqcn in self.agents.keys():
                    if FQCN.get_root(cell_fqcn) == FQCN.ROOT_SERVER:
                        agents_to_delete.append(cell_fqcn)
                    for a in agents_to_delete:
                        self.agents.pop(a, None)

            # recreate backbone connector to the root
            if self.generation <= 2:
                self._set_bb_external_connector()

    def get_internal_listener_url(self) -> Union[None, str]:
        """
        Get the cell's internal listener url.
        This method should only be used for cells that need to have child cells.
        The url returned is to be passed to child of this cell to create connection
        Returns:

        """
        if not self.int_listener:
            return None
        return self.int_listener.get_connection_url()

    def _add_adhoc_connector(self, to_cell: str, url: str):
        with self.adhoc_connector_lock:
            if to_cell in self.adhoc_connectors:
                return self.adhoc_connectors[to_cell]

            reqs = {
                DriverRequirementKey.URL: url,
                DriverRequirementKey.VISIBILITY: Visibility.EXTERNAL,
                DriverRequirementKey.SECURE: self.secure,
                DriverRequirementKey.USE: DriverUse.ADHOC
            }
            connector = self.driver_manager.get_connector(reqs)
            self.adhoc_connectors[to_cell] = connector
            if connector:
                self.communicator.add_connector(connector)
            return connector

    def _set_internal_listener(self):
        # internal listener is always backbone
        if not self.int_listener:
            reqs = {
                DriverRequirementKey.VISIBILITY: Visibility.INTERNAL,
                DriverRequirementKey.USE: DriverUse.BACKBONE,
                DriverRequirementKey.SECURE: False
            }
            self.int_listener = self.driver_manager.get_listener(reqs)
            self.communicator.add_listener(self.int_listener)
        return self.int_listener

    def _set_external_listener(self, use: str):
        with self.ext_listener_lock:
            if not self.ext_listener and not self.ext_listener_impossible:
                reqs = {
                    DriverRequirementKey.USE: use,
                    DriverRequirementKey.VISIBILITY: Visibility.EXTERNAL,
                    DriverRequirementKey.SECURE: self.secure
                }
                if use == DriverUse.BACKBONE:
                    reqs[DriverRequirementKey.URL] = self.root_url
                self.ext_listener = self.driver_manager.get_listener(reqs)
                if self.ext_listener:
                    self.communicator.add_listener(self.ext_listener)
                else:
                    self.ext_listener_impossible = True
        return self.ext_listener

    def _set_bb_for_server_root(self):
        self._set_external_listener(DriverUse.BACKBONE)
        self._set_internal_listener()

    def _set_bb_external_connector(self):
        reqs = {
            DriverRequirementKey.URL: self.root_url,
            DriverRequirementKey.VISIBILITY: Visibility.EXTERNAL,
            DriverRequirementKey.SECURE: self.secure,
            DriverRequirementKey.USE: DriverUse.BACKBONE
        }
        self.bb_ext_connector = self.driver_manager.get_connector(reqs)
        self.communicator.add_connector(self.bb_ext_connector)

    def _set_bb_for_client_root(self):
        self._set_bb_external_connector()
        self._set_internal_listener()

    def _set_bb_internal_connector(self, url: str):
        reqs = {
            DriverRequirementKey.URL: url,
            DriverRequirementKey.VISIBILITY: Visibility.INTERNAL,
            DriverRequirementKey.SECURE: False,
            DriverRequirementKey.USE: DriverUse.BACKBONE
        }
        self.bb_int_connector = self.driver_manager.get_connector(reqs)
        self.communicator.add_connector(self.bb_int_connector)

    def _set_bb_for_server_child(self, internal_url: str):
        self._set_bb_internal_connector(internal_url)

    def _set_bb_for_client_child(self, internal_url: str):
        self._set_bb_internal_connector(internal_url)
        if self.generation == 2:
            # we only connect to server root for gen2 child (the job cell)
            self._set_bb_external_connector()

    def set_cell_connected_cb(
            self,
            cb,
            *args,
            **kwargs
    ):
        """
        Set a callback that is called when an external cell is connected.

        Args:
            cb: the callback function. It must follow the signature of cell_connected_cb_signature.
            *args: args to be passed to the cb.
            **kwargs: kwargs to be passed to the cb

        Returns: None

        """
        self.cell_connected_cb = cb
        self.cell_connected_cb_args = args
        self.cell_connected_cb_kwargs = kwargs

    def set_cell_disconnected_cb(
            self,
            cb,
            *args,
            **kwargs
    ):
        """
        Set a callback that is called when an external cell is disconnected.

        Args:
            cb: the callback function. It must follow the signature of cell_disconnected_cb_signature.
            *args: args to be passed to the cb.
            **kwargs: kwargs to be passed to the cb

        Returns: None

        """
        self.cell_disconnected_cb = cb
        self.cell_disconnected_cb_args = args
        self.cell_disconnected_cb_kwargs = kwargs

    def start(self):
        """
        Start the cell after it is fully set up (connectors and listeners are added, CBs are setup)

        Returns:

        """
        self.communicator.start()

    def stop(self):
        """
        Stop the cell. Once the cell is stopped, it won't be able to send/receive messages.

        Returns:

        """
        self.communicator.stop()

    def register_request_cb(
            self,
            channel: str,
            topic: str,
            cb,
            *args,
            **kwargs
    ):
        """
        Register a callback for handling request. The CB must follow request_cb_signature.

        Args:
            channel: the channel of the request
            topic: topic of the request
            cb:
            *args:
            **kwargs:

        Returns:

        """
        self.req_reg.set(channel, topic, _CB(cb, args, kwargs))

    def add_incoming_request_filter(
            self,
            channel: str,
            topic: str,
            cb,
            *args,
            **kwargs
    ):
        self.in_req_filter_reg.append(channel, topic, _CB(cb, args, kwargs))

    def add_outgoing_reply_filter(
            self,
            channel: str,
            topic: str,
            cb,
            *args,
            **kwargs
    ):
        self.out_reply_filter_reg.append(channel, topic, _CB(cb, args, kwargs))

    def add_outgoing_request_filter(
            self,
            channel: str,
            topic: str,
            cb,
            *args,
            **kwargs
    ):
        self.out_req_filter_reg.append(channel, topic, _CB(cb, args, kwargs))

    def add_incoming_reply_filter(
            self,
            channel: str,
            topic: str,
            cb,
            *args,
            **kwargs
    ):
        self.in_reply_filter_reg.append(channel, topic, _CB(cb, args, kwargs))

    def add_error_handler(
            self,
            channel: str,
            topic: str,
            cb,
            *args,
            **kwargs
    ):
        self.error_handler_reg.set(channel, topic, _CB(cb, args, kwargs))

    def _filter_outgoing_request(
            self,
            channel: str,
            topic: str,
            request: Message
    ) -> Union[None, Message]:
        cbs = self.out_req_filter_reg.find(channel, topic)
        if not cbs:
            return None
        for _cb in cbs:
            assert isinstance(_cb, _CB)
            result = _cb.cb(
                cell=self,
                channel=channel,
                topic=topic,
                msg=request,
                *_cb.args,
                **_cb.kwargs
            )

    def _try_path(self, fqcn_path: List[str]) -> Union[None, Endpoint]:
        target = FQCN.join(fqcn_path)
        agent = self.agents.get(target, None)
        if agent:
            # there is a direct path to the target call
            return agent.endpoint

        if len(fqcn_path) == 1:
            return None
        return self._try_path(fqcn_path[:-1])

    def _find_ep(self, target_fqcn: str) -> Union[None, Endpoint]:
        path = FQCN.split(target_fqcn)
        ep = self._try_path(path)
        if ep:
            return ep

        # can't find endpoint based on the target's FQCN
        # let my parent(s) handle it
        ep = None
        path = FQCN.split(self.my_fqcn)
        if len(path) > 1:
            ep = self._try_path(path[:-1])
        if ep:
            return ep

    def find_endpoint(self, target_fqcn: str) -> Union[None, Endpoint]:
        if target_fqcn == self.my_fqcn:
            # sending request to myself? Not allowed!
            return None

        ep = self._find_ep(target_fqcn)
        if not ep:
            # cannot find endpoint through FQCN path
            # use the server root agent as last resort
            # this is the case that a client cell tries to talk to another client cell
            # and there is no direct link to it.
            # we assume that all client roots connect to the server root.
            with self.agent_lock:
                for _, agent in self.agents.items():
                    if agent.has_role(CellRole.SERVER) and agent.has_role(CellRole.ROOT):
                        return agent.endpoint
        return None

    def _async_send(self, to_endpoint: Endpoint, message: Message) -> str:
        err = ""
        try:
            self.communicator.send(to_endpoint, Cell.APP_ID, message)
        except:
            traceback.print_exc()
            err = "CommError"
        return err

    def _send_to_targets(
            self,
            channel: str,
            topic: str,
            targets: Union[str, List[str]],
            request: Message,
            headers: dict
    ) -> Dict[str, bool]:
        if not self.running:
            raise RuntimeError("Messenger is not running")

        result = {}
        if isinstance(targets, str):
            # turn it to list
            targets = [targets]

        reachable_targets = {}  # target fqcn => endpoint
        for t in targets:
            ep = self.find_endpoint(t)
            if ep:
                reachable_targets[t] = ep
            else:
                self.logger.error(f"no path to cell '{t}'")
                result[t] = False

        request.add_headers({
            MessageHeaderKey.CHANNEL: channel,
            MessageHeaderKey.TOPIC: topic,
            MessageHeaderKey.ORIGIN: self.my_fqcn,
            MessageHeaderKey.FROM_CELL: self.my_fqcn,
            MessageHeaderKey.MSG_TYPE: MessageType.REQ
        })

        request.add_headers(headers)

        for t, ep in reachable_targets.items():
            if len(reachable_targets) > 1:
                req = Message(
                    headers=copy.deepcopy(request.headers),
                    payload=request.payload
                )
            else:
                req = request

            req.add_headers({
                MessageHeaderKey.DESTINATION: t,
                MessageHeaderKey.TO_CELL: ep.name
            })

            # is this a direct path?
            if t != ep.name and not FQCN.same_family(t, self.my_fqcn):
                # Not a direct path since the destination and the next leg are not the same
                if CellRole.SERVER in self.roles:
                    # server side - try to create a listener and let the peer know the endpoint
                    listener = self._set_external_listener(DriverUse.ADHOC)
                    if listener:
                        conn_url = listener.get_connection_url()
                        req.set_header(MessageHeaderKey.CONN_URL, conn_url)
            try:
                self.communicator.send(
                    endpoint=ep,
                    app=Cell.APP_ID,
                    message=req
                )
                result[t] = True
            except:
                result[t] = False
        return result

    def send_request(
            self,
            channel: str,
            topic: str,
            target: str,
            request: Message,
            timeout=None) -> Message:
        result = self.broadcast_request(channel, topic, target, request, timeout)
        assert isinstance(result, dict)
        return result.get(target)

    def broadcast_request(
            self,
            channel: str,
            topic: str,
            targets: Union[str, List[str]],
            request: Message,
            timeout=None) -> Dict[str, Message]:
        """
        Send a message over a channel to specified destination cell(s), and wait for reply

        Args:
            channel: channel for the message
            topic: topic of the message
            targets: FQCN of the destination cell(s)
            request: message to be sent
            timeout: how long to wait for replies

        Returns: a dict of: cell_id => reply message

        """
        waiter = _Waiter(targets)
        self.waiters[waiter.id] = waiter
        now = time.time()

        try:
            status = self._send_to_targets(
                channel, topic, targets, request,
                {
                    MessageHeaderKey.REQ_ID: waiter.id,
                    MessageHeaderKey.REPLY_EXPECTED: True,
                    MessageHeaderKey.WAIT_UNTIL: time.time() + timeout
                }
            )

            send_count = 0
            err_reply = make_reply(ReturnCode.COMM_ERROR)
            for t, sent in status.items():
                if sent:
                    send_count += 1
                else:
                    waiter.replies[t] = err_reply
                    waiter.reply_time[t] = now

            if send_count > 0:
                self.num_sar_reqs += 1
                num_reqs = len(self.waiters)
                if self.req_hw < num_reqs:
                    self.req_hw = num_reqs

                # wait for reply
                if not waiter.wait(timeout=timeout):
                    # timeout
                    with self.stats_lock:
                        self.num_timeout_reqs += 1
        except BaseException as ex:
            raise ex
        finally:
            self.waiters.pop(waiter.id, None)
        return waiter.replies

    def fire_and_forget(
            self,
            channel: str,
            topic: str,
            targets: Union[str, List[str]],
            message: Message):
        """
        Send a message over a channel to specified destination cell(s), and do not wait for replies.

        Args:
            channel: channel for the message
            topic: topic of the message
            targets: one or more destination cell IDs. None means all.
            message: message to be sent

        Returns: None

        """
        self._send_to_targets(
            channel, topic, targets, message,
            {
                MessageHeaderKey.REPLY_EXPECTED: False
            }
        )

    def send_reply(
            self,
            reply: Message,
            to_cell: str,
            for_req_ids: List[str]
    ):
        """
        Send a reply to respond to one or more requests.
        This is useful if the request receiver needs to delay its reply as follows:
        - When a request is received, if it's not ready to reply (e.g. waiting for additional requests from
         other cells), simply remember the REQ_ID and returns None;
        - The receiver may queue up multiple such requests
        - When ready, call this method to send the reply for all the queued requests

        Args:
            reply:
            to_cell:
            for_req_ids:

        Returns:

        """
        reply.add_headers(
            {
                MessageHeaderKey.FROM_CELL: self.my_fqcn,
                MessageHeaderKey.ORIGIN: self.my_fqcn,
                MessageHeaderKey.DESTINATION: to_cell,
                MessageHeaderKey.REQ_ID: for_req_ids,
                MessageHeaderKey.MSG_TYPE: MessageType.REPLY,
            }
        )

        ep = self.find_endpoint(to_cell)
        if not ep:
            raise TargetCellUnreachable(f"no path to cell {to_cell}")
        reply.set_header(MessageHeaderKey.TO_CELL, ep.name)
        self.communicator.send(ep, Cell.APP_ID, reply)

    def process(self, endpoint: Endpoint, app: int, message: Message):
        # this is the receiver callback
        try:
            self._process_received_msg(endpoint, app, message)
        except:
            traceback.print_exc()

    def _process_request(
            self,
            origin: str,
            message: Message) -> Union[None, Message]:
        # this is a request for me - dispatch to the right CB
        channel = message.get_header(MessageHeaderKey.CHANNEL, "")
        topic = message.get_header(MessageHeaderKey.TOPIC, "")
        _cb = self.req_reg.find(channel, topic)
        if not _cb:
            self.logger.error(
                f"No callback for request ({topic}@{channel}) from cell '{origin}'")
            return make_reply(ReturnCode.PROCESS_EXCEPTION, error="no callback")

        try:
            assert isinstance(_cb, _CB)
            reply = _cb.cb(self, channel, topic, message, *_cb.args, **_cb.kwargs)
            if not reply:
                # the CB doesn't have anything to reply
                return None

            if not isinstance(reply, Message):
                self.logger.error(
                    f"bad result from request CB for topic {topic} on channel {channel}: "
                    f"expect Message but got {type(reply)}"
                )
                return make_reply(ReturnCode.PROCESS_EXCEPTION, error="bad cb result")
        except:
            traceback.print_exc()
            return make_reply(ReturnCode.PROCESS_EXCEPTION, error="cb exception")

        reply_expected = message.get_header(MessageHeaderKey.REPLY_EXPECTED, False)
        if not reply_expected:
            # this is fire and forget
            return None

        wait_until = message.get_header(MessageHeaderKey.WAIT_UNTIL, None)
        if isinstance(wait_until, float) and time.time() > wait_until:
            # no need to reply since peer already gave up waiting by now
            return None

        # send the reply back
        if not reply.headers.get(MessageHeaderKey.RETURN_CODE):
            reply.set_header(MessageHeaderKey.RETURN_CODE, ReturnCode.OK)
        return reply

    def _forward(self, endpoint: Endpoint, origin: str, destination: str, msg_type: str, message: Message):
        # not for me - need to forward it
        ep = self.find_endpoint(destination)
        if ep:
            message.add_headers({
                MessageHeaderKey.FROM_CELL: self.my_fqcn,
                MessageHeaderKey.TO_CELL: ep.name
            })
            err = self._async_send(to_endpoint=ep, message=message)
            if not err:
                return

        # cannot forward
        self.logger.error(f"Cannot forward {msg_type} to cell '{destination}' for {origin} - no path")
        if msg_type == MessageType.REQ:
            reply_expected = message.get_header(MessageHeaderKey.REPLY_EXPECTED, False)
            if not reply_expected:
                return

            wait_until = message.get_header(MessageHeaderKey.WAIT_UNTIL, None)
            if isinstance(wait_until, float) and time.time() > wait_until:
                # no need to reply since peer already gave up waiting by now
                return

            # tell the requester that message couldn't be delivered
            req_id = message.get_header(MessageHeaderKey.REQ_ID, "")
            reply = make_reply(ReturnCode.COMM_ERROR, error="cannot forward")
            reply.add_headers(
                {
                    MessageHeaderKey.FROM_CELL: self.my_fqcn,
                    MessageHeaderKey.TO_CELL: endpoint.name,
                    MessageHeaderKey.ORIGIN: destination,
                    MessageHeaderKey.DESTINATION: origin,
                    MessageHeaderKey.REQ_ID: [req_id],
                    MessageHeaderKey.MSG_TYPE: MessageType.RETURN,
                }
            )
            self._async_send(endpoint, reply)
        else:
            # msg_type is either RETURN or REPLY - drop it.
            pass

    def _process_reply(self, origin: str, message: Message):
        req_ids = message.get_header(MessageHeaderKey.REQ_ID)
        if not req_ids:
            raise RuntimeError("Proto Error: received reply does not have REQ_ID header")

        if isinstance(req_ids, str):
            req_ids = [req_ids]

        if not isinstance(req_ids, list):
            raise RuntimeError(f"Proto Error: REQ_ID must be list of ids but got {type(req_ids)}")

        for rid in req_ids:
            waiter = self.waiters.get(rid, None)
            if waiter:
                assert isinstance(waiter, _Waiter)
                if origin not in waiter.replies:
                    self.logger.error(f"received reply for {rid} from unexpected cell {origin}")
                    return
                waiter.replies[origin] = message
                waiter.reply_time[origin] = time.time()

                # all targets replied?
                all_targets_replied = True
                for t, _ in waiter.replies.items():
                    if not waiter.reply_time.get(t):
                        all_targets_replied = False
                        break

                if all_targets_replied:
                    self.logger.debug(f"replies received from all {len(waiter.replies)} targets for req {rid}")
                    waiter.set()  # trigger the waiting requests!
                else:
                    self.logger.debug(f"replies not received from all {len(waiter.replies)} targets for req {rid}")
            else:
                self.logger.debug(f"no waiter for req {rid}")

    def _process_received_msg(self, endpoint: Endpoint, app: int, message: Message):
        msg_type = message.get_header(MessageHeaderKey.MSG_TYPE)
        if not msg_type:
            raise RuntimeError("Proto Error: missing MSG_TYPE in received message")

        origin = message.get_header(MessageHeaderKey.ORIGIN)
        if not origin:
            raise RuntimeError("Proto Error: missing ORIGIN header in received message")

        # is this msg for me?
        destination = message.get_header(MessageHeaderKey.DESTINATION)
        if not destination:
            raise RuntimeError("Proto Error: missing DESTINATION header in received message")

        if destination != self.my_fqcn:
            # not for me - need to forward it
            self._forward(endpoint, origin, destination, msg_type, message)
            return

        # this message is for me
        # handle ad-hoc
        my_conn_url = None
        if msg_type in [MessageType.REQ, MessageType.REPLY]:
            from_cell = message.get_header(MessageHeaderKey.FROM_CELL)
            if from_cell != origin and not FQCN.same_family(origin, self.my_fqcn):
                # this is a forwarded message, so no direct path from the origin to me
                conn_url = message.get_header(MessageHeaderKey.CONN_URL)
                if conn_url:
                    # the origin already has a listener
                    # create an ad-hoc connector to connect to the origin cell
                    self._add_adhoc_connector(origin, conn_url)
                elif msg_type == MessageType.REQ:
                    # see whether we can offer a listener
                    listener = self._set_external_listener(DriverUse.ADHOC)
                    if listener:
                        my_conn_url = listener.get_connection_url()

        if msg_type == MessageType.REQ:
            # this is a request for me - dispatch to the right CB
            reply = self._process_request(origin, message)
            if reply:
                req_id = message.get_header(MessageHeaderKey.REQ_ID, "")
                reply.add_headers(
                    {
                        MessageHeaderKey.FROM_CELL: self.my_fqcn,
                        MessageHeaderKey.ORIGIN: self.my_fqcn,
                        MessageHeaderKey.DESTINATION: origin,
                        MessageHeaderKey.TO_CELL: endpoint.name,
                        MessageHeaderKey.REQ_ID: req_id,
                        MessageHeaderKey.MSG_TYPE: MessageType.REPLY,
                    }
                )

                if my_conn_url:
                    reply.set_header(MessageHeaderKey.CONN_URL, my_conn_url)

                self.communicator.send(endpoint, Cell.APP_ID, reply)
            return

        # the message is either a reply or a return: handle replies
        self._process_reply(origin, message)

    def state_change(self, endpoint: Endpoint, state: ConnState):
        fqcn = endpoint.properties.get(CellPropertyKey.FQCN, None)
        if not fqcn:
            self.logger.critical(f"missing fqcn in endpoint {endpoint.name}")
            return

        if state == ConnState.READY:
            # create the CellAgent for this endpoint
            roles = endpoint.properties.get(CellPropertyKey.ROLES, None)
            if not roles:
                self.logger.critical(f"missing roles in endpoint {endpoint.name}")
                return

            agent = self.agents.get(fqcn)
            if not agent:
                agent = CellAgent(fqcn, roles, endpoint)
                with self.agent_lock:
                    self.agents[fqcn] = agent
            else:
                agent.endpoint = endpoint
                agent.roles = roles
            if self.cell_connected_cb is not None:
                try:
                    self.cell_connected_cb(
                        self, agent,
                        *self.cell_connected_cb_args,
                        **self.cell_connected_cb_kwargs
                    )
                except:
                    self.logger.error("exception in cell_connected_cb")
                    traceback.print_exc()

        elif state in [ConnState.DISCONNECTING, ConnState.IDLE]:
            # remove this agent
            with self.agent_lock:
                agent = self.agents.pop(fqcn, None)
            if agent and self.cell_disconnected_cb is not None:
                try:
                    self.cell_disconnected_cb(
                        self, agent,
                        *self.cell_disconnected_cb_args,
                        **self.cell_disconnected_cb_kwargs
                    )
                except:
                    self.logger.error("exception in cell_disconnected_cb")
                    traceback.print_exc()


def cell_connected_cb_signature(
        cell: Cell,
        connected_cell: CellAgent,
        *args, **kwargs
):
    """
    This is the signature of the cell_connected callback.

    Args:
        cell: the cell that calls the CB
        connected_cell: the cell that just got connected
        *args:
        **kwargs:

    Returns:

    """
    pass


def cell_disconnected_cb_signature(
        cell: Cell,
        disconnected_cell: CellAgent,
        *args, **kwargs
):
    pass


def request_cb_signature(
        cell: Cell,
        channel: str,
        topic: str,
        request: Message,
        *args, **kwargs
) -> Message:
    pass


def filter_cb_signature(
        cell: Cell,
        channel: str,
        topic: str,
        msg: Message,
        *args, **kwargs
) -> Message:
    pass


def error_handler_cb_signature(
        cell: Cell,
        from_cell: CellAgent,
        error_type: str,
        channel: str,
        topic: str,
        msg: Message,
        *args, **kwargs
) -> Message:
    pass


# Convenience functions
def make_reply(rc: str, error: str="", body=None) -> Message:
    headers = Headers()
    headers[MessageHeaderKey.RETURN_CODE] = rc
    if error:
        headers[MessageHeaderKey.ERROR] = error
    return Message(headers, payload=body)
