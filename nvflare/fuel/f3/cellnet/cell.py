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
import os
import signal
import threading
import time
import traceback
import uuid

from typing import List, Union, Dict
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.endpoint import Endpoint, EndpointMonitor, EndpointState
from nvflare.fuel.f3.communicator import Communicator, MessageReceiver

from .connector_manager import ConnectorManager
from .defs import (
    MessageHeaderKey, MessageType, TargetMessage,
    ReturnCode, CellPropertyKey, Encoding
)
from .utils import make_reply, new_message, format_log_message
from .fqcn import FQCN, FqcnInfo, same_family

import nvflare.fuel.utils.fobs as fobs

_BULK_CHANNEL = "cellnet.bulk"


class CellAgent:
    """
    A CellAgent represents a cell in another cell.
    """

    def __init__(
            self,
            fqcn: str,
            endpoint: Endpoint
    ):
        """

        Args:
            fqcn: FQCN of the cell represented
        """
        err = FQCN.validate(fqcn)
        if err:
            raise ValueError(f"Invalid FQCN '{fqcn}': {err}")

        self.info = FqcnInfo(FQCN.normalize(fqcn))
        self.endpoint = endpoint


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


class _BulkSender:

    def __init__(self, cell, target: str, interval, max_queue_size):
        self.cell = cell
        self.target = target
        self.interval = interval
        self.max_queue_size = max_queue_size
        self.messages = []
        self.last_send_time = 0
        self.lock = threading.Lock()

    def queue_message(
            self,
            channel: str,
            topic: str,
            message: Message
    ):
        with self.lock:
            tm = TargetMessage(
                target=self.target,
                channel=channel,
                topic=topic,
                message=message
            )
            self.messages.append(tm)

    def send(self, must_send: bool):
        num_msgs = len(self.messages)
        if num_msgs == 0:
            return

        if not must_send and time.time() - self.last_send_time < self.interval and num_msgs < self.max_queue_size:
            return

        with self.lock:
            bulk_msg = new_message(payload=self.messages)
            sent = self.cell.fire_and_forget(
                channel=_BULK_CHANNEL,
                topic="bulk",
                targets=[self.target],
                message=bulk_msg
            )
            if sent[self.target]:
                self.messages = []
                self.last_send_time = time.time()
            else:
                self.cell.logger.warning(f"can't send bulk message to {self.target}")
                if num_msgs > self.max_queue_size:
                    self.messages.pop(0)
                    self.cell.logger.warning(
                        f"bulk sender for {self.target}: "
                        f"dropped one message (queue size {num_msgs} > limit {self.max_queue_size}")


class Cell(MessageReceiver, EndpointMonitor):

    APP_ID = 1

    def __init__(
            self,
            fqcn: str,
            root_url: str,
            secure: bool,
            credentials: dict,
            create_internal_listener: bool = False,
            parent_url: str = None,
            max_timeout=3600,
            bulk_check_interval=0.5,
            bulk_send_interval=1.0,
            max_bulk_size=100
    ):
        """

        Args:
            fqcn: the Cell's FQCN (Fully Qualified Cell Name)
            credentials: credentials for secure connections
            root_url: the URL for backbone external connection
            secure: secure mode or not
            max_timeout: default timeout for send_and_receive
            create_internal_listener: whether to create an internal listener for child cells
            parent_url: url for connecting to parent cell

        FQCN is the names of all ancestor, concatenated with dots.
        Note: internal listener is automatically created for root cells.

        Example:
            server.J12345       (the cell for job J12345 on the server)
            server              (the root cell of server)
            nih_1.J12345        (the cell for job J12345 on client_1's site)
            client_1.J12345.R0  (the cell for rank R0 of J12345 on client_1 site)
            client_1            (he root cell of client_1)

        """
        self._name = self.__class__.__name__
        self.logger = logging.getLogger(self._name)

        err = FQCN.validate(fqcn)
        if err:
            raise ValueError(f"Invalid FQCN '{fqcn}': {err}")

        self.my_info = FqcnInfo(FQCN.normalize(fqcn))
        self.secure = secure
        self.root_url = root_url
        self.bulk_check_interval = bulk_check_interval
        self.bulk_send_interval = bulk_send_interval
        self.max_bulk_size = max_bulk_size
        self.bulk_senders = {}
        self.bulk_checker = threading.Thread(target=self._check_bulk)
        self.bulk_lock = threading.Lock()
        self.agents = {}  # cell_fqcn => CellAgent
        self.agent_lock = threading.Lock()

        self.logger.debug(f"Creating Cell: {self.my_info.fqcn}")

        ep = Endpoint(
            name=fqcn,
            conn_props=credentials,
            properties={
                CellPropertyKey.FQCN: self.my_info.fqcn,
            })

        self.communicator = Communicator(
            local_endpoint=ep
        )

        self.connector_manager = ConnectorManager(communicator=self.communicator, secure=secure)

        self.communicator.register_message_receiver(app_id=self.APP_ID, receiver=self)
        self.communicator.register_monitor(monitor=self)
        self.req_reg = _Registry()
        self.in_req_filter_reg = _Registry()  # for request received
        self.out_reply_filter_reg = _Registry()  # for reply going out
        self.out_req_filter_reg = _Registry()  # for request sent
        self.in_reply_filter_reg = _Registry()  # for reply received
        self.error_handler_reg = _Registry()
        self.cleanup_reg = _Registry()
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

        # add appropriate drivers based on roles of the cell
        # a cell can have at most two listeners: one for external, one for internal
        self.ext_listener = None        # external listener
        self.ext_listener_lock = threading.Lock()
        self.ext_listener_impossible = False

        self.int_listener = None        # backbone internal listener - only for cells with child cells

        # a cell could have any number of connectors: some for backbone, some for ad-hoc
        self.bb_ext_connector = None    # backbone external connector - only for Client cells
        self.bb_int_connector = None    # backbone internal connector - only for non-root cells

        # ad-hoc connectors: currently only support ad-hoc external connectors
        self.adhoc_connectors = {}              # target cell fqcn => connector
        self.adhoc_connector_lock = threading.Lock()
        self.root_change_lock = threading.Lock()

        if self.my_info.is_on_server:
            if self.my_info.is_root:
                self._set_bb_for_server_root()
            else:
                self._set_bb_for_server_child(parent_url, create_internal_listener)
        else:
            # client side
            if self.my_info.is_root:
                self._set_bb_for_client_root()
            else:
                self._set_bb_for_client_child(parent_url, create_internal_listener)

        self.stop_waiter = threading.Event()
        self.stop_waiter_thread = threading.Thread(target=self._wait_to_stop)

    def _wait_to_stop(self):
        self.logger.debug(f"=========== {self.my_info.fqcn}: Stop Waiter is waiting ==============")
        self.stop_waiter.wait()
        self.logger.debug(f"=========== {self.my_info.fqcn}: Stop Waiter is triggered: start cleanup ==============")
        time.sleep(2.0)  # let pending messages to go out
        t = threading.Thread(target=self._do_cleanup)
        t.start()
        # self._do_cleanup()

        # wait for 2 secs to give others time to clean up
        self.cleanup_waiter = threading.Event()
        if not self.cleanup_waiter.wait(timeout=2.0):
            self.logger.debug(f"======== {self.my_info.fqcn}: Cleanup did not complete within 2 secs")

        self.logger.info(f"{self.my_info.fqcn}: Good Bye!")
        os.kill(os.getpid(), signal.SIGKILL)

    def _do_cleanup(self):
        self.logger.debug(f"{self.my_info.fqcn}: Start system cleanup ...")
        cb_list = self.cleanup_reg.find("*", "*")
        if cb_list:
            self.logger.debug(f"{self.my_info.fqcn}: found {len(cb_list)} cleanup CBs")
            for _cb in cb_list:
                assert isinstance(_cb, _CB)
                try:
                    self.logger.debug(f"{self.my_info.fqcn}: calling a cleanup CB ...")
                    _cb.cb(*_cb.args, **_cb.kwargs)
                    self.logger.debug(f"{self.my_info.fqcn}: called a cleanup CB")
                except BaseException as ex:
                    self.logger.warning(f"{self.my_info.fqcn}: ignored exception {ex} from cleanup CB")
        else:
            self.logger.debug(f"{self.my_info.fqcn}: Nothing to cleanup!")

        # closing cell finally!
        self._close()
        self.logger.debug(f"{self.my_info.fqcn}: Cleanup Finished!")
        self.cleanup_waiter.set()

    def get_fqcn(self) -> str:
        return self.my_info.fqcn

    def is_cell_reachable(self, target_fqcn: str) -> bool:
        ep = self._find_endpoint(target_fqcn)
        return ep is not None

    def is_cell_connected(self, target_fqcn: str) -> bool:
        agent = self.agents.get(target_fqcn)
        return agent is not None

    def _set_bb_for_client_root(self):
        self._create_bb_external_connector()
        self._create_internal_listener()

    def _set_bb_for_client_child(self, parent_url: str, create_internal_listener: bool):
        self._create_internal_connector(parent_url)
        if create_internal_listener:
            self._create_internal_listener()

        if self.my_info.gen == 2:
            # we only connect to server root for gen2 child (the job cell)
            self._create_bb_external_connector()

    def _set_bb_for_server_root(self):
        self._create_external_listener(False)
        self._create_internal_listener()

    def _set_bb_for_server_child(self, parent_url: str, create_internal_listener: bool):
        self._create_internal_connector(parent_url)
        if create_internal_listener:
            self._create_internal_listener()

    def change_server_root(self, to_url: str):
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

            if self.my_info.is_on_server:
                # only affect clients
                return

            # drop connections to all cells on server and their agents
            # drop the backbone connector
            if self.bb_ext_connector:
                self.communicator.remove_connector(self.bb_ext_connector.handle)
                self.bb_ext_connector = None

            # drop ad-hoc connectors to cells on server
            with self.adhoc_connector_lock:
                cells_to_delete = []
                for to_cell in self.adhoc_connectors.keys():
                    to_cell_info = FqcnInfo(to_cell)
                    if to_cell_info.is_on_server:
                        cells_to_delete.append(to_cell)
                for c in cells_to_delete:
                    connector = self.adhoc_connectors.pop(c, None)
                    if connector:
                        self.communicator.remove_connector(connector.handle)

            # drop agents
            with self.agent_lock:
                agents_to_delete = []
                for fqcn, agent in self.agents.items():
                    assert isinstance(agent, CellAgent)
                    if agent.info.is_on_server:
                        agents_to_delete.append(fqcn)
                    for a in agents_to_delete:
                        self.agents.pop(a, None)

            # recreate backbone connector to the root
            if self.my_info.gen <= 2:
                self._create_bb_external_connector()

    def create_internal_listener(self):
        """
        Create the internal listener for child cells of this cell to connect to.

        Returns:

        """
        self._create_internal_listener()

    def get_internal_listener_url(self) -> Union[None, str]:
        """
        Get the cell's internal listener url.
        This method should only be used for cells that need to have child cells.
        The url returned is to be passed to child of this cell to create connection

        Returns: url for child cells to connect

        """
        if not self.int_listener:
            return None
        return self.int_listener.get_connection_url()

    def _add_adhoc_connector(self, to_cell: str, url: str):
        with self.adhoc_connector_lock:
            if to_cell in self.adhoc_connectors:
                return self.adhoc_connectors[to_cell]

            connector = self.connector_manager.get_external_connector(url, adhoc=True)
            self.adhoc_connectors[to_cell] = connector
            if connector:
                self.logger.info(f"{self.my_info.fqcn}: created adhoc connector to {url} on {to_cell}")
            else:
                self.logger.info(f"{self.my_info.fqcn}: cannot create adhoc connector to {url} on {to_cell}")
            return connector

    def _create_internal_listener(self):
        # internal listener is always backbone
        if not self.int_listener:
            self.int_listener = self.connector_manager.get_internal_listener()
            if self.int_listener:
                self.logger.info(f"{self.my_info.fqcn}: created backbone internal listener "
                                 f"for {self.int_listener.get_connection_url()}")
            else:
                raise RuntimeError(f"{self.my_info.fqcn}: cannot create backbone internal listener")
        return self.int_listener

    def _create_external_listener(self, adhoc: bool):
        with self.ext_listener_lock:
            if not self.ext_listener and not self.ext_listener_impossible:
                self.logger.debug(f"{os.getpid()}: {self.my_info.fqcn}: trying create ext listener: adhoc={adhoc}")
                if not adhoc:
                    url = self.root_url
                else:
                    url = ""

                self.ext_listener = self.connector_manager.get_external_listener(url, adhoc)
                if self.ext_listener:
                    if not adhoc:
                        self.logger.info(
                            f"{os.getpid()}: {self.my_info.fqcn}: "
                            f"created backbone external listener for {self.root_url}")
                    else:
                        self.logger.info(f"{os.getpid()}: {self.my_info.fqcn}: created adhoc external listener "
                                         f"for {self.ext_listener.get_connection_url()}")
                else:
                    if not adhoc:
                        raise RuntimeError(
                            f"{os.getpid()}: {self.my_info.fqcn}: "
                            f"cannot create backbone external listener for {self.root_url}")
                    else:
                        self.logger.warning(
                            f"{os.getpid()}: {self.my_info.fqcn}: cannot create adhoc external listener")

                    self.ext_listener_impossible = True
        return self.ext_listener

    def _create_bb_external_connector(self):
        self.bb_ext_connector = self.connector_manager.get_external_connector(self.root_url, False)
        if self.bb_ext_connector:
            self.logger.info(f"{self.my_info.fqcn}: created backbone external connector to {self.root_url}")
        else:
            raise RuntimeError(f"{self.my_info.fqcn}: cannot create backbone external connector to {self.root_url}")

    def _create_internal_connector(self, url: str):
        self.bb_int_connector = self.connector_manager.get_internal_connector(url)
        if self.bb_int_connector:
            self.logger.info(f"{self.my_info.fqcn}: created backbone internal connector to {url} on parent")
        else:
            raise RuntimeError(f"{self.my_info.fqcn}: cannot create backbone internal connector to {url} on parent")

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
        self.register_request_cb(
            channel=_BULK_CHANNEL,
            topic="*",
            cb=self._process_bulk_message
        )
        self.bulk_checker.start()
        self.communicator.start()
        self.stop_waiter_thread.start()
        self.running = True

    def stop(self):
        if self.running:
            self.running = False
            # trigger stop process
            self.stop_waiter.set()

    def _close(self):
        """
        Cleanup the cell. Once the cell is stopped, it won't be able to send/receive messages.

        Returns:

        """
        self.logger.debug(f"{self.my_info.fqcn}: Closing Cell")
        try:
            self.communicator.stop()
        except:
            self.logger.error(f"{self.my_info.fqcn}: error stopping Communicator")
            traceback.print_exc()

        self.logger.debug(f"{self.my_info.fqcn}: Communicator Stopped!")
        self.running = False
        self.asked_to_stop = True
        self.bulk_checker.join()
        self.logger.debug(f"{self.my_info.fqcn}: CELL closed!")

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

    def add_cleanup_cb(
            self,
            cb,
            *args,
            **kwargs
    ):
        self.cleanup_reg.append("*", "*", _CB(cb, args, kwargs))

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
            if result:
                return result

    def _try_path(self, fqcn_path: List[str]) -> Union[None, Endpoint]:
        self.logger.debug(f"{self.my_info.fqcn}: trying path {fqcn_path} ...")
        target = FQCN.join(fqcn_path)
        agent = self.agents.get(target, None)
        if agent:
            # there is a direct path to the target call
            self.logger.debug(f"{self.my_info.fqcn}: got cell agent for {target}")
            return agent.endpoint
        else:
            self.logger.debug(f"{self.my_info.fqcn}: no CellAgent for {target}")

            if FQCN.is_parent(self.my_info.fqcn, target):
                raise RuntimeError(f"{self.my_info.fqcn}: backbone broken: no path to child {target}")

            elif FQCN.is_parent(target, self.my_info.fqcn):
                raise RuntimeError(f"{self.my_info.fqcn}: backbone broken: no path to parent {target}")

        if len(fqcn_path) == 1:
            return None
        return self._try_path(fqcn_path[:-1])

    def _find_endpoint(self, target_fqcn: str) -> Union[None, Endpoint]:
        try:
            return self._try_find_ep(target_fqcn)
        except:
            traceback.print_exc()
            return None

    def _try_find_ep(self, target_fqcn: str) -> Union[None, Endpoint]:
        self.logger.debug(f"{self.my_info.fqcn}: finding path to {target_fqcn}")
        if target_fqcn == self.my_info.fqcn:
            # sending request to myself? Not allowed!
            self.logger.error(f"{self.my_info.fqcn}: sending message to self is not allowed")
            return None

        target_info = FqcnInfo(target_fqcn)
        if same_family(self.my_info, target_info):
            self.logger.debug(f"{self.my_info.is_root}: find path in the same family")
            return self._try_path(target_info.path)

        # not the same family
        ep = self._try_path(target_info.path)
        if ep:
            return ep

        # cannot find path to the target
        # try the server root
        # we assume that all client roots connect to the server root.
        with self.agent_lock:
            for _, agent in self.agents.items():
                if agent.info.is_on_server and agent.info.is_root:
                    return agent.endpoint

        # no direct path to the server root
        # let my parent handle it if I have a parent
        if self.my_info.gen > 1:
            ep = self._try_path(self.my_info.path[:-1])
            if ep:
                return ep

        self.logger.error(f"{self.my_info.fqcn}: cannot find path to {target_fqcn}")
        return None

    def _send_to_endpoint(self, to_endpoint: Endpoint, message: Message) -> str:
        err = ""
        try:
            encoding = message.get_header(MessageHeaderKey.PAYLOAD_ENCODING)
            if not encoding:
                if message.payload is None:
                    encoding = Encoding.NONE
                elif isinstance(message.payload, bytes) or isinstance(message.payload, bytearray):
                    encoding = Encoding.BYTES
                else:
                    encoding = Encoding.FOBS
                    message.payload = fobs.dumps(message.payload)
                message.set_header(MessageHeaderKey.PAYLOAD_ENCODING, encoding)
            message.set_header(MessageHeaderKey.SEND_TIME, time.time())
            self.communicator.send(to_endpoint, Cell.APP_ID, message)
        except:
            self.logger.error(f"failed to send message to {to_endpoint.name}")
            traceback.print_exc()
            err = "CommError"
        return err

    def _send_target_messages(
            self,
            target_msgs: Dict[str, TargetMessage],
    ) -> Dict[str, bool]:
        if not self.running:
            raise RuntimeError("Messenger is not running")

        sent = {}
        reachable_targets = {}  # target fqcn => endpoint
        for t in target_msgs.keys():
            ep = self._find_endpoint(t)
            if ep:
                reachable_targets[t] = ep
            else:
                self.logger.error(f"{self.my_info.fqcn}: no path to cell '{t}'")
                sent[t] = False

        for t, ep in reachable_targets.items():
            tm = target_msgs[t]
            req = Message(
                headers=copy.copy(tm.message.headers),
                payload=tm.message.payload)

            req.add_headers({
                MessageHeaderKey.CHANNEL: tm.channel,
                MessageHeaderKey.TOPIC: tm.topic,
                MessageHeaderKey.ORIGIN: self.my_info.fqcn,
                MessageHeaderKey.FROM_CELL: self.my_info.fqcn,
                MessageHeaderKey.MSG_TYPE: MessageType.REQ,
                MessageHeaderKey.ROUTE: [self.my_info.fqcn],
                MessageHeaderKey.DESTINATION: t,
                MessageHeaderKey.TO_CELL: ep.name
            })

            # is this a direct path?
            ti = FqcnInfo(t)
            if t != ep.name and not same_family(ti, self.my_info):
                # Not a direct path since the destination and the next leg are not the same
                if self.my_info.is_on_server:
                    # server side - try to create a listener and let the peer know the endpoint
                    listener = self._create_external_listener(True)
                    if listener:
                        conn_url = listener.get_connection_url()
                        req.set_header(MessageHeaderKey.CONN_URL, conn_url)
            err = self._send_to_endpoint(ep, req)
            sent[t] = not err
        return sent

    def _send_to_targets(
            self,
            channel: str,
            topic: str,
            targets: Union[str, List[str]],
            message: Message,
    ) -> Dict[str, bool]:
        if isinstance(targets, str):
            targets = [targets]
        target_msgs = {}
        for t in targets:
            target_msgs[t] = TargetMessage(t, channel, topic, message)
        return self._send_target_messages(target_msgs)

    def send_request(
            self,
            channel: str,
            topic: str,
            target: str,
            request: Message,
            timeout=None) -> Message:
        self.logger.debug(f"{self.my_info.fqcn}: sending request {channel}:{topic} to {target}")
        result = self.broadcast_request(channel, topic, [target], request, timeout)
        assert isinstance(result, dict)
        return result.get(target)

    def broadcast_multi_requests(
            self,
            target_msgs: Dict[str, TargetMessage],
            timeout=None
    ) -> Dict[str, Message]:
        targets = [t for t in target_msgs]
        waiter = _Waiter(targets)
        self.waiters[waiter.id] = waiter
        now = time.time()
        if not timeout:
            timeout = self.max_timeout

        try:
            for _, tm in target_msgs.items():
                request = tm.message
                request.add_headers(
                    {
                        MessageHeaderKey.REQ_ID: waiter.id,
                        MessageHeaderKey.REPLY_EXPECTED: True,
                        MessageHeaderKey.WAIT_UNTIL: time.time() + timeout
                    }
                )
            status = self._send_target_messages(target_msgs)
            send_count = 0
            err_reply = make_reply(ReturnCode.COMM_ERROR)
            timeout_reply = make_reply(ReturnCode.TIMEOUT)
            for t, sent in status.items():
                if sent:
                    send_count += 1
                    waiter.replies[t] = timeout_reply
                else:
                    waiter.replies[t] = err_reply
                    waiter.reply_time[t] = now

            if send_count > 0:
                self.num_sar_reqs += 1
                num_reqs = len(self.waiters)
                if self.req_hw < num_reqs:
                    self.req_hw = num_reqs

                # wait for reply
                self.logger.debug(f"{self.my_info.fqcn}: set up waiter {waiter.id} to wait for {timeout} secs")
                if not waiter.wait(timeout=timeout):
                    # timeout
                    self.logger.info(f"{self.my_info.fqcn}: timeout on REQ {waiter.id} after {timeout} secs")
                    with self.stats_lock:
                        self.num_timeout_reqs += 1
        except BaseException as ex:
            raise ex
        finally:
            self.waiters.pop(waiter.id, None)
            self.logger.debug(f"released waiter on REQ {waiter.id}")
        return waiter.replies

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
        if isinstance(targets, str):
            targets = [targets]
        target_msgs = {}
        for t in targets:
            target_msgs[t] = TargetMessage(t, channel, topic, request)
        return self.broadcast_multi_requests(target_msgs, timeout)

    def fire_and_forget(
            self,
            channel: str,
            topic: str,
            targets: Union[str, List[str]],
            message: Message) -> Dict[str, bool]:
        """
        Send a message over a channel to specified destination cell(s), and do not wait for replies.

        Args:
            channel: channel for the message
            topic: topic of the message
            targets: one or more destination cell IDs. None means all.
            message: message to be sent

        Returns: None

        """
        message.add_headers(
            {
                MessageHeaderKey.REPLY_EXPECTED: False
            }
        )
        return self._send_to_targets(channel, topic, targets, message)

    def queue_message(
            self,
            channel: str,
            topic: str,
            targets: Union[str, List[str]],
            message: Message):
        if isinstance(targets, str):
            targets = [targets]

        with self.bulk_lock:
            for t in targets:
                sender = self.bulk_senders.get(t)
                if not sender:
                    sender = _BulkSender(
                        cell=self,
                        target=t,
                        interval=self.bulk_send_interval,
                        max_queue_size=self.max_bulk_size
                    )
                    self.bulk_senders[t] = sender
                sender.queue_message(
                    channel=channel,
                    topic=topic,
                    message=message
                )

    def _process_bulk_message(
            self,
            request: Message):
        target_msgs = request.payload
        assert isinstance(target_msgs, list)
        for tm in target_msgs:
            assert isinstance(tm, TargetMessage)
            req = tm.message
            req.add_headers(request.headers)
            req.add_headers(
                {
                    MessageHeaderKey.TOPIC: tm.topic,
                    MessageHeaderKey.CHANNEL: tm.channel
                }
            )
            origin = request.get_header(MessageHeaderKey.ORIGIN, "")
            self._process_request(origin=origin, message=req)

    def fire_multi_requests_and_forget(
            self,
            target_msgs: Dict[str, TargetMessage]
    ) -> Dict[str, bool]:
        for _, tm in target_msgs.items():
            request = tm.message
            request.add_headers(
                {
                    MessageHeaderKey.REPLY_EXPECTED: False,
                }
            )
        return self._send_target_messages(target_msgs)

    def send_reply(
            self,
            reply: Message,
            to_cell: str,
            for_req_ids: List[str]
    ) -> str:
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

        Returns: an error message if any

        """
        reply.add_headers(
            {
                MessageHeaderKey.FROM_CELL: self.my_info.fqcn,
                MessageHeaderKey.ORIGIN: self.my_info.fqcn,
                MessageHeaderKey.ROUTE: [self.my_info.fqcn],
                MessageHeaderKey.DESTINATION: to_cell,
                MessageHeaderKey.REQ_ID: for_req_ids,
                MessageHeaderKey.MSG_TYPE: MessageType.REPLY,
            }
        )

        ep = self._find_endpoint(to_cell)
        if not ep:
            return "CommError"
        reply.set_header(MessageHeaderKey.TO_CELL, ep.name)
        return self._send_to_endpoint(ep, reply)

    def process_message(self, endpoint: Endpoint, app_id: int, message: Message):
        # this is the receiver callback
        try:
            self._process_received_msg(endpoint, message)
        except:
            traceback.print_exc()

    def _process_request(
            self,
            origin: str,
            message: Message) -> Union[None, Message]:
        self.logger.debug(f"{self.my_info.fqcn}: processing request")
        # this is a request for me - dispatch to the right CB
        channel = message.get_header(MessageHeaderKey.CHANNEL, "")
        topic = message.get_header(MessageHeaderKey.TOPIC, "")
        _cb = self.req_reg.find(channel, topic)
        if not _cb:
            self.logger.error(
                f"{self.my_info.fqcn}: no callback for request ({topic}@{channel}) from cell '{origin}'")
            return make_reply(ReturnCode.PROCESS_EXCEPTION, error="no callback")

        try:
            assert isinstance(_cb, _CB)
            self.logger.debug(f"{self.my_info.fqcn}: calling registered request CB")
            reply = _cb.cb(message, *_cb.args, **_cb.kwargs)
            if not reply:
                # the CB doesn't have anything to reply
                self.logger.debug("no reply is returned from the CB")
                return None

            if not isinstance(reply, Message):
                self.logger.error(
                    f"{self.my_info.fqcn}: bad result from request CB for topic {topic} on channel {channel}: "
                    f"expect Message but got {type(reply)}"
                )
                return make_reply(ReturnCode.PROCESS_EXCEPTION, error="bad cb result")
        except:
            traceback.print_exc()
            return make_reply(ReturnCode.PROCESS_EXCEPTION, error="cb exception")

        reply_expected = message.get_header(MessageHeaderKey.REPLY_EXPECTED, False)
        if not reply_expected:
            # this is fire and forget
            self.logger.debug(f"{self.my_info.fqcn}: don't send response - request expects no reply")
            return None

        wait_until = message.get_header(MessageHeaderKey.WAIT_UNTIL, None)
        if isinstance(wait_until, float) and time.time() > wait_until:
            # no need to reply since peer already gave up waiting by now
            self.logger.debug(f"{self.my_info.fqcn}: don't send response - reply is too late")
            return None

        # send the reply back
        if not reply.headers.get(MessageHeaderKey.RETURN_CODE):
            self.logger.debug(f"{self.my_info.fqcn}: added return code OK")
            reply.set_header(MessageHeaderKey.RETURN_CODE, ReturnCode.OK)
        return reply

    def _add_to_route(self, message: Message):
        route = message.get_header(MessageHeaderKey.ROUTE, None)
        if route:
            if not isinstance(route, list):
                self.logger.error(
                    format_log_message(self.my_info.fqcn, message,
                                       "bad route header: expect list but got {type(route)}"))
            else:
                route.append(self.my_info.fqcn)

    def _forward(self, endpoint: Endpoint, origin: str, destination: str, msg_type: str, message: Message):
        # not for me - need to forward it
        self.logger.debug(f"{self.my_info.fqcn}: forwarding for {origin} to {destination}")
        ep = self._find_endpoint(destination)
        if ep:
            self.logger.debug(f"{self.my_info.fqcn}: found next leg {ep.name}")
            message.add_headers({
                MessageHeaderKey.FROM_CELL: self.my_info.fqcn,
                MessageHeaderKey.TO_CELL: ep.name
            })
            self._add_to_route(message)
            err = self._send_to_endpoint(to_endpoint=ep, message=message)
            if not err:
                self.logger.debug(f"{self.my_info.fqcn}: forwarded successfully!")
                return
            else:
                self.logger.error(
                    format_log_message(self.my_info.fqcn, message, f"failed to forward {msg_type}: {err}")
                )
        else:
            # cannot find next leg endpoint
            self.logger.error(
                format_log_message(self.my_info.fqcn, message, f"cannot forward {msg_type}: no path")
            )

        if msg_type == MessageType.REQ:
            reply_expected = message.get_header(MessageHeaderKey.REPLY_EXPECTED, False)
            if not reply_expected:
                self.logger.debug(f"{self.my_info.fqcn}: can't forward: drop the message since reply is not expected")
                return

            wait_until = message.get_header(MessageHeaderKey.WAIT_UNTIL, None)
            if isinstance(wait_until, float) and time.time() > wait_until:
                # no need to reply since peer already gave up waiting by now
                self.logger.debug(f"{self.my_info.fqcn}: can't forward: drop the message since too late")
                return

            # tell the requester that message couldn't be delivered
            req_id = message.get_header(MessageHeaderKey.REQ_ID, "")
            reply = make_reply(ReturnCode.COMM_ERROR, error="cannot forward")
            reply.add_headers(
                {
                    MessageHeaderKey.ORIGINAL_HEADERS: message.headers,
                    MessageHeaderKey.FROM_CELL: self.my_info.fqcn,
                    MessageHeaderKey.TO_CELL: endpoint.name,
                    MessageHeaderKey.ORIGIN: self.my_info.fqcn,
                    MessageHeaderKey.DESTINATION: origin,
                    MessageHeaderKey.REQ_ID: [req_id],
                    MessageHeaderKey.MSG_TYPE: MessageType.RETURN,
                    MessageHeaderKey.ROUTE: [self.my_info.fqcn]
                }
            )
            self._send_to_endpoint(endpoint, reply)
            self.logger.debug(f"{self.my_info.fqcn}: sent RETURN message back to {endpoint.name}")
        else:
            # msg_type is either RETURN or REPLY - drop it.
            self.logger.warning(format_log_message(self.my_info.fqcn, message, "dropped forwarded message"))

    def _process_reply(self, origin: str, message: Message, msg_type: str):
        self.logger.debug(f"{self.my_info.fqcn}: processing reply from {origin} for type {msg_type}")
        req_ids = message.get_header(MessageHeaderKey.REQ_ID)
        if not req_ids:
            raise RuntimeError(format_log_message(self.my_info.fqcn, message, "reply does not have REQ_ID header"))

        if isinstance(req_ids, str):
            req_ids = [req_ids]

        if not isinstance(req_ids, list):
            raise RuntimeError(format_log_message(self.my_info.fqcn, message,
                                                  f"REQ_ID must be list of ids but got {type(req_ids)}"))

        req_dest = origin
        if msg_type == MessageType.RETURN:
            original_headers = message.get_header(MessageHeaderKey.ORIGINAL_HEADERS, None)
            if not original_headers:
                raise RuntimeError(format_log_message(
                    self.my_info.fqcn, message, "missing ORIGINAL_HEADERS in returned message!"))
            req_dest = original_headers.get(MessageHeaderKey.DESTINATION, None)
            if not req_dest:
                raise RuntimeError(format_log_message(
                    self.my_info.fqcn, message, "missing DESTINATION header in original headers"))

        for rid in req_ids:
            waiter = self.waiters.get(rid, None)
            if waiter:
                assert isinstance(waiter, _Waiter)
                if req_dest not in waiter.replies:
                    self.logger.error(format_log_message(
                        self.my_info.fqcn, message, f"unexpected reply for {rid} from {req_dest}"))
                    return
                waiter.replies[req_dest] = message
                waiter.reply_time[req_dest] = time.time()

                # all targets replied?
                all_targets_replied = True
                for t, _ in waiter.replies.items():
                    if not waiter.reply_time.get(t):
                        all_targets_replied = False
                        break

                if all_targets_replied:
                    self.logger.debug(
                        format_log_message(
                            self.my_info.fqcn, message,
                            f"trigger waiter - replies received from {len(waiter.replies)} targets for req {rid}"))
                    waiter.set()  # trigger the waiting requests!
                else:
                    self.logger.debug(
                        format_log_message(
                            self.my_info.fqcn, message,
                            f"keep waiting - replies not received from {len(waiter.replies)} targets for req {rid}"))
            else:
                self.logger.debug(
                    format_log_message(
                        self.my_info.fqcn, message,
                        f"no waiter for req {rid} - the reply is too late"))

    def _process_received_msg(self, endpoint: Endpoint, message: Message):
        self.logger.debug(f"{self.my_info.fqcn}: received message: {message.headers}")
        msg_type = message.get_header(MessageHeaderKey.MSG_TYPE)
        if not msg_type:
            raise RuntimeError(format_log_message(
                self.my_info.fqcn, message, "missing MSG_TYPE in received message"))

        origin = message.get_header(MessageHeaderKey.ORIGIN)
        if not origin:
            raise RuntimeError(format_log_message(
                self.my_info.fqcn, message, "missing ORIGIN header in received message"))

        # is this msg for me?
        destination = message.get_header(MessageHeaderKey.DESTINATION)
        if not destination:
            raise RuntimeError(format_log_message(
                self.my_info.fqcn, message, "missing DESTINATION header in received message"))

        if destination != self.my_info.fqcn:
            # not for me - need to forward it
            self._forward(endpoint, origin, destination, msg_type, message)
            return

        # this message is for me
        self._add_to_route(message)

        # handle content type
        payload_encoding = message.get_header(MessageHeaderKey.PAYLOAD_ENCODING)
        if not payload_encoding:
            self.logger.warning(format_log_message(
                self.my_info.fqcn, message, "missing payload_encoding header received message"))

        if payload_encoding == Encoding.FOBS:
            message.payload = fobs.loads(message.payload)
        elif payload_encoding == Encoding.NONE:
            message.payload = None
        else:
            # assume to be bytes
            pass

        # handle ad-hoc
        my_conn_url = None
        if msg_type in [MessageType.REQ, MessageType.REPLY]:
            from_cell = message.get_header(MessageHeaderKey.FROM_CELL)
            oi = FqcnInfo(origin)
            if from_cell != origin and not same_family(oi, self.my_info):
                # this is a forwarded message, so no direct path from the origin to me
                conn_url = message.get_header(MessageHeaderKey.CONN_URL)
                if conn_url:
                    # the origin already has a listener
                    # create an ad-hoc connector to connect to the origin cell
                    self.logger.debug(f"{self.my_info.fqcn}: creating adhoc connector to {origin} at {conn_url}")
                    self._add_adhoc_connector(origin, conn_url)
                elif msg_type == MessageType.REQ:
                    # see whether we can offer a listener
                    if not oi.is_on_server:
                        self.logger.debug(f"{self.my_info.fqcn}: trying to offer ad-hoc listener to {origin}")
                        listener = self._create_external_listener(True)
                        if listener:
                            my_conn_url = listener.get_connection_url()

        if msg_type == MessageType.REQ:
            # this is a request for me - dispatch to the right CB
            reply = self._process_request(origin, message)
            if reply:
                req_id = message.get_header(MessageHeaderKey.REQ_ID, "")
                reply.add_headers(
                    {
                        MessageHeaderKey.FROM_CELL: self.my_info.fqcn,
                        MessageHeaderKey.ORIGIN: self.my_info.fqcn,
                        MessageHeaderKey.DESTINATION: origin,
                        MessageHeaderKey.TO_CELL: endpoint.name,
                        MessageHeaderKey.REQ_ID: req_id,
                        MessageHeaderKey.MSG_TYPE: MessageType.REPLY,
                        MessageHeaderKey.ROUTE: [self.my_info.fqcn]
                    }
                )

                if my_conn_url:
                    reply.set_header(MessageHeaderKey.CONN_URL, my_conn_url)
                self.logger.debug(f"{self.my_info.fqcn}: sending reply back to {endpoint.name}")
                self.logger.debug(f"Reply message: {reply.headers}")
                self._send_to_endpoint(endpoint, reply)
            else:
                self.logger.debug(f"{self.my_info.fqcn}: no reply to send!")
            return

        # the message is either a reply or a return for a previous request: handle replies
        self._process_reply(origin, message, msg_type)

    def _check_bulk(self):
        while not self.asked_to_stop:
            with self.bulk_lock:
                for _, sender in self.bulk_senders.items():
                    sender.send(False)
            time.sleep(self.bulk_check_interval)

        # force everything to be flushed
        with self.bulk_lock:
            for _, sender in self.bulk_senders.items():
                sender.send(True)

    def state_change(self, endpoint: Endpoint):
        self.logger.debug(f"========= {self.my_info.fqcn}: EP {endpoint.name} state changed to {endpoint.state}")
        fqcn = endpoint.name
        if endpoint.state == EndpointState.READY:
            # create the CellAgent for this endpoint
            agent = self.agents.get(fqcn)
            if not agent:
                agent = CellAgent(fqcn, endpoint)
                with self.agent_lock:
                    self.agents[fqcn] = agent
                self.logger.debug(f"{self.my_info.fqcn}: created CellAgent for {fqcn}")
            else:
                self.logger.debug(f"{self.my_info.fqcn}: found existing CellAgent for {fqcn} - shouldn't happen")
                agent.endpoint = endpoint

            if self.cell_connected_cb is not None:
                try:
                    self.logger.debug(f"{self.my_info.fqcn}: calling cell_connected_cb")
                    self.cell_connected_cb(
                        agent,
                        *self.cell_connected_cb_args,
                        **self.cell_connected_cb_kwargs
                    )
                except:
                    self.logger.error(f"{self.my_info.fqcn}: exception in cell_connected_cb")
                    traceback.print_exc()

        elif endpoint.state in [EndpointState.CLOSING, EndpointState.DISCONNECTED, EndpointState.IDLE]:
            # remove this agent
            with self.agent_lock:
                agent = self.agents.pop(fqcn, None)
                self.logger.debug(f"{self.my_info.fqcn}: removed CellAgent {fqcn}")
            if agent and self.cell_disconnected_cb is not None:
                try:
                    self.logger.debug(f"{self.my_info.fqcn}: calling cell_disconnected_cb")
                    self.cell_disconnected_cb(
                        agent,
                        *self.cell_disconnected_cb_args,
                        **self.cell_disconnected_cb_kwargs
                    )
                except:
                    self.logger.error(f"{self.my_info.fqcn}: exception in cell_disconnected_cb")
                    traceback.print_exc()

    def get_sub_cell_names(self) -> (List[str], List[str]):
        """
        Get call FQCNs of all subs, which are children or top-level client cells (if my cell is server).

        Returns: fqcns of child cells, fqcns of top-level client cells
        """
        children = []
        clients = []
        with self.agent_lock:
            for fqcn, agent in self.agents.items():
                if FQCN.is_parent(self.my_info.fqcn, fqcn):
                    children.append(fqcn)
                elif self.my_info.is_root and self.my_info.is_on_server:
                    # see whether the agent is a client cell
                    if agent.info.is_root and not agent.info.is_on_server:
                        clients.append(fqcn)
            return children, clients
