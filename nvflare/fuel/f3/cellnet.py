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
from .message import Message
from .driver import DriverSpec
from .endpoint import Endpoint
from .communicator import Communicator
from .receiver import Receiver


class TargetCellUnreachable(Exception):
    pass


class MessageHeaderKey:

    MSG_TYPE = "cellnet.msg_type"
    REQ_ID = "cellnet.req_id"
    REPLY_EXPECTED = "cellnet.reply_expected"
    TOPIC = "cellnet.topic"
    WAIT_UNTIL = "cellnet.wait_until"
    FROM_CELL = "cellnet.from"
    TO_CELL = "cellnet.to"
    CHANNEL = "cellnet.channel"


class MessageType:

    REQ = "req"
    REPLY = "reply"


class CellAgent:
    """
    A CellAgent represents a cell in another cell.
    """

    def __init__(
            self,
            fqcn: str,
            props: dict,
            endpoint: Endpoint
    ):
        """

        Args:
            fqcn: FQCN of the cell represented
        """
        self.fqcn = fqcn
        self.props = props
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
        self.targets = {}    # target_id => reply
        for t in targets:
            self.targets[t] = None
        self.send_time = time.time()
        self.id = str(uuid.uuid4())


class FQCN:

    SEPARATOR = "."

    @staticmethod
    def split(fqcn: str) -> List[str]:
        return fqcn.split(FQCN.SEPARATOR)

    @staticmethod
    def join(path: List[str]) -> str:
        return FQCN.SEPARATOR.join(path)


class Cell(Receiver):

    APP_ID = 1

    def __init__(
            self,
            fqcn: str,
            url: str,
            props: dict,
            max_timeout=3600,
    ):
        """

        Args:
            fqcn: the Cell's FQCN (Fully Qualified Cell Name)

        FQCN is the name of all ancestor names, concatenated with colons.

        Example:
            server.J12345       (the cell for job J12345 on the server)
            server              (the root cell of server)
            nih_1.J12345     (the cell for job J12345 on client_1's site)
            client_1.J12345.R0  (the cell for rank R0 of J12345 on client_1 site)
            client_1            (he root cell of client_1)

        """
        self.my_fqcn = fqcn
        self.props = props
        self.agents = {}  # cell_fqcn => CellAgent

        self.communicator = Communicator(
            local_endpoint=Endpoint(
                name=fqcn,
                url=url,
                properties=props)
        )
        self.communicator.register_receiver(endpoint=None, app=self.APP_ID, receiver=self)
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
        self.waiter_lock = threading.Lock()
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

    def add_connector(self, connector: DriverSpec):
        """
        Add a connector to be used to connect to another cell.

        Args:
            connector: the connector

        Returns:

        """
        self.communicator.add_connector(connector)

    def add_listener(self, listener: DriverSpec):
        """
        Add a listener.

        Args:
            listener: listener to be added

        Returns:

        """
        self.communicator.add_listener(listener)

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
        if agent and agent.endpoint:
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
        path = FQCN.split(self.my_fqcn)
        if len(path) > 1:
            return self._try_path(path[:-1])
        else:
            return None

    def find_endpoint(self, target_fqcn: str) -> Union[None, Endpoint]:
        if target_fqcn == self.my_fqcn:
            # sending request to myself? Not allowed!
            return None

        agent = self.agents.get(target_fqcn, None)
        if agent and agent.endpoint:
            # there is a direct path to the target call
            return agent.endpoint

        ep = self._find_ep(target_fqcn)
        if ep:
            # remember this so we don't need to repeat again
            if not agent:
                agent = CellAgent(fqcn=target_fqcn, props={}, endpoint=ep)
            self.agents[target_fqcn] = agent
        return ep

    def _send_to_targets(
            self,
            channel: str,
            topic: str,
            targets: Union[str, List[str]],
            request: Message,
            headers: dict
    ):
        if not self.running:
            raise RuntimeError("Messenger is not running")

        if isinstance(targets, str):
            # turn it to list
            targets = [targets]

        reachable_targets = {}  # target fqcn => endpoint
        for t in targets:
            ep = self.find_endpoint(t)
            if not ep:
                raise TargetCellUnreachable(f"no path to cell '{t}'")
            reachable_targets[t] = ep

        # the msg object must have the method set_header and get_header
        request.add_headers({
            MessageHeaderKey.CHANNEL: channel,
            MessageHeaderKey.TOPIC: topic,
            MessageHeaderKey.FROM_CELL: self.my_fqcn,
            MessageHeaderKey.MSG_TYPE: MessageType.REQ,
            }
        )

        request.add_headers(headers)

        for t, ep in reachable_targets.items():
            if len(reachable_targets) > 1:
                req = Message(
                    headers=copy.deepcopy(request.headers),
                    payload=request.payload
                )
            else:
                req = request
            req.set_header(MessageHeaderKey.TO_CELL, t)

            self.communicator.send(
                endpoint=ep,
                app=Cell.APP_ID,
                message=req
            )

    def send_and_receive(
            self,
            channel: str,
            topic: str,
            targets: Union[str, List[str]],
            request: Message,
            timeout=None) -> Dict[str, Union[None, Message]]:
        """
        Send a message over a channel to specified destination cell, and wait for reply

        Args:
            channel: channel for the message
            targets: FQCN of the destination cell(s)
            request: message to be sent
            timeout: how long to wait for replies

        Returns: a dict of: cell_id => reply message

        """
        waiter = _Waiter(targets)
        with self.waiter_lock:
            self._send_to_targets(
                channel, topic, targets, request,
                {
                    MessageHeaderKey.REQ_ID: waiter.id,
                    MessageHeaderKey.REPLY_EXPECTED: True,
                    MessageHeaderKey.WAIT_UNTIL: time.time() + timeout
                }
            )

            self.waiters[waiter.id] = waiter
            self.num_sar_reqs += 1
            num_reqs = len(self.waiters)
            if self.req_hw < num_reqs:
                self.req_hw = num_reqs

        # wait for reply
        if not waiter.wait(timeout=timeout):
            # timeout
            with self.stats_lock:
                self.num_timeout_reqs += 1

            with self.waiter_lock:
                self.waiters.pop(waiter.id, None)
        return waiter.targets

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
        reply.add_headers(
            {
                MessageHeaderKey.FROM_CELL: self.my_fqcn,
                MessageHeaderKey.TO_CELL: to_cell,
                MessageHeaderKey.REQ_ID: for_req_ids,
                MessageHeaderKey.MSG_TYPE: MessageType.REPLY,
            }
        )

        ep = self.find_endpoint(to_cell)
        if not ep:
            raise TargetCellUnreachable(f"no path to cell {to_cell}")
        self.communicator.send(ep, Cell.APP_ID, reply)

    def process(self, endpoint: Endpoint, app: int, message: Message):
        # this is the receiver callback
        try:
            self._process_received_msg(endpoint, app, message)
        except:
            traceback.print_exc()

    def _process_received_msg(self, endpoint: Endpoint, app: int, message: Message):
        # is this msg for me?
        to_cell = message.get_header(MessageHeaderKey.TO_CELL)
        if not to_cell:
            raise RuntimeError("Proto Error: missing TO_CELL in received message")

        if to_cell != self.my_fqcn:
            # not for me - need to forward it
            ep = self.find_endpoint(to_cell)
            if not ep:
                raise TargetCellUnreachable(f"Cannot forward to cell '{to_cell}' - no path")
            self.communicator.send(endpoint=ep, app=Cell.APP_ID, message=message)
            return

        from_cell = message.get_header(MessageHeaderKey.FROM_CELL)
        if not from_cell:
            raise RuntimeError("Proto Error: missing FROM_CELL in received message")

        # this message is for me
        msg_type = message.get_header(MessageHeaderKey.MSG_TYPE)
        if not msg_type:
            raise RuntimeError("Proto Error: missing MSG_TYPE in received message")

        if msg_type == MessageType.REQ:
            # this is a request for me - dispatch to the right CB
            channel = message.get_header(MessageHeaderKey.CHANNEL, "")
            topic = message.get_header(MessageHeaderKey.TOPIC, "")
            req_id = message.get_header(MessageHeaderKey.REQ_ID, "")
            _cb = self.req_reg.find(channel, topic)
            if not _cb:
                raise RuntimeError(
                    f"No callback for request ({topic}@{channel}) from cell '{from_cell}'")

            assert isinstance(_cb, _CB)
            reply = _cb.cb(self, channel, topic, message, *_cb.args, **_cb.kwargs)
            if not reply:
                # the CB doesn't have anything to reply
                print("No reply to send")
                return

            reply_expected = message.get_header(MessageHeaderKey.REPLY_EXPECTED, False)
            if not reply_expected:
                print("do not send reply - reply not expected")
                return

            wait_until = message.get_header(MessageHeaderKey.WAIT_UNTIL, None)
            if isinstance(wait_until, float) and time.time() > wait_until:
                # no need to reply since peer already gave up
                print("do not send reply - wait timeout")
                return

            # send the reply back
            assert isinstance(reply, Message)
            reply.add_headers(
                {
                    MessageHeaderKey.FROM_CELL: self.my_fqcn,
                    MessageHeaderKey.TO_CELL: from_cell,
                    MessageHeaderKey.REQ_ID: [req_id],
                    MessageHeaderKey.MSG_TYPE: MessageType.REPLY,
                }
            )
            self.communicator.send(endpoint, Cell.APP_ID, reply)
            return

        # handle replies
        req_ids = message.get_header(MessageHeaderKey.REQ_ID)
        if not req_ids:
            raise RuntimeError("Proto Error: received reply does not have REQ_ID header")

        if not isinstance(req_ids, list):
            raise RuntimeError(f"Proto Error: REQ_ID must be list of ids but got {type(req_ids)}")

        with self.waiter_lock:
            for rid in req_ids:
                waiter = self.waiters.pop(rid, None)
                if waiter:
                    assert isinstance(waiter, _Waiter)
                    if from_cell not in waiter.targets:
                        raise RuntimeError(f"received reply for {rid} from unexpected cell {from_cell}")
                    waiter.targets[from_cell] = message

                    # all targets replied?
                    all_targets_replied = True
                    for _, reply in waiter.targets.items():
                        if not reply:
                            all_targets_replied = False
                            break

                    if all_targets_replied:
                        waiter.set()  # trigger the waiting requests!


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
