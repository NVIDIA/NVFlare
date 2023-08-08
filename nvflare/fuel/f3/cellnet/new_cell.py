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

import logging
import threading
import time
import uuid
from typing import Dict, List, Union

from nvflare.apis.fl_constant import ServerCommandNames
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, MessageType, ReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.stream_cell import StreamCell
from nvflare.fuel.f3.streaming.stream_const import StreamHeaderKey
from nvflare.fuel.f3.streaming.stream_types import StreamFuture
from nvflare.private.defs import CellChannel


class SimpleWaiter(threading.Event):
    def __init__(self, req_id, timeout, result):
        super().__init__()
        self.req_id = req_id
        self.timeout = timeout
        self.result = result
        self.sending_complete = False
        self.in_receiving = False


class Adapter:
    def __init__(self, cb, my_info, nice_cell):
        self.cb = cb
        self.my_info = my_info
        self.nice_cell = nice_cell
        self.logger = logging.getLogger(self.__class__.__name__)

    def call(self, future):  # this will be called by StreamCell upon receiving the first byte of blob
        headers = future.headers
        stream_req_id = headers.get(StreamHeaderKey.STREAM_REQ_ID, "")
        origin = headers.get(MessageHeaderKey.ORIGIN, None)
        result = future.result()
        request = Message(headers, result)
        channel = request.get_header(StreamHeaderKey.CHANNEL)
        request.set_header(MessageHeaderKey.CHANNEL, channel)
        topic = request.get_header(StreamHeaderKey.TOPIC)
        request.set_header(MessageHeaderKey.TOPIC, topic)
        req_id = request.get_header(MessageHeaderKey.REQ_ID, "")
        response = self.cb(request)
        response.add_headers(
            {
                MessageHeaderKey.REQ_ID: req_id,
                MessageHeaderKey.MSG_TYPE: MessageType.REPLY,
                StreamHeaderKey.STREAM_REQ_ID: stream_req_id,
            }
        )
        messagesend_future = self.nice_cell.send_blob(channel, topic, origin, response)


class NewCell(StreamCell):
    def __init__(self, *args, **kwargs):
        self.core_cell = Cell(*args, **kwargs)
        super().__init__(self.core_cell)
        self.requests_dict = dict()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.register_blob_cb("*", "*", self._process_reply)  # this should be one-time registration

    def __getattr__(self, func):
        def method(*args, **kwargs):
            if not hasattr(self.core_cell, func):
                raise AttributeError(f"'{func}' not in core_cell.")
            return getattr(self.core_cell, func)(*args, **kwargs)

        return method

    def fire_and_forget(
        self, channel: str, topic: str, targets: Union[str, List[str]], message: Message, optional=False
    ) -> Dict[str, str]:
        """
        Send a message over a channel to specified destination cell(s), and do not wait for replies.

        Args:
            channel: channel for the message
            topic: topic of the message
            targets: one or more destination cell IDs. None means all.
            message: message to be sent
            optional: whether the message is optional

        Returns: None

        """
        if channel == CellChannel.SERVER_COMMAND and topic == ServerCommandNames.HANDLE_DEAD_JOB:
            if isinstance(targets, list):
                for target in targets:
                    self.send_blob(channel=channel, topic=topic, target=target, message=message)
            else:
                self.send_blob(channel=channel, topic=topic, target=targets, message=message)
        else:
            self.core_cell.fire_and_forget(
                channel=channel, topic=topic, targets=targets, message=message, optional=optional
            )

    def _get_result(self, req_id):
        waiter = self.requests_dict.pop(req_id)
        return waiter.result

    def send_request(self, channel, target, topic, request, timeout, optional=False):
        self.logger.debug(f"send_request: {channel=}, {target=}, {topic=}, {timeout=}")
        if channel != CellChannel.SERVER_COMMAND:
            return self.core_cell.send_request(
                channel=channel, target=target, topic=topic, request=request, timeout=timeout, optional=optional
            )

        req_id = str(uuid.uuid4())
        request.add_headers({StreamHeaderKey.STREAM_REQ_ID: req_id})

        # this future can be used to check sending progress, but not for checking return blob
        future = self.send_blob(channel, topic, target, request)

        waiter = SimpleWaiter(req_id=req_id, timeout=timeout, result=make_reply(ReturnCode.TIMEOUT))
        self.requests_dict[req_id] = waiter
        """
        when inside the while loop, following things could happen
        1. future is done
           we should exit this loop and start wait for reply

        2. waiter is set
           reply is ready and we can exit this loop

        3. future is making progress
           stay in the loop, reset timer

        4. no progress, not timeout yet
           stay in the loop

        5. no progress, timeout
           exit

        """
        self.logger.debug(f"{req_id=}: entering sending loop {timeout=}")
        last_progress = 0
        last_update_time = time.time()
        while True:
            if future.done():
                self.logger.debug(f"{req_id=}: sending complete")
                waiter.sending_complete = True
                break
            if waiter.is_set():
                self.logger.debug(f"{req_id=}: reply ready")
                return self._get_result(req_id)
            time.sleep(1.0)
            current_time = time.time()
            current_progress = future.get_progress()
            if current_progress == last_progress:
                if current_time - last_update_time > timeout:
                    self.logger.debug(f"{req_id=}: sending timeout")
                    future.cancel()
                    return self._get_result(req_id)
                else:
                    self.logger.debug(f"{req_id=}: no sending progress before timeout")
            else:
                self.logger.debug(f"{req_id=}: sending progress {current_progress=}")
                last_progress = current_progress
                last_update_time = current_time

        if not waiter.sending_complete:
            # should not be here
            self.logger.error(f"{req_id=}: unknown waiter state")
            return make_reply(ReturnCode.COMM_ERROR)

        # Now enter the remote process wait loop
        self.logger.debug(f"{req_id=}: entering remote process wait {timeout=}")
        while True:
            if waiter.is_set():
                self.logger.debug(f"{req_id=}: reply ready")
                return self._get_result(req_id)
            time.sleep(1.0)
            if waiter.in_receiving:  # roughly 1 second grace period
                self.logger.debug(f"{req_id=}: in receiving")
                break
            if time.time() - last_update_time > timeout:
                self.logger.debug(f"{req_id=}: remote processing timeout")
                return self._get_result(req_id)

        # Reaching here only because the in_receiving is True (thus inside receiving call back)
        # Wait here until the wait is set (only done by receiving call back)
        self.logger.debug(f"{req_id=}: entering preocessing reply wait")
        # the following may wait forever
        _ = waiter.wait()
        self.logger.debug(f"{req_id=}: waiter set by processing reply")
        return self._get_result(req_id)

    def _process_reply(self, future: StreamFuture):
        headers = future.headers
        req_id = headers.get(StreamHeaderKey.STREAM_REQ_ID, -1)
        try:
            waiter = self.requests_dict[req_id]
        except KeyError as e:
            self.logger.warning(f"Receiving unknown {req_id=}, discarded")
            return
        waiter.in_receiving = True
        last_progress = 0
        last_update_time = time.time()
        timeout = waiter.timeout
        self.logger.debug(f"{req_id=}: entering receiving loop {timeout=}")
        while True:
            if future.done():
                self.logger.debug(f"{req_id=}: receiving complete")
                waiter.result = Message(headers, future.result())
                waiter.set()
                break
            time.sleep(1.0)
            current_time = time.time()
            current_progress = future.get_progress()
            if current_progress == last_progress:
                if current_time - last_update_time > timeout:
                    self.logger.debug(f"{req_id=}: receiving timeout")
                    future.cancel()
                    waiter.receiving_timeout = True
                    waiter.set()
                    break
                else:
                    self.logger.debug(f"{req_id=}: no receiving progress before timeout")
            else:
                self.logger.debug(f"{req_id=}: receiving progress {current_progress=}")
                last_progress = current_progress
                last_update_time = current_time

    def register_request_cb(self, channel: str, topic: str, cb, *args, **kwargs):
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
        if not callable(cb):
            raise ValueError(f"specified request_cb {type(cb)} is not callable")
        if channel == CellChannel.SERVER_COMMAND and topic in [
            "*",
            ServerCommandNames.GET_TASK,
            ServerCommandNames.SUBMIT_UPDATE,
        ]:
            self.logger.debug(f"Register blob CB for {channel=}, {topic=}")
            adapter = Adapter(cb, self.core_cell.my_info, self)
            self.register_blob_cb(channel, topic, adapter.call, *args, **kwargs)
        else:
            self.logger.debug(f"Register regular CB for {channel=}, {topic=}")
            self.core_cell.register_request_cb(channel, topic, cb, *args, **kwargs)
