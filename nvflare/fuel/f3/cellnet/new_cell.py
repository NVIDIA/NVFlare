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
import uuid
from typing import Dict, List, Union

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, MessageType
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.stream_cell import StreamCell
from nvflare.fuel.f3.streaming.stream_const import StreamHeaderKey
from nvflare.fuel.f3.streaming.stream_types import StreamFuture
from nvflare.private.defs import CellChannel


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
        req_id = request.get_header(MessageHeaderKey.REQ_ID)
        # self.logger.info(f"=============================> Receiving: {channel}, {topic}, {len(result)} bytes.")
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

    def start(self):
        self.core_cell.start()

    def stop(self):
        self.core_cell.stop()

    def is_cell_connected(self, target_fqcn):
        return self.core_cell.is_cell_connected(target_fqcn)

    def broadcast_multi_requests(self, *args, **kwargs):
        return self.core_cell.broadcast_multi_requests(*args, **kwargs)

    def get_root_url_for_child(self):
        return self.core_cell.get_root_url_for_child()

    def fire_and_forget(
        self, channel: str, topic: str, targets: Union[str, List[str]], message: Message, optional=False
    ) -> Dict[str, str]:
        return self.core_cell.fire_and_forget(
            channel=channel, topic=topic, targets=targets, message=message, optional=optional
        )

    def get_fqcn(self):
        return self.core_cell.get_fqcn()

    def get_internal_listener_url(self) -> Union[None, str]:
        """Get the cell's internal listener url.

        This method should only be used for cells that need to have child cells.
        The url returned is to be passed to child of this cell to create connection

        Returns: url for child cells to connect

        """
        return self.core_cell.get_internal_listener_url()

    def send_request(self, channel, target, topic, request, timeout=None, optional=False):
        if channel != CellChannel.SERVER_COMMAND:
            return self.core_cell.send_request(
                channel=channel, target=target, topic=topic, request=request, timeout=timeout, optional=optional
            )

        req_id = str(uuid.uuid4())

        request.add_headers({StreamHeaderKey.STREAM_REQ_ID: req_id})
        print(f"Sending: {len(request.payload)} bytes")
        future = self.send_blob(channel, topic, target, request)  # StreamCell API

        # this future can be used to check sending progress, but not for checking return blob
        waiter = threading.Event()
        self.requests_dict[req_id] = [waiter, None]

        if not waiter.wait(timeout):
            self.requests_dict.pop(req_id)
            return Message()  # or raise TimeoutError(f"request to {channel=} {topic=} timeout")
        _, result = self.requests_dict.pop(req_id)
        return result

    def _process_reply(self, future: StreamFuture):
        req_id = future.headers.get(StreamHeaderKey.STREAM_REQ_ID, -1)
        if req_id not in self.requests_dict:
            return False  # response coming back too late.  The req_id was popped out during timeout
        headers = future.headers
        response_blob = future.result()
        waiter, _ = self.requests_dict.get(req_id, [None, None])
        self.requests_dict[req_id] = [waiter, Message(headers, response_blob)]
        waiter.set()

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
        if channel != CellChannel.SERVER_COMMAND:
            self.core_cell.register_request_cb(channel, topic, cb, *args, **kwargs)
        else:
            adapter = Adapter(cb, self.core_cell.my_info, self)
            self.register_blob_cb(
                channel, topic, adapter.call, *args, **kwargs
            )  # this cb is defined by Yuhong.  The cb expects the message, not a future.
