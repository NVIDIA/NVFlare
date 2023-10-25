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
import time
from typing import Any, List, Optional, Tuple

from nvflare.fuel.utils.pipe.pipe import Message, Pipe
from nvflare.fuel.utils.pipe.pipe_handler import PipeHandler, Topic


class DataExchangeException(Exception):
    pass


class ExchangeTimeoutException(DataExchangeException):
    pass


class ExchangeAbortException(DataExchangeException):
    pass


class ExchangeEndException(DataExchangeException):
    pass


class ExchangePeerGoneException(DataExchangeException):
    pass


class DataExchanger:
    def __init__(
        self,
        supported_topics: List[str],
        pipe: Pipe,
        pipe_name: str = "pipe",
        get_poll_interval: float = 0.5,
        read_interval: float = 0.1,
        heartbeat_interval: float = 5.0,
        heartbeat_timeout: float = 30.0,
    ):
        """Initializes the DataExchanger.

        Args:
            supported_topics (list[str]): Supported topics for data exchange. This allows the sender and receiver to identify
                the purpose or content of the data being exchanged.
            pipe (Pipe): The pipe used for data exchange.
            pipe_name (str): Name of the pipe. Defaults to "pipe".
            get_poll_interval (float): Interval for checking if the other side has sent data. Defaults to 0.5.
            read_interval (float): Interval for reading from the pipe. Defaults to 0.1.
            heartbeat_interval (float): Interval for sending heartbeat to the peer. Defaults to 5.0.
            heartbeat_timeout (float): Timeout for waiting for a heartbeat from the peer. Defaults to 30.0.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self._req_id: Optional[str] = None
        self.current_topic: Optional[str] = None
        self._supported_topics = supported_topics

        pipe.open(pipe_name)
        self.pipe_handler = PipeHandler(
            pipe,
            read_interval=read_interval,
            heartbeat_interval=heartbeat_interval,
            heartbeat_timeout=heartbeat_timeout,
        )
        self.pipe_handler.start()
        self._get_poll_interval = get_poll_interval

    def submit_data(self, data: Any) -> None:
        """Submits a data for exchange.

        Args:
            data (Any): The data to be submitted.

        Raises:
            DataExchangeException: If there is no request ID available (needs to pull data from server first).
        """
        if self._req_id is None:
            raise DataExchangeException("Missing req_id, need to pull a data first.")

        if self.current_topic is None:
            raise DataExchangeException("Missing current_topic, need to pull a data first.")

        self._send_reply(data=data, topic=self.current_topic, req_id=self._req_id)

    def receive_data(self, timeout: Optional[float] = None) -> Tuple[str, Any]:
        """Receives a data.

        Args:
            timeout (Optional[float]): Timeout for waiting to receive a data. Defaults to None.

        Returns:
            A tuple of (topic, data): The received data.

        Raises:
            ExchangeTimeoutException: If the data cannot be received within the specified timeout.
            ExchangeAbortException: If the other endpoint of the pipe requests to abort.
            ExchangeEndException: If the other endpoint has ended.
            ExchangePeerGoneException: If the other endpoint is gone.
        """
        msg = self._receive_request(timeout)
        self._req_id = msg.msg_id
        self.current_topic = msg.topic
        return msg.topic, msg.data

    def finalize(self, close_pipe: bool = True) -> None:
        if self.pipe_handler is None:
            raise RuntimeError("PipeMonitor is not initialized.")
        self.pipe_handler.stop(close_pipe=close_pipe)

    def _receive_request(self, timeout: Optional[float] = None) -> Message:
        """Receives a request.

        Args:
            timeout: how long to wait for the request to come.

        Returns:
            A Message.

        Raises:
            ExchangeTimeoutException: If can't receive data within timeout seconds.
            ExchangeAbortException: If the other endpoint of the pipe ask to abort.
            ExchangeEndException: If the other endpoint has ended.
            ExchangePeerGoneException: If the other endpoint is gone.
        """
        if self.pipe_handler is None:
            raise RuntimeError("PipeMonitor is not initialized.")
        start = time.time()
        while True:
            msg: Optional[Message] = self.pipe_handler.get_next()
            if not msg:
                if timeout and time.time() - start > timeout:
                    self.pipe_handler.notify_abort(msg)
                    raise ExchangeTimeoutException(f"get data timeout after {timeout} secs")
            elif msg.topic == Topic.ABORT:
                raise ExchangeAbortException("the other end ask to abort")
            elif msg.topic == Topic.END:
                raise ExchangeEndException(f"received msg: '{msg}' while waiting for requests")
            elif msg.topic == Topic.PEER_GONE:
                raise ExchangePeerGoneException(f"received msg: '{msg}' while waiting for requests")
            elif msg.topic in self._supported_topics:
                return msg
            time.sleep(self._get_poll_interval)

    def _send_reply(self, data: Any, topic: str, req_id: str, timeout: Optional[float] = None) -> bool:
        """Sends a reply.

        Args:
            data: The data exchange object to be sent.
            topic: message topic.
            req_id: request ID.
            timeout: how long to wait for the peer to read the data.
                If not specified, return False immediately.

        Returns:
            A bool indicates whether the peer has read the data.
        """
        if self.pipe_handler is None:
            raise RuntimeError("PipeMonitor is not initialized.")
        msg = Message.new_reply(topic=topic, data=data, req_msg_id=req_id)
        has_been_read = self.pipe_handler.send_to_peer(msg, timeout)
        return has_been_read
