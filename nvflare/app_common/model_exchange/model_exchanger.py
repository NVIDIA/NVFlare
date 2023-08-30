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
from typing import Any, Optional, Tuple

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


class ModelExchanger:
    def __init__(
        self,
        pipe: Pipe,
        pipe_name: str = "pipe",
        topic: str = "data",
        get_poll_interval: float = 0.5,
        read_interval: float = 0.1,
        heartbeat_interval: float = 5.0,
        heartbeat_timeout: float = 30.0,
    ):
        """Initializes the ModelExchanger.

        Args:
            pipe (Pipe): The pipe used for data exchange.
            pipe_name (str): Name of the pipe. Defaults to "pipe".
            topic (str): Topic for data exchange. Defaults to "data".
            get_poll_interval (float): Interval for checking if the other side has sent data. Defaults to 0.5.
            read_interval (float): Interval for reading from the pipe. Defaults to 0.1.
            heartbeat_interval (float): Interval for sending heartbeat to the peer. Defaults to 5.0.
            heartbeat_timeout (float): Timeout for waiting for a heartbeat from the peer. Defaults to 30.0.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self._req_id: Optional[str] = None
        self._topic = topic

        pipe.open(pipe_name)
        self.pipe_handler = PipeHandler(
            pipe,
            read_interval=read_interval,
            heartbeat_interval=heartbeat_interval,
            heartbeat_timeout=heartbeat_timeout,
        )
        self.pipe_handler.start()
        self._get_poll_interval = get_poll_interval

    def submit_model(self, model: Any) -> None:
        """Submits a model for exchange.

        Args:
            model (Any): The model to be submitted.

        Raises:
            DataExchangeException: If there is no request ID available (needs to pull model from server first).
        """
        if self._req_id is None:
            raise DataExchangeException("need to pull a model first.")
        self._send_reply(data=model, req_id=self._req_id)

    def receive_model(self, timeout: Optional[float] = None) -> Any:
        """Receives a model.

        Args:
            timeout (Optional[float]): Timeout for waiting to receive a model. Defaults to None.

        Returns:
            Any: The received model.

        Raises:
            ExchangeTimeoutException: If the data cannot be received within the specified timeout.
            ExchangeAbortException: If the other endpoint of the pipe requests to abort.
            ExchangeEndException: If the other endpoint has ended.
            ExchangePeerGoneException: If the other endpoint is gone.
        """
        model, req_id = self._receive_request(timeout)
        self._req_id = req_id
        return model

    def finalize(self, close_pipe: bool = True) -> None:
        if self.pipe_handler is None:
            raise RuntimeError("PipeMonitor is not initialized.")
        self.pipe_handler.stop(close_pipe=close_pipe)

    def _receive_request(self, timeout: Optional[float] = None) -> Tuple[Any, str]:
        """Receives a request.

        Args:
            timeout: how long to wait for the request to come.

        Returns:
            A tuple of (data, request id).

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
                raise ExchangeAbortException("the other end is aborted")
            elif msg.topic == Topic.END:
                raise ExchangeEndException(
                    f"received {msg.topic}: {msg.data} while waiting for result for {self._topic}"
                )
            elif msg.topic == Topic.PEER_GONE:
                raise ExchangePeerGoneException(
                    f"received {msg.topic}: {msg.data} while waiting for result for {self._topic}"
                )
            elif msg.topic == self._topic:
                return msg.data, msg.msg_id
            time.sleep(self._get_poll_interval)

    def _send_reply(self, data: Any, req_id: str, timeout: Optional[float] = None) -> bool:
        """Sends a reply.

        Args:
            data: The data exchange object to be sent.
            req_id: request ID.
            timeout: how long to wait for the peer to read the data.
                If not specified, return False immediately.

        Returns:
            A bool indicates whether the peer has read the data.
        """
        if self.pipe_handler is None:
            raise RuntimeError("PipeMonitor is not initialized.")
        msg = Message.new_reply(topic=self._topic, data=data, req_msg_id=req_id)
        has_been_read = self.pipe_handler.send_to_peer(msg, timeout)
        return has_been_read
