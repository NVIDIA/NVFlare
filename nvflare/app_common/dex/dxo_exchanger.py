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
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from nvflare.apis.dxo import DXO
from nvflare.fuel.utils.pipe.pipe import Message, Pipe
from nvflare.fuel.utils.pipe.pipe_monitor import PipeMonitor, Topic


class DXOExchangeException(Exception):
    pass


class ExchangeTimeoutException(DXOExchangeException):
    pass


class ExchangeAbortException(DXOExchangeException):
    pass


class ExchangeEndException(DXOExchangeException):
    pass


class ExchangePeerGoneException(DXOExchangeException):
    pass


class DXOExchanger(ABC):
    def __init__(
        self,
        pipe_role: str,
        pipe_name: str = "pipe",
        get_poll_interval: float = 0.5,
        heartbeat_interval: float = 5.0,
        heartbeat_timeout: float = 30.0,
    ):
        """Exchanges DXO.

        Args:
            get_poll_interval (float): how often to check if the other side has sent data.
            heartbeat_interval (float): how often to send a heartbeat to the peer.
            heartbeat_timeout (float): how long to wait for a heartbeat from the peer before treating the peer as gone,
                0 means DO NOT check for heartbeat.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self._get_poll_interval = get_poll_interval

        self._pipe_role = pipe_role
        self._pipe_name = pipe_name
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_timeout = heartbeat_timeout
        self._topic = "dxo"
        self.pipe_monitor = None

    @abstractmethod
    def create_pipe(self, *args, **kwargs) -> Pipe:
        """Creates a Pipe."""
        pass

    def initialize(self, *args, **kwargs):
        """Initializes the DXOExchanger.

        The *arg and **kwargs will be passed into the "create_pipe" method.
        After the pipe is created and open, it will be passed into PipeMonitor.
        """
        pipe = self.create_pipe(*args, **kwargs)
        pipe.open(self._pipe_name, self._pipe_role)
        self.pipe_monitor = PipeMonitor(
            pipe,
            read_interval=self._get_poll_interval,
            heartbeat_interval=self._heartbeat_interval,
            heartbeat_timeout=self._heartbeat_timeout,
        )
        self.pipe_monitor.start()

    def send_request(self, dxo: DXO, timeout: Optional[float] = None) -> Tuple[bool, str]:
        """Sends a request.

        Args:
            dxo: The data exchange object to be sent.
            timeout: how long to wait for the peer to read the data.

        Returns:
            A tuple of (has_been_read, request message id).
            has_been_read indicates whether the peer has read the data.
        """
        if self.pipe_monitor is None:
            raise RuntimeError("PipeMonitor is not initialized.")
        msg = Message.new_request(topic=self._topic, data=dxo)
        has_been_read = self.pipe_monitor.send_to_peer(msg, timeout)
        return has_been_read, msg.msg_id

    def receive_request(self, timeout: Optional[float] = None) -> Tuple[DXO, str]:
        """Receives a request.

        Args:
            timeout: how long to wait for the request to come.

        Returns:
            A tuple of (data, request id).
        """
        data, req_id = self._receive_message(timeout=timeout)
        return data, req_id

    def send_reply(self, dxo: DXO, req_id: str, timeout: Optional[float] = None) -> bool:
        """Sends a reply.

        Args:
            dxo: The data exchange object to be sent.
            req_id: request ID.
            timeout: how long to wait for the peer to read the data.

        Returns:
            A bool indicates whether the peer has read the data.
        """
        if self.pipe_monitor is None:
            raise RuntimeError("PipeMonitor is not initialized.")
        msg = Message.new_reply(topic=self._topic, data=dxo, req_msg_id=req_id)
        has_been_read = self.pipe_monitor.send_to_peer(msg, timeout)
        return has_been_read

    def receive_reply(self, req_id: str, timeout: Optional[float] = None) -> DXO:
        """Receives a reply.

        Args:
            req_id: request ID.
            timeout: how long to wait for the request to come.

        Returns:
            A data of type DXO.
        """
        data, _ = self._receive_message(req_msg_id=req_id, timeout=timeout)
        return data

    def _receive_message(self, req_msg_id: Optional[str] = None, timeout: Optional[float] = None) -> Tuple[DXO, str]:
        if self.pipe_monitor is None:
            raise RuntimeError("PipeMonitor is not initialized.")
        start = time.time()
        while True:
            msg: Message = self.pipe_monitor.get_next()
            if not msg:
                if timeout and time.time() - start > timeout:
                    self.pipe_monitor.notify_abort()
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
            elif msg.topic == self._topic and (req_msg_id is None or msg.req_id == req_msg_id):
                return msg.data, msg.msg_id
            time.sleep(self._get_poll_interval)

    def finalize(self):
        if self.pipe_monitor is None:
            raise RuntimeError("PipeMonitor is not initialized.")
        self.pipe_monitor.stop(close_pipe=True)
