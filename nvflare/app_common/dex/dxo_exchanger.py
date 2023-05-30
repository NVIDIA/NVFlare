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
import os
import time
from typing import Optional

from nvflare.apis.dxo import DXO
from nvflare.fuel.utils.pipe.file_accessor import FileAccessor
from nvflare.fuel.utils.pipe.file_pipe import FilePipe
from nvflare.fuel.utils.pipe.pickle_file_accessor import PickleFileAccessor
from nvflare.fuel.utils.pipe.pipe import Message
from nvflare.fuel.utils.pipe.pipe_monitor import PipeMonitor, Topic


class DXOExchanger:
    def __init__(
        self,
        pipe_role: str,
        pipe_name: str = "pipe",
        get_poll_interval: float = 0.1,
        heartbeat_interval: float = 0.5,
        heartbeat_timeout: float = 0.0,
    ):
        """Exchanges DXO.

        Args:
            pipe_role (str): Should be either "x", or "y". A pipe has two endpoints, one side should be "x" while
                the other side should be "y".
            pipe_name (str): Name of the pipe, default is "pipe".
            get_poll_interval (float): how often to check if the other side has sent file
            heartbeat_interval (float): how often to send a heartbeat to the peer
            heartbeat_timeout (float): how long to wait for a heartbeat from the peer before treating the peer as gone,
                0 means DO NOT check for heartbeat.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self._get_poll_interval = get_poll_interval

        self.data_exchange_path = None
        self._pipe_role = pipe_role
        self._pipe_name = pipe_name
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_timeout = heartbeat_timeout
        self.pipe_monitor = None
        self._file_accessor_name = None

    def initialize(self, data_exchange_path: str, file_accessor: Optional[FileAccessor] = None):
        self.data_exchange_path = os.path.abspath(data_exchange_path)
        file_pipe = FilePipe(self.data_exchange_path)
        if file_accessor is not None:
            file_pipe.set_file_accessor(file_accessor)
        else:
            file_pipe.set_file_accessor(PickleFileAccessor())

        # use file accessor name to determine serialize/deserialize
        self._file_accessor_name = file_pipe.accessor.__class__.__name__
        file_pipe.open(self._pipe_name, self._pipe_role)
        self.pipe_monitor = PipeMonitor(
            file_pipe,
            read_interval=self._get_poll_interval,
            heartbeat_interval=self._heartbeat_interval,
            heartbeat_timeout=self._heartbeat_timeout,
        )
        self.pipe_monitor.start()

    def put(self, data_id: str, data: DXO):
        if self.pipe_monitor is None:
            raise RuntimeError("PipeMonitor is not initialized.")
        req = Message.new_request(self._file_accessor_name, data, data_id)
        self.pipe_monitor.send_to_peer(req)

    def _get_file(self, req_topic: str, data_id: Optional[str] = None, timeout: Optional[float] = None):
        if self.pipe_monitor is None:
            raise RuntimeError("PipeMonitor is not initialized.")
        start = time.time()
        while True:
            msg: Message = self.pipe_monitor.get_next()
            if not msg:
                if timeout and time.time() - start > timeout:
                    self.pipe_monitor.notify_abort()
                    raise TimeoutError(f"get file timeout after {timeout} secs")
            elif msg.topic == Topic.HEARTBEAT:
                continue
            elif msg.topic == Topic.ABORT:
                raise RuntimeError("the other end is aborted")
            elif msg.topic in [Topic.END, Topic.PEER_GONE]:
                raise RuntimeError(f"received {msg.topic}: {msg.data} while waiting for result for {req_topic}")
            elif msg.topic == req_topic and data_id is None:
                return msg.data
            elif msg.topic == req_topic and msg.msg_id == data_id:
                return msg.data
            time.sleep(self._get_poll_interval)

    def get(self, data_id: Optional[str] = None, timeout: Optional[float] = None):
        if self.pipe_monitor is None:
            raise RuntimeError("PipeMonitor is not initialized.")
        data = self._get_file(self._file_accessor_name, data_id, timeout)
        return data

    def finalize(self):
        if self.pipe_monitor is None:
            raise RuntimeError("PipeMonitor is not initialized.")
        self.pipe_monitor.stop(close_pipe=True)
