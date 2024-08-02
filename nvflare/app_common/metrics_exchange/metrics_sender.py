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

from typing import Any

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.utils.analytix_utils import create_analytic_dxo
from nvflare.app_common.widgets.streaming import AnalyticsSender
from nvflare.fuel.utils.constants import PipeChannelName
from nvflare.fuel.utils.pipe.pipe import Message, Pipe
from nvflare.fuel.utils.pipe.pipe_handler import PipeHandler


class MetricsSender(AnalyticsSender):
    def __init__(
        self,
        pipe_id: str = "_memory_pipe",
        read_interval: float = 0.1,
        heartbeat_interval: float = 5.0,
        heartbeat_timeout: float = 30.0,
        pipe_channel_name=PipeChannelName.METRIC,
    ):
        """MetricsSender is a special type of AnalyticsSender that uses `Pipe` to communicate.

        Args:
            pipe_id (str): Identifier for obtaining the Pipe from NVFlare components.
            read_interval (float): Interval for reading from the pipe.
            heartbeat_interval (float): Interval for sending heartbeat to the peer.
            heartbeat_timeout (float): Timeout for waiting for a heartbeat from the peer.
            pipe_channel_name: the channel name for sending task requests.

        Note:
            Users can use MetricsSender with `FilePipe`, `CellPipe`, or any other customize
            `Pipe` class.

        """
        super().__init__()
        self._pipe_id = pipe_id
        self._read_interval = read_interval
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_timeout = heartbeat_timeout
        self._pipe_handler = None
        self._pipe_channel_name = pipe_channel_name

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.ABOUT_TO_START_RUN:
            self.engine = fl_ctx.get_engine()
            pipe = self.engine.get_component(self._pipe_id)
            if not isinstance(pipe, Pipe):
                self.log_error(fl_ctx, f"component {self._pipe_id} must be Pipe but got {type(pipe)}")
                self.system_panic(f"bad component {self._pipe_id}", fl_ctx)
                return

            pipe.open(self._pipe_channel_name)

            self._pipe_handler = PipeHandler(
                pipe,
                read_interval=self._read_interval,
                heartbeat_interval=self._heartbeat_interval,
                heartbeat_timeout=self._heartbeat_timeout,
            )
            self._pipe_handler.start()

    def add(self, tag: str, value: Any, data_type: AnalyticsDataType, **kwargs):
        data = create_analytic_dxo(tag=tag, value=value, data_type=data_type, **kwargs)
        req = Message.new_request(topic="_metrics_sender", data=data)
        self._pipe_handler.send_to_peer(req)
