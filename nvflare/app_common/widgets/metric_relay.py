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

from typing import Tuple

from nvflare.apis.dxo import DXO
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.utils.analytix_utils import send_analytic_dxo
from nvflare.app_common.tracking.tracker_types import ANALYTIC_EVENT_TYPE
from nvflare.client.config import ConfigKey
from nvflare.fuel.utils.attributes_exportable import AttributesExportable
from nvflare.fuel.utils.constants import PipeChannelName
from nvflare.fuel.utils.pipe.pipe import Message, Pipe
from nvflare.fuel.utils.pipe.pipe_handler import PipeHandler
from nvflare.widgets.widget import Widget


class MetricRelay(Widget, AttributesExportable):
    def __init__(
        self,
        pipe_id: str,
        read_interval=0.1,
        heartbeat_interval=5.0,
        heartbeat_timeout=60.0,
        pipe_channel_name=PipeChannelName.METRIC,
        event_type: str = ANALYTIC_EVENT_TYPE,
        fed_event: bool = True,
    ):
        super().__init__()
        self.pipe_id = pipe_id
        self._read_interval = read_interval
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_timeout = heartbeat_timeout
        self.pipe_channel_name = pipe_channel_name
        self.pipe = None
        self.pipe_handler = None
        self._fl_ctx = None
        self._event_type = event_type
        self._fed_event = fed_event

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.ABOUT_TO_START_RUN:
            engine = fl_ctx.get_engine()
            pipe = engine.get_component(self.pipe_id)
            if not isinstance(pipe, Pipe):
                self.log_error(fl_ctx, f"component {self.pipe_id} must be Pipe but got {type(pipe)}")
                self.system_panic(f"bad component {self.pipe_id}", fl_ctx)
                return
            self._fl_ctx = fl_ctx
            self.pipe = pipe
            self.pipe_handler = PipeHandler(
                pipe=self.pipe,
                read_interval=self._read_interval,
                heartbeat_interval=self._heartbeat_interval,
                heartbeat_timeout=self._heartbeat_timeout,
            )
            self.pipe_handler.set_status_cb(self._pipe_status_cb)
            self.pipe_handler.set_message_cb(self._pipe_msg_cb)
            self.pipe.open(self.pipe_channel_name)
        elif event_type == EventType.BEFORE_TASK_EXECUTION:
            self.pipe_handler.start()
        elif event_type == EventType.ABOUT_TO_END_RUN:
            self.log_info(fl_ctx, "Stopping pipe handler")
            if self.pipe_handler:
                self.pipe_handler.notify_end("end_of_job")
                self.pipe_handler.stop()

    def _pipe_status_cb(self, msg: Message):
        self.logger.info(f"{self.pipe_channel_name} pipe status changed to {msg.topic}")
        self.pipe_handler.stop()

    def _pipe_msg_cb(self, msg: Message):
        if not isinstance(msg.data, DXO):
            self.logger.error(f"bad metric data: expect DXO but got {type(msg.data)}")
        send_analytic_dxo(self, msg.data, self._fl_ctx, self._event_type, fire_fed_event=self._fed_event)

    def export(self, export_mode: str) -> Tuple[str, dict]:
        pipe_export_class, pipe_export_args = self.pipe.export(export_mode)
        config_dict = {
            ConfigKey.PIPE_CHANNEL_NAME: self.pipe_channel_name,
            ConfigKey.PIPE: {
                ConfigKey.CLASS_NAME: pipe_export_class,
                ConfigKey.ARG: pipe_export_args,
            },
            ConfigKey.HEARTBEAT_TIMEOUT: self._heartbeat_timeout,
        }
        return ConfigKey.METRICS_EXCHANGE, config_dict
