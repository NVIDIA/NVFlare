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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.metrics_exchange.metrics_exchanger import MetricsExchanger
from nvflare.app_common.widgets.metric_relay import MetricRelay
from nvflare.fuel.utils.constants import Mode
from nvflare.fuel.utils.pipe.memory_pipe import MemoryPipe
from nvflare.fuel.utils.pipe.pipe_handler import PipeHandler


class MetricsRetriever(MetricRelay):
    def __init__(
        self,
        metrics_exchanger_id: str,
        pipe_id: str = "_memory_pipe",
        read_interval: float = 0.1,
        heartbeat_interval: float = 5.0,
        heartbeat_timeout: float = 30.0,
    ):
        """Metrics retriever.

        Please do not use this class.

        This class creates `MetricsExchanger` that is used by
        class `LogWriterForMetricsExchanger` and the classes that extends it.
        This exists just for compatibility of `LogWriterForMetricsExchanger`.
        We will re-factor those classes later.
        """
        super().__init__(
            pipe_id=pipe_id,
            read_interval=read_interval,
            heartbeat_interval=heartbeat_interval,
            heartbeat_timeout=heartbeat_timeout,
        )
        self.metrics_exchanger_id = metrics_exchanger_id

    def _create_metrics_exchanger(self):
        pipe = MemoryPipe(token=self.pipe.token, mode=Mode.ACTIVE)
        pipe.open(self.pipe_channel_name)

        # init pipe handler
        pipe_handler = PipeHandler(
            pipe,
            read_interval=self._read_interval,
            heartbeat_interval=self._heartbeat_interval,
            heartbeat_timeout=self._heartbeat_timeout,
        )
        pipe_handler.start()
        metrics_exchanger = MetricsExchanger(pipe_handler=pipe_handler)
        return metrics_exchanger

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        super().handle_event(event_type, fl_ctx)
        if event_type == EventType.ABOUT_TO_START_RUN:
            engine = fl_ctx.get_engine()
            # inserts MetricsExchanger into engine components
            metrics_exchanger = self._create_metrics_exchanger()
            all_components = engine.get_all_components()
            all_components[self.metrics_exchanger_id] = metrics_exchanger
