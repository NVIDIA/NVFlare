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

import time
from queue import Queue
from threading import Event, Thread
from typing import Optional

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.metrics_exchange.metrics_exchanger import MetricData, MetricsExchanger
from nvflare.app_common.tracking.tracker_types import LogWriterName
from nvflare.app_common.widgets.streaming import ANALYTIC_EVENT_TYPE, AnalyticsSender
from nvflare.fuel.utils.constants import Mode
from nvflare.fuel.utils.pipe.memory_pipe import MemoryPipe
from nvflare.fuel.utils.pipe.pipe import Message
from nvflare.fuel.utils.pipe.pipe_handler import PipeHandler, Topic


class MetricsRetriever(FLComponent):
    def __init__(
        self,
        metrics_exchanger_id: str,
        event_type=ANALYTIC_EVENT_TYPE,
        writer_name=LogWriterName.TORCH_TB,
        topic: str = "metrics",
        get_poll_interval: float = 0.5,
        read_interval: float = 0.1,
        heartbeat_interval: float = 5.0,
        heartbeat_timeout: float = 30.0,
    ):
        """Metrics retriever.

        Args:
            event_type (str): event type to fire (defaults to "analytix_log_stats").
            writer_name: the log writer for syntax information (defaults to LogWriterName.TORCH_TB)
        """
        super().__init__()
        self.metrics_exchanger_id = metrics_exchanger_id
        self.analytic_sender = AnalyticsSender(event_type=event_type, writer_name=writer_name)
        self.x_queue = Queue()
        self.y_queue = Queue()

        self.read_interval = read_interval
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.pipe_handler = self._create_pipe_handler(mode=Mode.PASSIVE)

        self._topic = topic
        self._get_poll_interval = get_poll_interval
        self.stop = Event()
        self._receive_thread = Thread(target=self.receive_data)
        self.fl_ctx = None

    def _create_pipe_handler(self, *, mode):
        memory_pipe = MemoryPipe(x_queue=self.x_queue, y_queue=self.y_queue, mode=mode)
        pipe_handler = PipeHandler(
            memory_pipe,
            read_interval=self.read_interval,
            heartbeat_interval=self.heartbeat_interval,
            heartbeat_timeout=self.heartbeat_timeout,
        )
        pipe_handler.start()
        return pipe_handler

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.ABOUT_TO_START_RUN:
            engine = fl_ctx.get_engine()
            self.analytic_sender.handle_event(event_type, fl_ctx)
            # inserts MetricsExchanger into engine components
            pipe_handler = self._create_pipe_handler(mode=Mode.ACTIVE)
            metrics_exchanger = MetricsExchanger(pipe_handler=pipe_handler)
            all_components = engine.get_all_components()
            all_components[self.metrics_exchanger_id] = metrics_exchanger
            self.fl_ctx = fl_ctx
            self._receive_thread.start()
        elif event_type == EventType.ABOUT_TO_END_RUN:
            self.stop.set()
            self._receive_thread.join()

    def receive_data(self):
        """Receives data and sends with AnalyticsSender."""
        while True:
            if self.stop.is_set():
                break
            msg: Optional[Message] = self.pipe_handler.get_next()
            if msg is not None:
                if msg.topic == [Topic.END, Topic.PEER_GONE, Topic.ABORT]:
                    self.task_panic("abort task", self.fl_ctx)
                elif msg.topic != self._topic:
                    self.task_panic(f"ignored '{msg.topic}' when waiting for '{self._topic}'", self.fl_ctx)
                else:
                    data: MetricData = msg.data
                    # TODO: unpack the format and pass it into "add"
                    self.analytic_sender.add(
                        tag=data.key, value=data.value, data_type=data.data_type, **data.additional_args
                    )
            time.sleep(self._get_poll_interval)
