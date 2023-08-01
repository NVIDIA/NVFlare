# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from queue import Empty, Queue
from threading import Event, Thread

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.metric_exchange.metric_exchanger import MetricsExchanger
from nvflare.app_common.tracking.tracker_types import LogWriterName
from nvflare.app_common.widgets.streaming import ANALYTIC_EVENT_TYPE, AnalyticsSender


class MetricHandler(FLComponent):
    def __init__(
        self,
        metric_exchanger_id: str,
        event_type=ANALYTIC_EVENT_TYPE,
        writer_name=LogWriterName.TORCH_TB,
        get_poll_interval=0.5,
    ):
        """Metric Handler.

        Args:
            event_type (str): event type to fire (defaults to "analytix_log_stats").
            writer_name: the log writer for syntax information (defaults to LogWriterName.TORCH_TB)
        """
        super().__init__()
        self.metric_exchanger_id = metric_exchanger_id
        self.analytic_sender = AnalyticsSender(event_type=event_type, writer_name=writer_name)
        self.data_queue = Queue()
        self.start = Event()
        self._get_poll_interval = get_poll_interval
        self._receive_thread = Thread(target=self.receive_data)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.ABOUT_TO_START_RUN:
            engine = fl_ctx.get_engine()
            self.analytic_sender.handle_event(event_type, fl_ctx)
            # inserts MetricsExchanger into engine components
            metrics_exchanger = MetricsExchanger(queue=self.data_queue)
            all_components = engine.get_all_components()
            all_components[self.metric_exchanger_id] = metrics_exchanger
            self._receive_thread.start()
        elif event_type == EventType.ABOUT_TO_END_RUN:
            self.start.set()
            self._receive_thread.join()

    def receive_data(self):
        """Receives a data."""
        while True:
            if self.start.is_set():
                break
            try:
                data = self.data_queue.get(block=False)
            except Empty:
                data = None
            if data is not None:
                print(f"MetricHandler getting things {data} from queue")
                self.analytic_sender.add(**data)
            time.sleep(self._get_poll_interval)
