# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from abc import ABC, abstractmethod
from typing import Dict, List

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.fuel.data_event.data_bus import DataBus
from nvflare.metrics.metrics_keys import MetricKeys, MetricTypes
from nvflare.metrics.metrics_publisher import collect_metrics


class MetricsCollector(FLComponent, ABC):
    def __init__(self, tags: dict, streaming_to_server: bool = False):
        """
        Args:
            tags: comma separated static tags. used to specify server, client, production, test etc.
        """
        super().__init__()

        self.tags = tags
        self.data_bus = DataBus()
        self.streaming_to_server = streaming_to_server

        self.event_start_time = {}

    @abstractmethod
    def get_single_events() -> List[str]:
        pass

    @abstractmethod
    def get_pair_events() -> Dict:
        pass

    def get_pair_start_events(self):
        """Returns events that explicitly start duration measurements.

        Subclasses that define multiple pair events for one duration metric should
        list the start event here. If this returns None or no start event is
        defined for a metric, the collector preserves legacy toggle behavior for
        that metric.
        """
        return None

    @staticmethod
    def _build_pair_event_maps(pair_event_specs):
        """Build pair-event lookup maps from (start_event, *end_events, metric_name) specs.

        A spec with only one event keeps the legacy toggle behavior and does not
        declare an explicit start event.
        """
        pair_events = {}
        pair_start_events = []
        for *events, metric_name in pair_event_specs:
            if not events:
                continue
            for event in events:
                pair_events[event] = metric_name
            if len(events) > 1:
                pair_start_events.append(events[0])
        return pair_events, pair_start_events

    def collect_event_metrics(self, event: str, tags, fl_ctx: FLContext):

        current_time = time.time()
        metric_name = event

        metrics = {MetricKeys.count: 1, MetricKeys.type: MetricTypes.COUNTER}
        duration_metrics = {MetricKeys.time_taken: 0, MetricKeys.type: MetricTypes.GAUGE}
        pair_events = self.get_pair_events()
        pair_start_events = self.get_pair_start_events()
        if pair_start_events is not None:
            pair_start_events = set(pair_start_events)

        if event in self.get_single_events():
            self.publish_metrics(metrics, metric_name, tags, fl_ctx)
        elif event in pair_events.keys():
            self.publish_metrics(metrics, metric_name, tags, fl_ctx)
            key = pair_events.get(event)
            pair_event_role = self._get_pair_event_role(
                event=event, key=key, pair_events=pair_events, pair_start_events=pair_start_events
            )
            if pair_event_role == "start":
                self.event_start_time[key] = current_time
            elif pair_event_role == "end":
                self._publish_pair_duration(key, current_time, duration_metrics, tags, fl_ctx)
            else:
                self._collect_legacy_pair_event(key, current_time, duration_metrics, tags, fl_ctx)

    def _get_pair_event_role(self, event: str, key: str, pair_events: Dict, pair_start_events: set):
        if pair_start_events is None:
            return None
        if event in pair_start_events:
            return "start"

        events_for_metric = [event_name for event_name, metric_key in pair_events.items() if metric_key == key]
        if len(events_for_metric) <= 1:
            return None
        if any(event_name in pair_start_events for event_name in events_for_metric):
            return "end"
        return None

    def _collect_legacy_pair_event(
        self, key: str, current_time: float, duration_metrics: dict, tags, fl_ctx: FLContext
    ):
        if self.event_start_time.get(key) is None:
            self.event_start_time[key] = current_time
        else:
            self._publish_pair_duration(key, current_time, duration_metrics, tags, fl_ctx)

    def _publish_pair_duration(self, key: str, current_time: float, duration_metrics: dict, tags, fl_ctx: FLContext):
        start_time = self.event_start_time.get(key)
        if start_time is None:
            return

        time_taken = current_time - start_time
        self.event_start_time[key] = None
        duration_metrics[MetricKeys.time_taken] = time_taken
        self.publish_metrics(duration_metrics, key, tags, fl_ctx)

    def publish_metrics(self, metrics: dict, metric_name: str, tags: dict, fl_ctx: FLContext):

        collect_metrics(self, self.streaming_to_server, metrics, metric_name, tags, self.data_bus, fl_ctx)
