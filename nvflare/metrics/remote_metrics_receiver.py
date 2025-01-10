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


from typing import List, Optional

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import EventScope, FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.fuel.data_event.data_bus import DataBus
from nvflare.metrics.metrics_keys import METRICS_EVENT_TYPE, MetricKeys
from nvflare.metrics.metrics_publisher import publish_app_metrics


class RemoteMetricsReceiver(FLComponent):
    def __init__(self, events: Optional[List[str]] = None):
        """Receives metrics data from client sites and publishes it to the local data bus.
        Args:
            events (optional, List[str]): A list of event that this receiver will handle.
        """
        super().__init__()
        if events is None:
            events = [METRICS_EVENT_TYPE, f"fed.{METRICS_EVENT_TYPE}"]
        self.events = events
        self.data_bus = DataBus()

    def handle_event(self, event_type: str, fl_ctx: FLContext):

        if event_type in self.events:
            data = fl_ctx.get_prop(FLContextKey.EVENT_DATA, None)
            if data is None:
                self.log_error(fl_ctx, "Missing event data.", fire_event=False)
                return
            if not isinstance(data, Shareable):
                self.log_error(
                    fl_ctx, f"Expect data to be an instance of Shareable but got {type(data)}", fire_event=False
                )
                return

            # if fed event use peer name to save
            if fl_ctx.get_prop(FLContextKey.EVENT_SCOPE) == EventScope.FEDERATION:
                record_origin = data.get_peer_prop(ReservedKey.IDENTITY_NAME, None)
            else:
                record_origin = fl_ctx.get_identity_name()

            if record_origin is None:
                self.log_error(fl_ctx, "record_origin can't be None.", fire_event=False)
                return

            metrics_data = data.get("METRICS")
            if metrics_data is None:
                self.log_error(fl_ctx, "Missing metrics data.", fire_event=False)
                return

            metric_name = metrics_data.get(MetricKeys.metric_name)
            metrics = metrics_data.get(MetricKeys.value)
            tags = metrics_data.get(MetricKeys.tags)

            publish_app_metrics(metrics=metrics, metric_name=metric_name, tags=tags, data_bus=self.data_bus)
