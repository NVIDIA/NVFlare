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


from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.metrics.metrics_collector import MetricsCollector


class SysMetricsCollector(MetricsCollector):
    def __init__(self, tags: dict, streaming_to_server: bool = False):
        """
        Args:
            tags: comma separated static tags. used to specify server, client, production, test etc.
        """
        super().__init__(tags=tags, streaming_to_server=streaming_to_server)

        self.pair_events = {
            EventType.SYSTEM_START: "_system",
            EventType.SYSTEM_END: "_system",
            EventType.BEFORE_CHECK_CLIENT_RESOURCES: "_check_client_resources",
            EventType.AFTER_CHECK_CLIENT_RESOURCES: "_check_client_resources",
            EventType.BEFORE_CLIENT_REGISTER: "_client_register",
            EventType.AFTER_CLIENT_REGISTER: "_client_register",
            EventType.BEFORE_CLIENT_HEARTBEAT: "_client_heartbeat",
            EventType.AFTER_CLIENT_HEARTBEAT: "_client_heartbeat",
        }

        self.single_events = [
            EventType.CLIENT_DISCONNECTED,
            EventType.CLIENT_RECONNECTED,
            EventType.BEFORE_CHECK_RESOURCE_MANAGER,
            EventType.BEFORE_SEND_ADMIN_COMMAND,
            EventType.CLIENT_REGISTER_RECEIVED,
            EventType.CLIENT_REGISTER_PROCESSED,
            EventType.CLIENT_QUIT,
            EventType.SYSTEM_BOOTSTRAP,
            EventType.CLIENT_HEARTBEAT_RECEIVED,
            EventType.CLIENT_HEARTBEAT_PROCESSED,
            EventType.BEFORE_JOB_LAUNCH,
        ]

    def handle_event(self, event: str, fl_ctx: FLContext):
        super().collect_event_metrics(event=event, tags=self.tags, fl_ctx=fl_ctx)

    def get_single_events(self):
        return self.single_events

    def get_pair_events(self):
        return self.pair_events
