# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


import traceback

import requests

from nvflare.apis.fl_constant import ReservedTopic
from nvflare.fuel.data_event.data_bus import DataBus


class MetricsCollector:
    def __init__(self, metrics_server_url="http://localhost:8000/update_metrics"):
        self.metrics = {}
        self.data_bus = DataBus()
        self.data_bus.subscribe([ReservedTopic.APP_METRICS], self.process_metrics)
        self.metrics_server_url = metrics_server_url

    def process_metrics(self, topic, metrics, data_bus):
        try:
            if topic == ReservedTopic.APP_METRICS:
                # Send metrics data via HTTP POST
                try:
                    print(f"post metrics = {metrics} to {self.metrics_server_url}")
                    response = requests.post(self.metrics_server_url, json=metrics)
                    response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    print(f"Failed to send metrics: {e}")

        except Exception as e:
            print(traceback.format_exc())
