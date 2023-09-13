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
from nvflare.fuel.utils.pipe.pipe import Message
from nvflare.fuel.utils.pipe.pipe_handler import PipeHandler


class MetricData:
    def __init__(self, key, value, data_type: AnalyticsDataType, additional_args=None):
        self.key = key
        self.value = value
        self.data_type = data_type
        self.additional_args = {} if additional_args is None else additional_args


class MetricsExchanger:
    def __init__(
        self,
        pipe_handler: PipeHandler,
        topic: str = "metrics",
    ):
        self._pipe_handler = pipe_handler
        self._topic = topic

    def log(self, key: str, value: Any, data_type: AnalyticsDataType, **kwargs):
        data = MetricData(key=key, value=value, data_type=data_type, additional_args=kwargs)
        req = Message.new_request(topic=self._topic, data=data)
        self._pipe_handler.send_to_peer(req)
