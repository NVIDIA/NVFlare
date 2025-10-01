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
from nvflare.fox.api.ctx import Context
from nvflare.fox.api.dec import collab


class MetricReceiver:

    @collab
    def accept_metric(self, metrics: dict, context: Context):
        print(f"[{context.callee}] received metric report from {context.caller}: {metrics}")

    def initialize(self, context: Context):
        context.app.register_event_handler("metrics", self._accept_metric)
        print("MetricReceiver initialized!")

    def _accept_metric(self, event_type: str, data, context: Context):
        print(f"[{context.callee}] received event '{event_type}' from {context.caller}: {data}")
