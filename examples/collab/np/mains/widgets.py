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
from nvflare.collab import collab
from nvflare.fuel.utils.log_utils import get_obj_logger


class MetricReceiver:

    def __init__(self):
        self.logger = get_obj_logger(self)

    @collab.publish
    def accept_metric(self, metrics: dict):
        self.logger.info(f"[{collab.callee}] received metric report from {collab.caller}: {metrics}")

    @collab.init
    def init(self):
        collab.register_event_handler("metrics", self._accept_metric)
        self.logger.info("MetricReceiver initialized!")

    def _accept_metric(self, event_type: str, data):
        self.logger.info(f"[{collab.callee}] received metrics event '{event_type}' from {collab.caller}: {data}")
        return "OK"
