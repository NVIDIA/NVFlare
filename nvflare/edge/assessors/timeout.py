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

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.edge.assessor import Assessment, Assessor


class TimeoutAssessor(Assessor):
    def __init__(self, timeout: float):
        Assessor.__init__(self)
        self.timeout = timeout
        self._aggregator = None
        self._start_time = None

    def initialize(self, aggregator: Aggregator, fl_ctx: FLContext):
        self._aggregator = aggregator

    def start(self, fl_ctx: FLContext):
        self._start_time = time.time()

    def reset(self, fl_ctx: FLContext):
        self._start_time = None

    def assess(self, fl_ctx: FLContext) -> Assessment:
        if time.time() - self._start_time > self.timeout:
            return Assessment.TASK_DONE
        else:
            return Assessment.CONTINUE
