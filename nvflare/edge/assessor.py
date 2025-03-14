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
from abc import ABC, abstractmethod
from enum import Enum

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.aggregator import Aggregator


class AssessResult(Enum):
    CONTINUE = "continue"
    TASK_DONE = "task_done"
    JOB_DONE = "job_done"


class Assessor(FLComponent, ABC):
    def __init__(self):
        FLComponent.__init__(self)

    @abstractmethod
    def initialize(self, aggregator: Aggregator, fl_ctx: FLContext):
        pass

    @abstractmethod
    def assess(self, fl_ctx: FLContext) -> AssessResult:
        pass

    def start(self, fl_ctx: FLContext):
        pass

    def reset(self, fl_ctx: FLContext):
        pass

    def finalize(self, fl_ctx: FLContext):
        pass
