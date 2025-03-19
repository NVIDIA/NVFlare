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


class Assessment(Enum):
    CONTINUE = "continue"
    TASK_DONE = "task_done"
    WORKFLOW_DONE = "workflow_done"


class Assessor(FLComponent, ABC):
    """Assessor is a component used by SAGE (ScatterAndGatherForEdge) workflow controller to assess the quality
    of the current task, and decides whether to continue the execution of the current task.

    """

    def __init__(self):
        FLComponent.__init__(self)

    @abstractmethod
    def initialize(self, aggregator: Aggregator, fl_ctx: FLContext):
        """This is called by SAGE before starting the workflow.

        Args:
            aggregator: the Aggregator used by the workflow.
            fl_ctx: FLContext object

        Returns: None

        """
        pass

    @abstractmethod
    def assess(self, fl_ctx: FLContext) -> Assessment:
        """This is called by SAGE to assess the situation of the current task, and decides whether the task should
        continue to run.

        Note: the fl_ctx contains task info that could be used to make assessment.

        Args:
            fl_ctx: FLContext object

        Returns: an Assessment value.

        """
        pass

    def start(self, fl_ctx: FLContext):
        """This is called by SAGE at the start of a task.

        Args:
            fl_ctx: FLContext object

        Returns: None

        """
        pass

    def reset(self, fl_ctx: FLContext):
        """This is called by SAGE at the end of a task.

        Args:
            fl_ctx: FLContext object

        Returns: None

        """
        pass

    def finalize(self, fl_ctx: FLContext):
        """Called at the end of the SAGE workflow to finalize the assessor.

        Args:
            fl_ctx: FLContext object

        Returns: None

        """
        pass
