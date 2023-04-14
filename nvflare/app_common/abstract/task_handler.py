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

from abc import ABC, abstractmethod
from typing import Optional

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.init_final_component import InitFinalComponent
from nvflare.app_common.utils.component_utils import check_component_type


class TaskHandler(InitFinalComponent, ABC):
    """TaskHandler focuses on computing and returning results only for given task."""

    def __init__(self, local_comp_id: str, local_comp_type: type):
        super().__init__()
        self.fl_ctx = None
        self.local_comp_id = local_comp_id
        self.local_comp: Optional[InitFinalComponent] = None
        self.target_local_comp_type: type = local_comp_type

    def initialize(self, fl_ctx: FLContext):
        """
        This is called when client is start Run. At this point
        the server hasn't communicated to the local component yet.
        Args:
            fl_ctx: fl_ctx: FLContext of the running environment
        """
        self.fl_ctx = fl_ctx
        self.load_and_init_local_comp(fl_ctx)

    def load_and_init_local_comp(self, fl_ctx):
        engine = fl_ctx.get_engine()
        local_comp: InitFinalComponent = engine.get_component(self.local_comp_id)
        check_component_type(local_comp, self.target_local_comp_type)
        local_comp.initialize(fl_ctx)
        self.local_comp = local_comp

    @abstractmethod
    def execute_task(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """Executes a task.

        Args:
            task_name: task name
            shareable: input data
            fl_ctx: FLContext
            abort_signal (Signal): signal to check during execution to determine whether this task is aborted.

        Returns:
            Output data
        """
        pass

    def finalize(self, fl_ctx: FLContext):
        if self.local_comp:
            self.local_comp.finalize(fl_ctx)
        self.local_comp = None
