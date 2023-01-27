# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
from nvflare.app_common.executors.init_final_component import InitFinalComponent
from nvflare.app_common.utils.component_utils import check_component_type


class ClientExecutor(InitFinalComponent, ABC):
    """
    ClientExecutor is to be used together with CommonExecutor,
    where most of the error handling, local component initialization exception handling, finalize are implemented.
    how sharable return to server by converting to DXO is also handled by Common Executor
    This class is focused on compute and return results only for given task
    """

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
        Returns:
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
    def client_exec(self, task_name: str, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        """
        Args:
            task_name: task name
            shareable: input from server
            fl_ctx: FLContext

        Returns: DataKind, Shareable Tuple
        """
        pass

    def finalize(self):
        if self.local_comp:
            self.local_comp.finalize()
        self.local_comp = None
