# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Union, List, Optional, Dict

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import Task
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.client_controller import ClientController
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.app_constant import AppConstants


class ClientScatterAndGather(FLComponent):

    def __init__(self,
                 aggregator_id=AppConstants.DEFAULT_AGGREGATOR_ID
                 ):
        self.aggregator_id = aggregator_id

        self.aggregator = None
        self.controller = ClientController()

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.SYSTEM_START:
            self.initialize(fl_ctx)

    def initialize(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        self.aggregator = engine.get_component(self.aggregator_id)
        if not isinstance(self.aggregator, Aggregator):
            self.system_panic(
                f"aggregator {self.aggregator_id} must be an Aggregator type object but got {type(self.aggregator)}",
                fl_ctx,
            )

        self.controller.start_controller(fl_ctx)

    def broadcast_tasks(
        self,
        task_name: str,
        task_input: Shareable,
        fl_ctx: FLContext,
        targets: Union[List[Client], List[str], None] = None,
        task_props: Optional[Dict] = None,
        timeout=0
        ) -> Shareable:

        task = Task(name=task_name, data=task_input, props=task_props, timeout=timeout)
        results = self.controller.broadcast_and_wait(
            task=task,
            fl_ctx=fl_ctx,
            targets=targets,
        )

        return self._aggregate_results(results, fl_ctx)

    def _aggregate_results(self, results, fl_ctx):
        for _, result in results.items():
            self.aggregator.accept(result, fl_ctx)
        final_result = self.aggregator.aggregate(fl_ctx)
        return final_result
