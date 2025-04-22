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
from abc import abstractmethod
from typing import Any

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.edge.constants import EdgeEventType
from nvflare.edge.executors.hug import HierarchicalUpdateGatherer, TaskInfo
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.security.logging import secure_format_exception


class EdgeTaskExecutor(HierarchicalUpdateGatherer):
    """This is the base class for building executors to manage federated learning on edge devices.
    Subclasses must implement the required abstract methods defined here.

    Note: This class is based on HUG (HierarchicalUpdateGatherer).
    All interactions with parent clients are already handled by HUG.
    """

    def __init__(
        self,
        updater_id: str,
        update_timeout: float,
        learner_id="",
    ):
        """Constructor of EdgeTaskExecutor.

        Args:
            updater_id: ID of the updater
            update_timeout: timeout for update messages sent to parent
            learner_id: ID of the learner component. Optional.
        """
        HierarchicalUpdateGatherer.__init__(
            self,
            updater_id=updater_id,
            update_timeout=update_timeout,
            learner_id=learner_id,
        )
        self.register_event_handler(EdgeEventType.EDGE_REQUEST_RECEIVED, self._handle_edge_request)

    @abstractmethod
    def process_edge_request(self, request: Any, current_task: TaskInfo, fl_ctx: FLContext) -> Any:
        """This is called to process an edge request sent from the edge device.

        Args:
            request: the request from edge device
            current_task: the current pending task
            fl_ctx: FLContext object

        Returns: reply to the edge device

        """
        pass

    def _handle_edge_request(self, event_type: str, fl_ctx: FLContext):
        task_info = self.get_current_task(fl_ctx)
        if not task_info:
            self.logger.debug(f"received edge event {event_type} but I don't have pending task")
            return

        try:
            msg = fl_ctx.get_prop(FLContextKey.CELL_MESSAGE)
            assert isinstance(msg, CellMessage)
            self.log_debug(fl_ctx, f"received edge request: {msg.payload}")
            reply = self.process_edge_request(request=msg.payload, fl_ctx=fl_ctx, current_task=task_info)
            fl_ctx.set_prop(FLContextKey.TASK_RESULT, reply, private=True, sticky=False)
        except Exception as ex:
            self.log_exception(fl_ctx, f"exception process_edge_request: {secure_format_exception(ex)}")
            fl_ctx.set_prop(FLContextKey.EXCEPTIONS, ex, private=True, sticky=False)
