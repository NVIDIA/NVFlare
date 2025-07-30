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
from typing import Optional

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.edge.constants import EdgeApiStatus, EdgeContextKey, EdgeEventType
from nvflare.edge.executors.hug import HierarchicalUpdateGatherer, TaskInfo
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.result_report import ResultReport
from nvflare.edge.web.models.result_response import ResultResponse
from nvflare.edge.web.models.selection_request import SelectionRequest
from nvflare.edge.web.models.selection_response import SelectionResponse
from nvflare.edge.web.models.task_request import TaskRequest
from nvflare.edge.web.models.task_response import TaskResponse


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
        self.register_event_handler(
            EdgeEventType.EDGE_TASK_REQUEST_RECEIVED,
            self._handle_edge_request,
            process_f=self.process_edge_task_request,
            no_task_reply=TaskResponse(EdgeApiStatus.RETRY),
        )
        self.register_event_handler(
            EdgeEventType.EDGE_SELECTION_REQUEST_RECEIVED,
            self._handle_edge_request,
            process_f=self.process_edge_selection_request,
            no_task_reply=SelectionResponse(EdgeApiStatus.RETRY),
        )
        self.register_event_handler(
            EdgeEventType.EDGE_RESULT_REPORT_RECEIVED,
            self._handle_edge_request,
            process_f=self.process_edge_result_report,
            no_task_reply=ResultResponse(EdgeApiStatus.OK),
        )
        self.register_event_handler(
            EdgeEventType.EDGE_JOB_REQUEST_RECEIVED,
            self._handle_edge_job_request,
        )

    @abstractmethod
    def process_edge_task_request(
        self, request: TaskRequest, current_task: TaskInfo, fl_ctx: FLContext
    ) -> Optional[TaskResponse]:
        """This is called to process an edge task request sent from the edge device.

        Args:
            request: the request from edge device
            current_task: the current pending task
            fl_ctx: FLContext object

        Returns: reply to the edge device

        """
        pass

    @abstractmethod
    def process_edge_selection_request(
        self, request: SelectionRequest, current_task: TaskInfo, fl_ctx: FLContext
    ) -> Optional[SelectionResponse]:
        """This is called to process an edge selection request sent from the edge device.

        Args:
            request: the request from edge device
            current_task: the current pending task
            fl_ctx: FLContext object

        Returns: reply to the edge device

        """
        pass

    @abstractmethod
    def process_edge_result_report(
        self, request: ResultReport, current_task: TaskInfo, fl_ctx: FLContext
    ) -> Optional[ResultResponse]:
        """This is called to process an edge result report sent from the edge device.

        Args:
            request: the request from edge device
            current_task: the current pending task
            fl_ctx: FLContext object

        Returns: reply to the edge device

        """
        pass

    def _handle_edge_request(self, event_type: str, fl_ctx: FLContext, process_f, no_task_reply):
        task_info = self.get_current_task(fl_ctx)
        if not task_info:
            self.log_debug(fl_ctx, f"received edge event {event_type} but I don't have pending task")
            reply = no_task_reply
        else:
            request = fl_ctx.get_prop(EdgeContextKey.REQUEST_FROM_EDGE)
            self.log_debug(fl_ctx, f"received edge request: {request}")
            reply = process_f(request=request, fl_ctx=fl_ctx, current_task=task_info)

        self.log_debug(fl_ctx, f"Reply to edge: {reply}")
        fl_ctx.set_prop(EdgeContextKey.REPLY_TO_EDGE, reply, private=True, sticky=False)

    def _handle_edge_job_request(self, event_type: str, fl_ctx: FLContext):
        # This is only used to process job requests from embedded simulator.
        # We do not send device config to simulators.
        job_meta = fl_ctx.get_prop(FLContextKey.JOB_META)
        job_name = job_meta.get(JobMetaKey.JOB_NAME)
        job_id = fl_ctx.get_job_id()

        # job_data is empty for now since we do not need to send device config to the simulator.
        reply = JobResponse(EdgeApiStatus.OK, job_id, job_name, job_data={})
        self.log_debug(fl_ctx, f"Reply to edge: {reply}")
        fl_ctx.set_prop(EdgeContextKey.REPLY_TO_EDGE, reply, private=True, sticky=False)
