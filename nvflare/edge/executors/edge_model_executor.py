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
from typing import Any, Optional

from nvflare.apis.dxo import DXO, from_dict
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey
from nvflare.edge.constants import CookieKey, EdgeApiStatus, EdgeProtoKey, MsgKey
from nvflare.edge.executors.ete import EdgeTaskExecutor
from nvflare.edge.executors.hug import TaskInfo
from nvflare.edge.mud import BaseState, Device, ModelUpdate, StateUpdateReport
from nvflare.edge.updaters.emd import AggregatorFactory, EdgeModelUpdater
from nvflare.edge.web.models.job_request import JobRequest
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.result_report import ResultReport
from nvflare.edge.web.models.result_response import ResultResponse
from nvflare.edge.web.models.selection_request import SelectionRequest
from nvflare.edge.web.models.selection_response import SelectionResponse
from nvflare.edge.web.models.task_request import TaskRequest
from nvflare.edge.web.models.task_response import TaskResponse
from nvflare.security.logging import secure_format_exception


class EdgeModelExecutor(EdgeTaskExecutor):

    def __init__(self, aggr_factory_id: str, max_model_versions: int, update_timeout=60):
        EdgeTaskExecutor.__init__(self, "", update_timeout)
        self.aggr_factory_id = aggr_factory_id
        self.max_model_versions = max_model_versions

    def get_updater(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        aggr_factory = engine.get_component(self.aggr_factory_id)
        if not isinstance(aggr_factory, AggregatorFactory):
            self.system_panic(
                f"component {self.aggr_factory_id} should be AggregatorFactory but got {type(aggr_factory)}",
                fl_ctx,
            )
            return None

        return EdgeModelUpdater(aggr_factory, self.max_model_versions)

    def _convert_task(self, task_state: BaseState, current_task: TaskInfo, fl_ctx: FLContext) -> dict:
        """Convert task_data to a plain dict"""
        self.log_debug(fl_ctx, f"Converting task for task: {current_task.id}")

        # Add model version to the payload - WHY?
        model_dxo = task_state.model
        model_dxo.set_meta_prop(MsgKey.MODEL_VERSION, task_state.model_version)
        return model_dxo.to_dict()

    def _convert_device_result_to_model_update(
        self, result_report: ResultReport, current_task: TaskInfo, fl_ctx: FLContext
    ) -> Optional[ModelUpdate]:
        self.log_debug(fl_ctx, f"Converting result for task: {current_task.id}")

        assert isinstance(result_report.result, dict)
        dxo = from_dict(result_report.result)
        assert isinstance(dxo, DXO)
        dxo.set_meta_prop(ReservedHeaderKey.TASK_ID, current_task.id)

        device_id = result_report.get_device_id()
        cookie = result_report.cookie
        if not cookie:
            self.log_error(fl_ctx, f"missing cookie in result report from device {device_id}")
            raise ValueError("missing cookie")

        model_version = cookie.get(CookieKey.MODEL_VERSION)
        if not model_version:
            self.log_error(
                fl_ctx, f"missing '{CookieKey.MODEL_VERSION}' cookie in result report from device {device_id}"
            )
            raise ValueError(f"missing '{CookieKey.MODEL_VERSION}' cookie")

        return ModelUpdate(
            model_version=model_version,
            update=dxo.to_shareable(),
            devices={result_report.get_device_id(): time.time()},
        )

    def accept_alive_device(self, device_id: str, fl_ctx: FLContext):
        client_name = fl_ctx.get_identity_name()
        update_report = StateUpdateReport(
            current_model_version=0,
            current_device_selection_version=0,
            model_updates=None,
            available_devices={device_id: Device(device_id, client_name, time.time())},
        )
        return self.accept_update("", update_report.to_shareable(), fl_ctx)

    def accept_device_result(self, result_report: ResultReport, current_task: TaskInfo, fl_ctx: FLContext):
        client_name = fl_ctx.get_identity_name()
        device_id = result_report.get_device_id()
        model_update = self._convert_device_result_to_model_update(result_report, current_task, fl_ctx)
        update_report = StateUpdateReport(
            current_model_version=0,
            current_device_selection_version=0,
            model_updates={model_update.model_version: model_update},
            available_devices={device_id: Device(device_id, client_name, time.time())},
        )
        return self.accept_update(result_report.task_id, update_report.to_shareable(), fl_ctx)

    @staticmethod
    def _make_retry(job_id, msg: str):
        return TaskResponse(EdgeApiStatus.RETRY, job_id, 30, message=msg)

    @staticmethod
    def _make_cookie(model_version, device_selection_id):
        return {
            CookieKey.MODEL_VERSION: model_version,
            CookieKey.DEVICE_SELECTION_ID: device_selection_id,
        }

    def _handle_selection_request(
        self, request: SelectionRequest, current_task: TaskInfo, fl_ctx: FLContext
    ) -> SelectionResponse:
        """Handle selection request from device"""
        device_id = request.get_device_id()
        job_id = fl_ctx.get_job_id()

        self.accept_alive_device(device_id, fl_ctx)

        task_state = current_task.task
        assert isinstance(task_state, BaseState)

        if not task_state.model_version or not task_state.device_selection_version:
            # nothing to train
            return SelectionResponse(
                EdgeApiStatus.OK,
                job_id=job_id,
            )

        return SelectionResponse(
            status=EdgeApiStatus.OK,
            job_id=job_id,
            task_id=current_task.id,
            selection=task_state.device_selection,
        )

    def _handle_task_request(self, request: TaskRequest, current_task: TaskInfo, fl_ctx: FLContext) -> TaskResponse:
        """Handle task request from device"""

        device_id = request.get_device_id()
        job_id = fl_ctx.get_job_id()

        self.accept_alive_device(device_id, fl_ctx)

        task_state = current_task.task
        assert isinstance(task_state, BaseState)

        if not task_state.model_version:
            # nothing to train
            return self._make_retry(job_id, "Model not ready")

        cookie = request.cookie
        if cookie:
            device_selection_id = cookie.get(CookieKey.DEVICE_SELECTION_ID, 0)
        else:
            device_selection_id = 0

        selected, new_selection_id = task_state.is_device_selected(device_id, device_selection_id)
        if not selected:
            return self._make_retry(job_id, "Device not selected")

        self.log_debug(
            fl_ctx, f"task for model V{task_state.model_version} sent to device {device_id}: {new_selection_id=}"
        )
        task_data = self._convert_task(task_state, current_task, fl_ctx)
        return TaskResponse(
            status=EdgeApiStatus.OK,
            job_id=job_id,
            retry_wait=0,
            task_id=current_task.id,
            task_name=current_task.name,
            task_data=task_data,
            cookie=self._make_cookie(task_state.model_version, new_selection_id),
        )

    def _handle_result_report(self, report: ResultReport, current_task: TaskInfo, fl_ctx: FLContext) -> ResultResponse:
        """Handle result report from device
        The report task_id may be different from current task_id. Let HAM deal with it
        """

        try:
            if not report.result or report.status != EdgeApiStatus.OK:
                self.log_error(
                    fl_ctx,
                    f"no result or bad status ({report.status}) in report from device "
                    f"{report.get_device_id()} for task {report.task_id}",
                )
                return ResultResponse(
                    EdgeApiStatus.ERROR,
                    task_id=report.task_id,
                    task_name=report.task_name,
                    message="missing result or bad status",
                )
            else:
                self.accept_device_result(report, current_task, fl_ctx)
                return ResultResponse(EdgeApiStatus.OK, task_id=report.task_id, task_name=report.task_name)
        except Exception as ex:
            msg = f"Error accepting contribution: {secure_format_exception(ex)}"
            self.log_error(fl_ctx, msg)
            return ResultResponse(EdgeApiStatus.ERROR, task_id=report.task_id, task_name=report.task_name, message=msg)

    def task_started(self, task: TaskInfo, fl_ctx: FLContext):
        self.log_info(fl_ctx, f"Got task_started: {task.id} (seq {task.seq})")

    def task_ended(self, task: TaskInfo, fl_ctx: FLContext):
        self.log_info(fl_ctx, f"Got task_ended: {task.id} (seq {task.seq})")

    @staticmethod
    def _edge_response(status: str, resp):
        return {EdgeProtoKey.STATUS: status, EdgeProtoKey.RESPONSE: resp}

    def process_edge_request(self, request: Any, current_task: TaskInfo, fl_ctx: FLContext) -> Any:
        if isinstance(request, dict):
            self.log_debug(fl_ctx, f"Received edge request from device: {request.get('device_info')}")

        try:
            if isinstance(request, JobRequest):
                device_id = request.get_device_id()
                self.accept_alive_device(device_id, fl_ctx)

                response = JobResponse(
                    status=EdgeApiStatus.OK,
                    job_id=fl_ctx.get_job_id(),
                )
            elif isinstance(request, SelectionRequest):
                response = self._handle_selection_request(request, current_task, fl_ctx)
            elif isinstance(request, TaskRequest):
                response = self._handle_task_request(request, current_task, fl_ctx)
            elif isinstance(request, ResultReport):
                response = self._handle_result_report(request, current_task, fl_ctx)
            else:
                raise RuntimeError(f"Received unknown request type: {type(request)}")
            return self._edge_response(EdgeApiStatus.OK, response)
        except Exception as ex:
            self.log_exception(fl_ctx, f"exception processing edge request: {secure_format_exception(ex)}")
            return self._edge_response(EdgeApiStatus.ERROR, None)
