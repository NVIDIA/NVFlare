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

from nvflare.apis.client_engine_spec import ClientEngineSpec
from nvflare.edge.web.models.job_request import JobRequest
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.result_report import ResultReport
from nvflare.edge.web.models.result_response import ResultResponse
from nvflare.edge.web.models.selection_request import SelectionRequest
from nvflare.edge.web.models.selection_response import SelectionResponse
from nvflare.edge.web.models.task_request import TaskRequest
from nvflare.edge.web.models.task_response import TaskResponse


class EdgeTaskHandler(ABC):
    @abstractmethod
    def set_engine(self, engine: ClientEngineSpec):
        pass

    @abstractmethod
    def handle_job(self, job_request: JobRequest) -> JobResponse:
        pass

    @abstractmethod
    def handle_task(self, task_request: TaskRequest) -> TaskResponse:
        pass

    @abstractmethod
    def handle_result(self, result_report: ResultReport) -> ResultResponse:
        pass

    @abstractmethod
    def handle_selection(self, selection_request: SelectionRequest) -> SelectionResponse:
        pass
