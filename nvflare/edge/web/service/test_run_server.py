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
import logging
import threading
import time
import uuid

from nvflare.edge.constants import CookieKey, EdgeApiStatus
from nvflare.edge.web.models.job_request import JobRequest
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.result_report import ResultReport
from nvflare.edge.web.models.result_response import ResultResponse
from nvflare.edge.web.models.selection_request import SelectionRequest
from nvflare.edge.web.models.selection_response import SelectionResponse
from nvflare.edge.web.models.task_request import TaskRequest
from nvflare.edge.web.models.task_response import TaskResponse
from nvflare.edge.web.service.query_handler import QueryHandler
from nvflare.edge.web.service.server import EdgeApiServer
from nvflare.fuel.utils.log_utils import get_obj_logger


class TestQueryHandler(QueryHandler):

    def __init__(self):
        QueryHandler.__init__(self)
        self.logger = get_obj_logger(self)

    def handle_job_request(self, request: JobRequest) -> JobResponse:
        device_id = request.get_device_id()
        self.logger.info(f"received job request from device {device_id}")
        time.sleep(5.0)
        return JobResponse(status=EdgeApiStatus.OK, job_id=str(uuid.uuid4()), job_name="test")

    def handle_task_request(self, request: TaskRequest) -> TaskResponse:
        cookie = {
            CookieKey.MODEL_VERSION: 1,
            CookieKey.DEVICE_SELECTION_ID: 12,
        }
        return TaskResponse(
            status=EdgeApiStatus.RETRY,
            job_id=request.job_id,
            cookie=cookie,
        )

    def handle_result_report(self, request: ResultReport) -> ResultResponse:
        return ResultResponse(
            status=EdgeApiStatus.OK,
            message="got it",
            task_id=request.task_id,
            task_name=request.task_name,
        )

    def handle_selection_request(self, request: SelectionRequest) -> SelectionResponse:
        return SelectionResponse(
            status=EdgeApiStatus.OK,
            job_id=request.job_id,
            selection={
                "aaaaa": 2,
                "bbbbb": 10,
            },
        )


def shutdown_server(server):
    run_duration = 60
    print(f"NOTE: server will only run for {run_duration} seconds!")
    time.sleep(run_duration)
    print(f"stopping server after {run_duration} seconds")
    server.shutdown()
    print("stopped server")


def main():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    server = EdgeApiServer(address="127.0.0.1:8009", handler=TestQueryHandler(), max_workers=100)
    t = threading.Thread(target=shutdown_server, daemon=True, args=(server,))
    t.start()
    server.start()


if __name__ == "__main__":
    main()
