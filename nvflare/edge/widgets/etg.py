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

"""Edge Task Generator - for test only
Randomly generate edge tasks.
It should be installed to the CP.
"""
import threading
import time
import uuid

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.edge.constants import EdgeApiStatus, EdgeContextKey
from nvflare.edge.constants import EdgeEventType as EdgeEventType
from nvflare.edge.web.models.capabilities import Capabilities
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.job_request import JobRequest
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.task_request import TaskRequest
from nvflare.edge.web.models.task_response import TaskResponse
from nvflare.edge.web.models.user_info import UserInfo
from nvflare.widgets.widget import Widget


class EdgeTaskGenerator(Widget):
    def __init__(self):
        Widget.__init__(self)
        self.generator = None
        self.engine = None
        self.job_id = None
        self.abort_signal = Signal()
        self.logger.info("EdgeTaskGenerator created!")

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.SYSTEM_START:
            # start the generator
            self.logger.info("Starting generator ...")
            self.engine = fl_ctx.get_engine()
            self.generator = threading.Thread(target=self._generate_tasks, daemon=True)
            self.generator.start()
        elif event_type == EventType.SYSTEM_END:
            self.abort_signal.trigger(True)

    @staticmethod
    def _make_task(job_id: str) -> TaskRequest:
        return TaskRequest(
            job_id=job_id, device_info=DeviceInfo(device_id=str(uuid.uuid4())), user_info=UserInfo(), cookie={}
        )

    def _generate_tasks(self):
        caps = Capabilities(["xgb", "llm"])
        while True:
            if self.abort_signal.triggered:
                self.logger.info("received abort signal - exiting")
                return

            with self.engine.new_context() as fl_ctx:
                assert isinstance(fl_ctx, FLContext)
                if not self.job_id:
                    job_request = JobRequest(
                        device_info=DeviceInfo(device_id=str(uuid.uuid4())),
                        user_info=UserInfo(),
                        capabilities=caps,
                    )

                    self.logger.debug(f"trying to get job: {job_request}")
                    fl_ctx.set_prop(EdgeContextKey.REQUEST_FROM_EDGE, job_request, private=True, sticky=False)
                    self.fire_event(EdgeEventType.EDGE_JOB_REQUEST_RECEIVED, fl_ctx)
                    result = fl_ctx.get_prop(EdgeContextKey.REPLY_TO_EDGE)
                    self.logger.debug(f"job response received: {result}")

                    if result:
                        assert isinstance(result, JobResponse)
                        status = result.status
                        job_id = result.job_id
                        self.logger.debug(f"job reply from ETD: {status=} {job_id=}")
                        if job_id:
                            self.job_id = job_id
                    else:
                        self.logger.error(f"no result from ETD for event {EdgeEventType.EDGE_JOB_REQUEST_RECEIVED}")
                else:
                    task_req = self._make_task(job_id)
                    self.logger.debug(f"sending task request {task_req}")
                    fl_ctx.set_prop(EdgeContextKey.REQUEST_FROM_EDGE, task_req, sticky=False, private=True)
                    self.fire_event(EdgeEventType.EDGE_TASK_REQUEST_RECEIVED, fl_ctx)
                    result = fl_ctx.get_prop(EdgeContextKey.REPLY_TO_EDGE)
                    self.logger.debug(f"got task response {result}")
                    if not result:
                        self.logger.error(f"no result from ETD for event {EdgeEventType.EDGE_TASK_REQUEST_RECEIVED}")
                    else:
                        assert isinstance(result, TaskResponse)
                        status = result.status
                        self.logger.debug(f"task reply from ETD: {result}")
                        if status == EdgeApiStatus.NO_JOB:
                            # job already finished
                            self.job_id = None

            time.sleep(1.0)
