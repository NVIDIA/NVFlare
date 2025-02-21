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
Randomly generate edge tasks
"""
import threading
import time
import uuid

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.edge.constants import EdgeContextKey, EdgeProtoKey
from nvflare.edge.constants import EventType as EdgeEventType
from nvflare.edge.constants import Status
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
    def _make_task():
        return {
            "device_id": str(uuid.uuid4()),
            "request_type": "getTask",
        }

    def _generate_tasks(self):
        caps = ["xgb", "llm"]
        while True:
            if self.abort_signal.triggered:
                self.logger.info("received abort signal - exiting")
                return

            with self.engine.new_context() as fl_ctx:
                assert isinstance(fl_ctx, FLContext)
                if not self.job_id:
                    fl_ctx.set_prop(EdgeContextKey.EDGE_CAPABILITIES, caps, private=True, sticky=False)
                    self.fire_event(EdgeEventType.EDGE_JOB_REQUEST_RECEIVED, fl_ctx)
                    result = fl_ctx.get_prop(EdgeContextKey.REPLY_TO_EDGE)
                    if result:
                        assert isinstance(result, dict)
                        status = result[EdgeProtoKey.STATUS]
                        job_id = result[EdgeProtoKey.DATA]
                        self.logger.info(f"job reply from ETD: {status=} {job_id=}")
                        if job_id:
                            self.job_id = job_id
                    else:
                        self.logger.error(f"no result from ETD for event {EdgeEventType.EDGE_JOB_REQUEST_RECEIVED}")
                else:
                    task = self._make_task()
                    fl_ctx.set_prop(EdgeContextKey.JOB_ID, self.job_id, sticky=False, private=True)
                    fl_ctx.set_prop(EdgeContextKey.REQUEST_FROM_EDGE, task, sticky=False, private=True)
                    self.fire_event(EdgeEventType.EDGE_REQUEST_RECEIVED, fl_ctx)
                    result = fl_ctx.get_prop(EdgeContextKey.REPLY_TO_EDGE)
                    if not result:
                        self.logger.error(f"no result from ETD for event {EdgeEventType.EDGE_REQUEST_RECEIVED}")
                    else:
                        status = result[EdgeProtoKey.STATUS]
                        edge_reply = result[EdgeProtoKey.DATA]
                        self.logger.info(f"task reply from ETD: {status=} {edge_reply=}")

                        if status == Status.NO_JOB:
                            # job already finished
                            self.job_id = None

            time.sleep(1.0)
