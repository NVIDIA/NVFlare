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

import time

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.model import ModelLearnable
from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.fuel.utils.pipe.pipe import Pipe

from .defs import Topic, receive_from_pipe, send_to_pipe


class HubShareableGenerator(ShareableGenerator):
    def __init__(
        self, pipe_id: str, task_name: str = "train", task_wait_time=None, task_data_poll_interval: float = 0.5
    ):
        ShareableGenerator.__init__(self)
        self.pipe_id = pipe_id
        self.task_name = task_name
        self.task_wait_time = task_wait_time
        self.task_data_poll_interval = task_data_poll_interval
        self.pipe = None
        self.run_ended = False

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        if event_type == EventType.START_RUN:
            job_id = fl_ctx.get_job_id()
            self.pipe = engine.get_component(self.pipe_id)
            if not isinstance(self.pipe, Pipe):
                raise TypeError(f"pipe must be Pipe type. Got: {type(self.pipe)}")
            self.pipe.open(name=job_id, me="y")
        elif event_type == EventType.END_RUN:
            send_to_pipe(self.pipe, topic=Topic.END_RUN, data="")
            self.run_ended = True

    def learnable_to_shareable(self, model_learnable: ModelLearnable, fl_ctx: FLContext) -> Shareable:
        """Convert ModelLearnable to Shareable.

        Args:
            model_learnable (ModelLearnable): model to be converted
            fl_ctx (FLContext): FL context

        Returns:
            Shareable: a shareable containing a DXO object.
        """
        # try to get task data from T1
        start = time.time()
        while not self.run_ended:
            topic, data = receive_from_pipe(self.pipe)
            if not topic:
                if self.task_wait_time and time.time() - start > self.task_wait_time:
                    # timed out
                    self.log_error(fl_ctx, f"task data timeout after {self.task_wait_time} secs")
                    self.system_panic(reason="cannot get task data from T1", fl_ctx=fl_ctx)
                    break
            elif topic == Topic.END_RUN:
                self.system_panic(reason="received EndRun from T1 while waiting for task data", fl_ctx=fl_ctx)
                break
            elif topic == Topic.ABORT_TASK:
                self.system_panic(reason="received AbortTask from T1 while waiting for task data", fl_ctx=fl_ctx)
                break
            elif topic != self.task_name:
                # ignore wrong task name
                self.log_error(fl_ctx, f"ignored '{topic}' from T1 when waiting for '{self.task_name}'")
            else:
                self.log_info(fl_ctx, f"got data for task '{topic}' from T1")
                if not isinstance(data, Shareable):
                    self.system_panic(
                        reason=f"bad task data from T1 - must be Shareable but got {type(data)}", fl_ctx=fl_ctx
                    )
                    break
                return data
            time.sleep(self.task_data_poll_interval)
        return None

    def shareable_to_learnable(self, shareable: Shareable, fl_ctx: FLContext) -> ModelLearnable:
        self.log_info(fl_ctx, f"sent shareable to T1 for task {self.task_name}")
        send_to_pipe(self.pipe, topic=self.task_name, data=shareable)
        return ModelLearnable()
