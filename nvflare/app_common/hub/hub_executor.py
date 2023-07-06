# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.fuel.utils.pipe.pipe import Message, Pipe
from nvflare.fuel.utils.pipe.pipe_handler import PipeHandler, Topic
from nvflare.fuel.utils.validation_utils import check_positive_number, check_str


class HubExecutor(Executor):
    """
    This executor is to be used by Tier-1 (T1) clients.
    It exchanges task data/result with the Hub Controller of Tier-2 (T2) Server
    """

    def __init__(
        self, pipe_id: str, task_wait_time=None, result_poll_interval: float = 0.1, task_read_wait_time: float = 10.0
    ):
        """
        Args:
            pipe_id:
            task_wait_time: how long to wait for result from T2
            result_poll_interval: polling interval for T2 result
            task_read_wait_time: how long to wait for T2 to read a task assignment
        """
        Executor.__init__(self)
        check_str("pipe_id", pipe_id)
        if task_wait_time is not None:
            check_positive_number("task_wait_time", task_wait_time)
        check_positive_number("result_poll_interval", result_poll_interval)
        check_positive_number("task_read_wait_time", task_read_wait_time)

        self.pipe_id = pipe_id
        self.task_wait_time = task_wait_time
        self.result_poll_interval = result_poll_interval
        self.task_read_wait_time = task_read_wait_time
        self.task_seq_num = 0
        self.t2_ended = False
        self.pipe_handler = None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        if event_type == EventType.START_RUN:
            job_id = fl_ctx.get_job_id()
            pipe: Pipe = engine.get_component(self.pipe_id)
            if not isinstance(pipe, Pipe):
                raise TypeError(f"pipe must be Pipe type. Got: {type(pipe)}")
            pipe.open(name=job_id)
            self.pipe_handler = PipeHandler(pipe)
            self.pipe_handler.start()
        elif event_type == EventType.END_RUN:
            # tell T2 system to end run
            self.log_info(fl_ctx, "END_RUN received - telling T2 to stop")
            self.pipe_handler.notify_end("END_RUN received")
            self.pipe_handler.stop()

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        contrib_round = shareable.get_cookie(AppConstants.CONTRIBUTION_ROUND)
        if contrib_round is None:
            self.log_warning(fl_ctx, "CONTRIBUTION_ROUND Not Set in task data!")

        # send the task to T2
        task_id = shareable.get_header(ReservedHeaderKey.TASK_ID)
        self.log_info(fl_ctx, f"sending task data to T2 for task {task_name}")
        req = Message.new_request(topic=task_name, data=shareable)
        task_received_by_t2 = self.pipe_handler.send_to_peer(req, timeout=self.task_read_wait_time)
        if not task_received_by_t2:
            self.log_error(
                fl_ctx, f"T2 failed to read task '{task_name}' in {self.task_read_wait_time} secs - aborting task!"
            )
            return make_reply(ReturnCode.SERVICE_UNAVAILABLE)

        # wait for result from T2
        start = time.time()
        while True:
            if abort_signal.triggered:
                # notify T2 that the task is aborted
                self.pipe_handler.notify_abort(task_id)
                return make_reply(ReturnCode.TASK_ABORTED)

            reply = self.pipe_handler.get_next()
            if not reply:
                if self.task_wait_time and time.time() - start > self.task_wait_time:
                    # timed out
                    self.log_error(fl_ctx, f"task '{task_name}' timeout after {self.task_wait_time} secs")
                    # also tell T2 to abort the task
                    self.pipe_handler.notify_abort(task_id)
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)
            elif reply.topic == Topic.ABORT:
                # T2 told us to abort the task!
                return make_reply(ReturnCode.TASK_ABORTED)
            elif reply.topic in [Topic.END, Topic.PEER_GONE]:
                # T2 told us it has ended the run
                self.log_error(fl_ctx, f"received {reply.topic} from T2 while waiting for result for {task_name}")
                return make_reply(ReturnCode.SERVICE_UNAVAILABLE)
            elif reply.msg_type != Message.REPLY:
                self.log_warning(
                    fl_ctx, f"ignored msg '{reply.topic}.{reply.req_id}' when waiting for '{req.topic}.{req.msg_id}'"
                )
            elif req.topic != reply.topic:
                # ignore wrong task name
                self.log_warning(fl_ctx, f"ignored '{reply.topic}' when waiting for '{req.topic}'")
            elif req.msg_id != reply.req_id:
                self.log_warning(fl_ctx, f"ignored '{reply.req_id}' when waiting for '{req.msg_id}'")
            else:
                self.log_info(fl_ctx, f"got result for request '{task_name}' from T2")
                if not isinstance(reply.data, Shareable):
                    self.log_error(fl_ctx, f"bad result data from T2 - must be Shareable but got {type(reply.data)}")
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)
                # add important meta information
                current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
                if current_round:
                    reply.data.set_header(AppConstants.CURRENT_ROUND, current_round)
                return reply.data
            time.sleep(self.result_poll_interval)
