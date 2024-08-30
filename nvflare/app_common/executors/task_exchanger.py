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
from typing import Optional

from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, FLMetaKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.fuel.utils.constants import PipeChannelName
from nvflare.fuel.utils.pipe.pipe import Message, Pipe
from nvflare.fuel.utils.pipe.pipe_handler import PipeHandler, Topic
from nvflare.fuel.utils.validation_utils import (
    check_non_negative_int,
    check_non_negative_number,
    check_positive_number,
    check_str,
)
from nvflare.security.logging import secure_format_exception


class TaskExchanger(Executor):
    def __init__(
        self,
        pipe_id: str,
        read_interval: float = 0.5,
        heartbeat_interval: float = 5.0,
        heartbeat_timeout: Optional[float] = 60.0,
        resend_interval: float = 2.0,
        max_resends: Optional[int] = None,
        peer_read_timeout: Optional[float] = 60.0,
        task_wait_time: Optional[float] = None,
        result_poll_interval: float = 0.5,
        pipe_channel_name=PipeChannelName.TASK,
    ):
        """Constructor of TaskExchanger.

        Args:
            pipe_id (str): component id of pipe.
            read_interval (float): how often to read from pipe.
            heartbeat_interval (float): how often to send heartbeat to peer.
            heartbeat_timeout (float, optional): how long to wait for a
                heartbeat from the peer before treating the peer as dead,
                0 means DO NOT check for heartbeat.
            resend_interval (float): how often to resend a message if failing to send.
                None means no resend. Note that if the pipe does not support resending,
                then no resend.
            max_resends (int, optional): max number of resend. None means no limit.
                Defaults to None.
            peer_read_timeout (float, optional): time to wait for peer to accept sent message.
            task_wait_time (float, optional): how long to wait for a task to complete.
                None means waiting forever. Defaults to None.
            result_poll_interval (float): how often to poll task result.
                Defaults to 0.5.
            pipe_channel_name: the channel name for sending task requests.
                Defaults to "task".
        """
        Executor.__init__(self)
        check_str("pipe_id", pipe_id)
        check_positive_number("read_interval", read_interval)
        check_positive_number("heartbeat_interval", heartbeat_interval)
        if heartbeat_timeout is not None:
            check_non_negative_number("heartbeat_timeout", heartbeat_timeout)
        check_positive_number("resend_interval", resend_interval)
        if max_resends is not None:
            check_non_negative_int("max_resends", max_resends)
        if peer_read_timeout is not None:
            check_positive_number("peer_read_timeout", peer_read_timeout)
        if task_wait_time is not None:
            check_positive_number("task_wait_time", task_wait_time)
        check_positive_number("result_poll_interval", result_poll_interval)
        check_str("pipe_channel_name", pipe_channel_name)

        self.pipe_id = pipe_id
        self.read_interval = read_interval
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.resend_interval = resend_interval
        self.max_resends = max_resends
        self.peer_read_timeout = peer_read_timeout
        self.task_wait_time = task_wait_time
        self.result_poll_interval = result_poll_interval
        self.pipe_channel_name = pipe_channel_name
        self.pipe = None
        self.pipe_handler = None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            engine = fl_ctx.get_engine()
            self.pipe = engine.get_component(self.pipe_id)
            if not isinstance(self.pipe, Pipe):
                self.system_panic(f"component of {self.pipe_id} must be Pipe but got {type(self.pipe)}", fl_ctx)
                return
            self.pipe_handler = PipeHandler(
                pipe=self.pipe,
                read_interval=self.read_interval,
                heartbeat_interval=self.heartbeat_interval,
                heartbeat_timeout=self.heartbeat_timeout,
                resend_interval=self.resend_interval,
                max_resends=self.max_resends,
            )
            self.pipe_handler.set_status_cb(self._pipe_status_cb)
            self.pipe.open(self.pipe_channel_name)
        elif event_type == EventType.BEFORE_TASK_EXECUTION:
            self.pipe_handler.start()
        elif event_type == EventType.ABOUT_TO_END_RUN:
            self.log_info(fl_ctx, "Stopping pipe handler")
            if self.pipe_handler:
                self.pipe_handler.notify_end("end_of_job")
                self.pipe_handler.stop()

    def _pipe_status_cb(self, msg: Message):
        self.logger.info(f"pipe status changed to {msg.topic}")
        self.pipe_handler.stop()

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """
        The TaskExchanger always sends the Shareable to the peer, and expects to receive a Shareable object from the
        peer. The peer can convert the Shareable object to whatever format that is best for its applications (e.g.
        DXO or FLModel object). Similarly, when submitting result, the peer must convert its result object to a
        Shareable object before sending it back to the TaskExchanger.

        This "late-binding" (binding of the Shareable object to an application-friendly object) strategy makes the
        TaskExchanger generic and can be reused for any applications (e.g. Shareable based, DXO based, or any custom
        data based).
        """
        if not self.check_input_shareable(task_name, shareable, fl_ctx):
            self.log_error(fl_ctx, "bad input task shareable")
            return make_reply(ReturnCode.BAD_TASK_DATA)

        shareable.set_header(FLMetaKey.JOB_ID, fl_ctx.get_job_id())
        shareable.set_header(FLMetaKey.SITE_NAME, fl_ctx.get_identity_name())
        task_id = shareable.get_header(key=FLContextKey.TASK_ID)

        # send to peer
        self.log_info(fl_ctx, f"sending task to peer {self.peer_read_timeout=}")
        req = Message.new_request(topic=task_name, data=shareable, msg_id=task_id)
        start_time = time.time()
        has_been_read = self.pipe_handler.send_to_peer(req, timeout=self.peer_read_timeout, abort_signal=abort_signal)
        if self.peer_read_timeout and not has_been_read:
            self.log_error(
                fl_ctx,
                f"peer does not accept task '{task_name}' in {time.time()-start_time} secs - aborting task!",
            )
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        self.log_info(fl_ctx, f"task {task_name} sent to peer in {time.time()-start_time} secs")

        # wait for result
        self.log_debug(fl_ctx, "Waiting for result from peer")
        start = time.time()
        while True:
            if abort_signal.triggered:
                # notify peer that the task is aborted
                self.log_debug(fl_ctx, f"task '{task_name}' is aborted.")
                self.pipe_handler.notify_abort(task_id)
                self.pipe_handler.stop()
                return make_reply(ReturnCode.TASK_ABORTED)

            if self.pipe_handler.asked_to_stop:
                self.log_debug(fl_ctx, "task pipe stopped!")
                self.pipe_handler.notify_abort(task_id)
                abort_signal.trigger("task pipe stopped!")
                return make_reply(ReturnCode.TASK_ABORTED)

            reply: Optional[Message] = self.pipe_handler.get_next()
            if reply is None:
                if self.task_wait_time and time.time() - start > self.task_wait_time:
                    # timed out
                    self.log_error(fl_ctx, f"task '{task_name}' timeout after {self.task_wait_time} secs")
                    # also tell peer to abort the task
                    self.pipe_handler.notify_abort(task_id)
                    abort_signal.trigger(f"task '{task_name}' timeout after {self.task_wait_time} secs")
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)
            elif reply.msg_type != Message.REPLY:
                self.log_warning(
                    fl_ctx, f"ignored reply: '{reply}' (wrong message type) while waiting for the result of {task_name}"
                )
            elif req.topic != reply.topic:
                # ignore wrong topic
                self.log_warning(
                    fl_ctx,
                    f"ignored reply: '{reply}' (reply topic does not match req: '{req}') while waiting for the result of {task_name}",
                )
            elif req.msg_id != reply.req_id:
                self.log_warning(
                    fl_ctx,
                    f"ignored reply: '{reply}' (reply req_id does not match req msg_id: '{req}') while waiting for the result of {task_name}",
                )
            else:
                self.log_info(fl_ctx, f"got result '{reply}' for task '{task_name}'")

                try:
                    result = reply.data
                    if not isinstance(result, Shareable):
                        self.log_error(fl_ctx, f"bad task result from peer: expect Shareable but got {type(result)}")
                        return make_reply(ReturnCode.EXECUTION_EXCEPTION)

                    current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
                    if current_round is not None:
                        result.set_header(AppConstants.CURRENT_ROUND, current_round)

                    if not self.check_output_shareable(task_name, result, fl_ctx):
                        self.log_error(fl_ctx, "bad task result from peer")
                        return make_reply(ReturnCode.EXECUTION_EXCEPTION)

                    self.log_info(fl_ctx, f"received result of {task_name} from peer in {time.time()-start} secs")
                    return result
                except Exception as ex:
                    self.log_error(fl_ctx, f"Failed to convert result: {secure_format_exception(ex)}")
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)
            time.sleep(self.result_poll_interval)

    def check_input_shareable(self, task_name: str, shareable: Shareable, fl_ctx: FLContext) -> bool:
        """Checks input shareable before execute.

        Returns:
            True, if input shareable looks good; False, otherwise.
        """
        return True

    def check_output_shareable(self, task_name: str, shareable: Shareable, fl_ctx: FLContext) -> bool:
        """Checks output shareable after execute.

        Returns:
            True, if output shareable looks good; False, otherwise.
        """
        return True

    def ask_peer_to_end(self, fl_ctx: FLContext) -> bool:
        req = Message.new_request(topic=Topic.END, data="END")
        has_been_read = self.pipe_handler.send_to_peer(req, timeout=self.peer_read_timeout)
        if self.peer_read_timeout and not has_been_read:
            self.log_warning(
                fl_ctx,
                f"3rd party does not read END msg in {self.peer_read_timeout} secs!",
            )
            return False
        return True

    def peer_is_up_or_dead(self) -> bool:
        return self.pipe_handler.peer_is_up_or_dead.is_set()

    def pause_pipe_handler(self):
        """Stops pipe_handler heartbeat."""
        self.pipe_handler.pause()

    def resume_pipe_handler(self):
        """Resumes pipe_handler heartbeat."""
        self.pipe_handler.resume()

    def get_pipe(self):
        """Gets pipe."""
        return self.pipe

    def get_pipe_channel_name(self):
        """Gets pipe_channel_name."""
        return self.pipe_channel_name
