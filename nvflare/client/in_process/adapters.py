# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Internal adapters for job and task lifetime of the shared in-process API."""

from nvflare.app_common.app_constant import AppConstants

TOPIC_LOG_DATA = "LOG_DATA"
TOPIC_STOP = "STOP"
TOPIC_ABORT = "ABORT"
TOPIC_LOCAL_RESULT = "LOCAL_RESULT"
TOPIC_GLOBAL_RESULT = "GLOBAL_RESULT"
TASK_RESULT_STOP_REASON = "task result handed off"


class JobInProcessAdapter:
    def bind(self, api, receive_callback, abort_callback):
        self.api = api
        self._receive_callback = receive_callback
        self._abort_callback = abort_callback
        api.data_bus.subscribe([TOPIC_GLOBAL_RESULT], receive_callback)
        api.data_bus.subscribe([TOPIC_ABORT, TOPIC_STOP], abort_callback)

    def publish_result(self, shareable):
        self.api.event_manager.fire_event(TOPIC_LOCAL_RESULT, shareable)

    def publish_log(self, message):
        self.api.event_manager.fire_event(TOPIC_LOG_DATA, message)

    def close(self):
        self.api.data_bus.unsubscribe(TOPIC_GLOBAL_RESULT, self._receive_callback)
        self.api.data_bus.unsubscribe(TOPIC_ABORT, self._abort_callback)
        self.api.data_bus.unsubscribe(TOPIC_STOP, self._abort_callback)


class TaskInProcessAdapter:
    def __init__(self, on_result=None, log_handler=None):
        self._on_result = on_result
        self._log_handler = log_handler
        self._result = None
        self._current_round = None

    def bind(self, api, receive_callback, abort_callback):
        self.api = api

    def stage_input(self, shareable):
        self._current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
        self.api._set_received_shareable(shareable)

    def publish_result(self, shareable):
        if self._result is not None:
            raise RuntimeError("a task worker may send exactly one result")
        if self._current_round is not None:
            shareable.set_header(AppConstants.CURRENT_ROUND, self._current_round)
        if self._on_result is None:
            raise RuntimeError("task result writer is not configured")
        # The writer freezes the values before send() returns. Retain only its
        # storage reference, never another copy of the script's model.
        self._result = self._on_result(shareable)
        self.api.stop = True
        self.api.stop_reason = TASK_RESULT_STOP_REASON

    def publish_log(self, message):
        if self._log_handler is not None:
            self._log_handler(message)

    def get_result(self):
        if self._result is None:
            raise RuntimeError("Client API task script exited without sending a result")
        return self._result

    def close(self):
        pass
