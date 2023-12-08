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

from typing import Any

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import TaskExecutionError


class FLComponentWrapper(FLComponent):
    STATE = None

    def __init__(self):
        super().__init__()
        self.engine = None
        self.fl_ctx = None
        self.workspace = None
        self.shareable = None
        self.args = None
        self.site_name = None
        self.job_id = None
        self.app_root = None
        self.job_root = None
        self.workspace_root = None
        self.abort_signal = None
        self.current_round = 0
        self.total_rounds = 0

    def is_aborted(self) -> bool:
        """Check whether the task has been asked to abort by the framework.

        Returns: whether the task has been asked to abort by the framework

        """
        return self.abort_signal and self.abort_signal.triggered

    def get_shareable_header(self, key: str, default=None):
        """Convenience method for getting specified header from the shareable.

        Args:
            key: name of the header
            default: default value if the header doesn't exist

        Returns: value of the header if it exists in the shareable; or the specified default if it doesn't.

        """
        if not self.shareable:
            return default
        return self.shareable.get_header(key, default)

    def get_context_prop(self, key: str, default=None):
        """Convenience method for getting specified property from the FL Context.

        Args:
            key: name of the property
            default: default value if the prop doesn't exist in FL Context

        Returns: value of the prop if it exists in the context; or the specified default if it doesn't.

        """
        if not self.fl_ctx:
            return default
        assert isinstance(self.fl_ctx, FLContext)
        return self.fl_ctx.get_prop(key, default)

    def get_component(self, component_id: str) -> Any:
        """Get the specified component from the context

        Args:
            component_id: ID of the component

        Returns: the specified component if it is defined; or None if not.

        """
        if self.engine:
            return self.engine.get_component(component_id)
        else:
            return None

    def debug(self, msg: str):
        """Convenience method for logging a DEBUG message with contextual info

        Args:
            msg: the message to be logged

        Returns:

        """
        self.log_debug(self.fl_ctx, msg)

    def info(self, msg: str):
        """Convenience method for logging an INFO message with contextual info

        Args:
            msg: the message to be logged

        Returns:

        """
        self.log_info(self.fl_ctx, msg)

    def error(self, msg: str):
        """Convenience method for logging an ERROR message with contextual info

        Args:
            msg: the message to be logged

        Returns:

        """
        self.log_error(self.fl_ctx, msg)

    def warning(self, msg: str):
        """Convenience method for logging a WARNING message with contextual info

        Args:
            msg: the message to be logged

        Returns:

        """
        self.log_warning(self.fl_ctx, msg)

    def exception(self, msg: str):
        """Convenience method for logging an EXCEPTION message with contextual info

        Args:
            msg: the message to be logged

        Returns:

        """
        self.log_exception(self.fl_ctx, msg)

    def critical(self, msg: str):
        """Convenience method for logging a CRITICAL message with contextual info

        Args:
            msg: the message to be logged

        Returns:

        """
        self.log_critical(self.fl_ctx, msg)

    def stop_task(self, reason: str):
        """Stop the current task.
        This method is to be called by the Learner's training or validation code when it runs into
        a situation that the task processing cannot continue.

        Args:
            reason: why the task cannot continue

        Returns:

        """
        self.log_error(self.fl_ctx, f"Task stopped: {reason}")
        raise TaskExecutionError(reason)

    def initialize(self):
        """Called by the framework to initialize the Learner object.
        This is called before the Learner can train or validate.
        This is called only once.

        """
        pass

    def abort(self):
        """Called by the framework for the Learner to gracefully abort the current task.

        This could be caused by multiple reasons:
        - user issued the abort command to stop the whole job
        - Controller runs into some condition that requires the job to be aborted
        """
        pass

    def finalize(self):
        """Called by the framework to finalize the Learner (close/release resources gracefully) when
        the job is finished.

        After this call, the Learner will be destroyed.

        Args:

        """
        pass

    def event(self, event_type):
        """Fires an event.

        Args:
            event_type (str): The type of event.
        """
        self.fire_event(event_type, self.fl_ctx)

    def panic(self, reason: str):
        """Signals a fatal condition that could cause the RUN to end.

        Args:
            reason (str): The reason for panic.
        """
        self.system_panic(reason, self.fl_ctx)
