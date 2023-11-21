# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Any

from nvflare.apis.fl_exception import TaskExecutionError
from nvflare.apis.utils.fl_context_utils import generate_log_message
from nvflare.security.logging import secure_format_traceback

from .analytix import AnalyticsData, AnalyticsDataType
from .event_type import EventType
from .fl_constant import EventScope, FedEventHeader, FLContextKey, LogMessageTag
from .fl_context import FLContext
from .persistable import StatePersistable
from .shareable import Shareable


class FLComponent(StatePersistable):
    def __init__(self):
        """Init FLComponent.

        The FLComponent is the base class of all FL Components.
        (executors, controllers, responders, filters, aggregators, and widgets are all FLComponents)

        FLComponents have the capability to handle and fire events and contain various methods for logging.
        """
        self._name = self.__class__.__name__
        self.logger = logging.getLogger(self._name)

    @property
    def name(self):
        return self._name

    def _fire(self, event_type: str, fl_ctx: FLContext):
        fl_ctx.set_prop(FLContextKey.EVENT_ORIGIN, self._name, private=True, sticky=False)
        engine = fl_ctx.get_engine()
        if engine is None:
            self.log_error(fl_ctx=fl_ctx, msg="Logic Error: no engine in fl_ctx: {}".format(fl_ctx), fire_event=False)
        else:
            engine.fire_event(event_type, fl_ctx)

    def fire_event(self, event_type: str, fl_ctx: FLContext):
        """Fires an event.

        Args:
            event_type (str): The type of event.
            fl_ctx (FLContext): FLContext information.
        """
        if not isinstance(event_type, str):
            raise TypeError("expect event_type to be str, but got {}".format(type(event_type)))

        if not event_type:
            raise ValueError("event_type must be specified")

        if not isinstance(fl_ctx, FLContext):
            raise TypeError("expect fl_ctx to be FLContext, but got {}".format(type(fl_ctx)))

        fl_ctx.set_prop(FLContextKey.EVENT_SCOPE, value=EventScope.LOCAL, private=True, sticky=False)
        self._fire(event_type, fl_ctx)

    def fire_fed_event(self, event_type: str, event_data: Shareable, fl_ctx: FLContext, targets=None):
        """Fires a federation event.

        A federation event means that the event will be sent to different sites.
        For example, if fire a federation event on the server side, one can decide what clients to send via the
        parameter `targets`.
        If fire a federation event on the client side, the event will be sent to the server.

        Args:
            event_type (str): The type of event.
            event_data (Shareable): The data of this fed event.
            fl_ctx (FLContext): FLContext information.
            targets: The targets to send to. It is only used when fire federation event from server side.
        """
        if not isinstance(fl_ctx, FLContext):
            raise TypeError("expect fl_ctx to be FLContext, but got {}".format(type(fl_ctx)))

        if not isinstance(event_data, Shareable):
            raise TypeError("expect event_data to be Shareable, but got {}".format(type(event_data)))

        event_data.set_header(key=FedEventHeader.TARGETS, value=targets)
        fl_ctx.set_prop(FLContextKey.EVENT_DATA, event_data, private=True, sticky=False)
        fl_ctx.set_prop(FLContextKey.EVENT_SCOPE, value=EventScope.FEDERATION, private=True, sticky=False)
        self._fire(event_type, fl_ctx)

    def system_panic(self, reason: str, fl_ctx: FLContext):
        """Signals a fatal condition that could cause the RUN to end.

        Args:
            reason (str): The reason for panic.
            fl_ctx (FLContext): FLContext information.
        """
        fl_ctx.set_prop(FLContextKey.EVENT_DATA, reason, private=True, sticky=False)
        self.fire_event(EventType.FATAL_SYSTEM_ERROR, fl_ctx)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        """Handles events.

        Args:
            event_type (str): event type fired by workflow.
            fl_ctx (FLContext): FLContext information.
        """
        pass

    def log_info(self, fl_ctx: FLContext, msg: str, fire_event=False):
        """Logs a message with logger.info.

        These log_XXX methods are implemented because we want to have a unified way of logging messages.
        For example, in this method, we are using generate_log_message to add the FLContext information
        into the message. And we can decide whether to fire a log event afterwards.

        Args:
            fl_ctx (FLContext): FLContext information.
            msg (str): The message to log.
            fire_event (bool): Whether to fire a log event.
        """
        log_msg = generate_log_message(fl_ctx, msg)
        self.logger.info(log_msg)

        if fire_event:
            self._fire_log_event(
                event_type=EventType.INFO_LOG_AVAILABLE, log_tag=LogMessageTag.INFO, log_msg=log_msg, fl_ctx=fl_ctx
            )

    def log_warning(self, fl_ctx: FLContext, msg: str, fire_event=True):
        """Logs a message with logger.warning.

        Args:
            fl_ctx (FLContext): FLContext information.
            msg (str): The message to log.
            fire_event (bool): Whether to fire a log event.
        """
        log_msg = generate_log_message(fl_ctx, msg)
        self.logger.warning(log_msg)
        if fire_event:
            self._fire_log_event(
                event_type=EventType.WARNING_LOG_AVAILABLE,
                log_tag=LogMessageTag.WARNING,
                log_msg=log_msg,
                fl_ctx=fl_ctx,
            )

    def log_error(self, fl_ctx: FLContext, msg: str, fire_event=True):
        """Logs a message with logger.error.

        Args:
            fl_ctx (FLContext): FLContext information.
            msg (str): The message to log.
            fire_event (bool): Whether to fire a log event.
        """
        log_msg = generate_log_message(fl_ctx, msg)
        self.logger.error(log_msg)
        if fire_event:
            self._fire_log_event(
                event_type=EventType.ERROR_LOG_AVAILABLE, log_tag=LogMessageTag.ERROR, log_msg=log_msg, fl_ctx=fl_ctx
            )

    def log_debug(self, fl_ctx: FLContext, msg: str, fire_event=False):
        """Logs a message with logger.debug.

        Args:
            fl_ctx (FLContext): FLContext information.
            msg (str): The message to log.
            fire_event (bool): Whether to fire a log event.
        """
        log_msg = generate_log_message(fl_ctx, msg)
        self.logger.debug(log_msg)
        if fire_event:
            self._fire_log_event(
                event_type=EventType.DEBUG_LOG_AVAILABLE, log_tag=LogMessageTag.DEBUG, log_msg=log_msg, fl_ctx=fl_ctx
            )

    def log_critical(self, fl_ctx: FLContext, msg: str, fire_event=True):
        """Logs a message with logger.critical.

        Args:
            fl_ctx (FLContext): FLContext information.
            msg (str): The message to log.
            fire_event (bool): Whether to fire a log event.
        """
        log_msg = generate_log_message(fl_ctx, msg)
        self.logger.critical(log_msg)
        if fire_event:
            self._fire_log_event(
                event_type=EventType.CRITICAL_LOG_AVAILABLE,
                log_tag=LogMessageTag.CRITICAL,
                log_msg=log_msg,
                fl_ctx=fl_ctx,
            )

    def log_exception(self, fl_ctx: FLContext, msg: str, fire_event=False):
        """Logs exception message with logger.error.

        Args:
            fl_ctx (FLContext): FLContext information.
            msg (str): The message to log.
            fire_event (bool): Whether to fire a log event. Unused.
        """
        log_msg = generate_log_message(fl_ctx, msg)
        self.logger.error(log_msg)
        ex_text = secure_format_traceback()
        self.logger.error(ex_text)

        if fire_event:
            ex_msg = "{}\n{}".format(log_msg, ex_text)
            self._fire_log_event(
                event_type=EventType.EXCEPTION_LOG_AVAILABLE,
                log_tag=LogMessageTag.EXCEPTION,
                log_msg=ex_msg,
                fl_ctx=fl_ctx,
            )

    def _fire_log_event(self, event_type: str, log_tag: str, log_msg: str, fl_ctx: FLContext):
        if not fl_ctx:
            return

        event_data = AnalyticsData(key=log_tag, value=log_msg, data_type=AnalyticsDataType.TEXT, kwargs=None)
        dxo = event_data.to_dxo()
        fl_ctx.set_prop(key=FLContextKey.EVENT_DATA, value=dxo.to_shareable(), private=True, sticky=False)
        self.fire_event(event_type=event_type, fl_ctx=fl_ctx)


class FLComponentHelper(FLComponent):
    STATE = None

    def __init__(self):
        print("FLComponentHelper.init")
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
