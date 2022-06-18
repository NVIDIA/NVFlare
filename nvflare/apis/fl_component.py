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

import logging
import traceback

from nvflare.apis.utils.fl_context_utils import generate_log_message

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

    def task_panic(self, reason: str, fl_ctx: FLContext):
        """Signals a fatal condition that could cause the current task (on Client) to end.

        Args:
            reason (str): The reason for panic.
            fl_ctx (FLContext): FLContext information.
        """
        fl_ctx.set_prop(FLContextKey.EVENT_DATA, reason, private=True, sticky=False)
        self.fire_event(EventType.FATAL_TASK_ERROR, fl_ctx)

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
        traceback.print_exc()

        if fire_event:
            ex_text = traceback.format_exc()
            ex_msg = "{}\n{}".format(log_msg, ex_text)
            self._fire_log_event(
                event_type=EventType.EXCEPTION_LOG_AVAILABLE,
                log_tag=LogMessageTag.EXCEPTION,
                log_msg=ex_msg,
                fl_ctx=fl_ctx,
            )

    def _fire_log_event(self, event_type: str, log_tag: str, log_msg: str, fl_ctx: FLContext):
        event_data = AnalyticsData(tag=log_tag, value=log_msg, data_type=AnalyticsDataType.TEXT, kwargs=None)
        dxo = event_data.to_dxo()
        fl_ctx.set_prop(key=FLContextKey.EVENT_DATA, value=dxo.to_shareable(), private=True, sticky=False)
        self.fire_event(event_type=event_type, fl_ctx=fl_ctx)
