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
from typing import Any, List, Union

from nvflare.apis.utils.fl_context_utils import generate_log_message
from nvflare.fuel.utils.log_utils import get_obj_logger
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
        self.logger = get_obj_logger(self)
        self._event_handlers = {}

    def _self_check(self):
        # This is used to dynamically construct all required elements of FLComponent.
        # We try to make it work for subclasses that fail to call super().__init__(), due to bad programming.
        if not hasattr(self, "_name"):
            self._name = self.__class__.__name__

        if not hasattr(self, "logger"):
            self.logger = get_obj_logger(self)

        if not hasattr(self, "_event_handlers"):
            self._event_handlers = {}

    @property
    def name(self):
        self._self_check()
        return self._name

    def _fire(self, event_type: str, fl_ctx: FLContext):
        self._self_check()
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

    def fire_event_with_data(self, event_type: str, fl_ctx: FLContext, key: str, data: Any):
        """
        Set the data for the event and clean it up afterward
        """
        try:
            fl_ctx.set_prop(key=key, value=data, private=True, sticky=False)
            self.fire_event(event_type, fl_ctx)
        finally:
            fl_ctx.set_prop(key=key, value=None, private=True, sticky=False)

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
        self._self_check()
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
        self._self_check()
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
        self._self_check()
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
        self._self_check()
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
        self._self_check()
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
        self._self_check()
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

    def register_event_handler(self, event_types: Union[str, List[str]], handler, **kwargs):
        self._self_check()
        if isinstance(event_types, str):
            event_types = [event_types]
        elif not isinstance(event_types, list):
            raise ValueError(f"event_types must be string or list of strings but got {type(event_types)}")

        if not callable(handler):
            raise ValueError(f"handler {handler.__name__} is not callable")

        for e in event_types:
            entries = self._event_handlers.get(e)
            if not entries:
                entries = []
                self._event_handlers[e] = entries

            already_registered = False
            for h, _ in entries:
                if handler == h:
                    # already registered: either by a super class or by the class itself.
                    already_registered = True
                    break

            if not already_registered:
                entries.append((handler, kwargs))

    def get_event_handlers(self):
        self._self_check()
        return self._event_handlers
