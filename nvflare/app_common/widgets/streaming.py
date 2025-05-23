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

from abc import ABC, abstractmethod
from threading import Lock
from typing import List, Optional

from nvflare.apis.analytix import ANALYTIC_EVENT_TYPE, AnalyticsDataType, LogWriterName, TrackConst
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import EventScope, FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.utils.analytix_utils import create_analytic_dxo, send_analytic_dxo
from nvflare.widgets.widget import Widget


class AnalyticsSender(Widget):
    def __init__(self, event_type=ANALYTIC_EVENT_TYPE, writer_name=LogWriterName.TORCH_TB):
        """Sender for analytics data.

        This class has some legacy methods that implement some common methods following signatures from
        PyTorch SummaryWriter. New code should use :py:class:`TBWriter <nvflare.app_opt.tracking.tb.tb_writer.TBWriter>` instead,
        which contains an AnalyticsSender.

        Args:
            event_type (str): event type to fire (defaults to "analytix_log_stats").
            writer_name: the log writer for syntax information (defaults to LogWriterName.TORCH_TB)
        """
        super().__init__()
        self.engine = None
        self.event_type = event_type
        self.writer = writer_name

    def get_writer_name(self) -> LogWriterName:
        return self.writer

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.ABOUT_TO_START_RUN:
            self.engine = fl_ctx.get_engine()

    def add(self, tag: str, value, data_type: AnalyticsDataType, global_step: Optional[int] = None, **kwargs):
        """Create and send a DXO by firing an event.

        Args:
            tag (str): Tag name
            value (_type_): Value to send
            data_type (AnalyticsDataType): Data type of the value being sent
            global_step (optional, int): Global step value.

        Raises:
            TypeError: global_step must be an int
        """
        kwargs = kwargs if kwargs else {}
        if global_step is not None:
            if not isinstance(global_step, int):
                raise TypeError(f"Expect global step to be an instance of int, but got {type(global_step)}")
            kwargs[TrackConst.GLOBAL_STEP_KEY] = global_step
        dxo = create_analytic_dxo(tag=tag, value=value, data_type=data_type, writer=self.get_writer_name(), **kwargs)
        with self.engine.new_context() as fl_ctx:
            send_analytic_dxo(self, dxo=dxo, fl_ctx=fl_ctx, event_type=self.event_type)

    def close(self):
        """Close resources."""
        if self.engine:
            self.engine = None


class AnalyticsReceiver(Widget, ABC):
    def __init__(self, events: Optional[List[str]] = None):
        """Receives analytic data.

        Args:
            events (optional, List[str]): A list of event that this receiver will handle.
        """
        super().__init__()
        if events is None:
            events = [ANALYTIC_EVENT_TYPE, f"fed.{ANALYTIC_EVENT_TYPE}"]
        self.events = events
        self._initialized = False
        self._save_lock = Lock()
        self._end = False

    @abstractmethod
    def initialize(self, fl_ctx: FLContext):
        """Initializes the receiver.

        Called after EventType.START_RUN.

        Args:
            fl_ctx (FLContext): fl context.
        """
        pass

    @abstractmethod
    def save(self, fl_ctx: FLContext, shareable: Shareable, record_origin: str):
        """Saves the received data.

        Specific implementations of AnalyticsReceiver will implement save in their own way.

        Args:
            fl_ctx (FLContext): fl context.
            shareable (Shareable): the received message.
            record_origin (str): the sender of this message / record.
        """
        pass

    @abstractmethod
    def finalize(self, fl_ctx: FLContext):
        """Finalizes the receiver.

        Called after EventType.END_RUN.

        Args:
            fl_ctx (FLContext): fl context.
        """
        pass

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self._handle_start_run_event(fl_ctx)
        elif event_type in self.events:
            self._handle_data_event(event_type, fl_ctx)
        elif event_type == EventType.END_RUN:
            self._handle_end_run_event(fl_ctx)

    def _handle_start_run_event(self, fl_ctx: FLContext):
        try:
            self.initialize(fl_ctx)
        except Exception as e:
            # catch the exception so the job can continue
            self.log_error(fl_ctx, f"Receiver initialize failed with {e}.", fire_event=False)
            return
        self._initialized = True

    def _handle_data_event(self, event_type: str, fl_ctx: FLContext):
        if self._initialized:
            if self._end:
                self.log_debug(fl_ctx, f"Already received end run event, drop event {event_type}.", fire_event=False)
                return
            data = fl_ctx.get_prop(FLContextKey.EVENT_DATA, None)
            if data is None:
                self.log_error(fl_ctx, "Missing event data.", fire_event=False)
                return
            if not isinstance(data, Shareable):
                self.log_error(
                    fl_ctx, f"Expect data to be an instance of Shareable but got {type(data)}", fire_event=False
                )
                return

            record_origin = self._get_record_origin(fl_ctx, data)
            if record_origin is None:
                self.log_error(fl_ctx, "record_origin can't be None.", fire_event=False)
                return

            try:
                with self._save_lock:
                    self.save(shareable=data, fl_ctx=fl_ctx, record_origin=record_origin)
            except Exception as e:
                self.log_error(fl_ctx, f"Receiver save method failed with {e}.", fire_event=False)

    def _handle_end_run_event(self, fl_ctx: FLContext):
        if self._initialized:
            self._end = True
            try:
                with self._save_lock:
                    self.finalize(fl_ctx)
            except Exception as e:
                # catch the exception so the job can continue
                self.log_error(fl_ctx, f"Receiver finalize failed with {e}.", fire_event=False)

    def _get_record_origin(self, fl_ctx: FLContext, data: Shareable) -> Optional[str]:
        if fl_ctx.get_prop(FLContextKey.EVENT_SCOPE) == EventScope.FEDERATION:
            return data.get_peer_prop(ReservedKey.IDENTITY_NAME, None)
        else:
            return fl_ctx.get_identity_name()
