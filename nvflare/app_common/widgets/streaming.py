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

from nvflare.apis.analytix import AnalyticsData, AnalyticsDataType
from nvflare.apis.dxo import DXO
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import EventScope, FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.tracking.tracker_types import Tracker
from nvflare.widgets.widget import Widget

ANALYTIC_EVENT_TYPE = "analytix_log_stats"


def send_analytic_dxo(comp: FLComponent, dxo: DXO, fl_ctx: FLContext, event_type: str = ANALYTIC_EVENT_TYPE):
    """Sends analytic dxo.

    Args:
        comp (FLComponent): An FLComponent.
        dxo (DXO): analytic data in dxo.
        fl_ctx (FLContext): fl context info.
        event_type (str): Event type.
    """
    if not isinstance(comp, FLComponent):
        raise TypeError(f"expect comp to be an instance of FLComponent, but got {type(comp)}")
    if not isinstance(dxo, DXO):
        raise TypeError(f"expect dxo to be an instance of DXO, but got {type(dxo)}")
    if not isinstance(fl_ctx, FLContext):
        raise TypeError(f"expect fl_ctx to be an instance of FLContext, but got {type(fl_ctx)}")

    fl_ctx.set_prop(key=FLContextKey.EVENT_DATA, value=dxo.to_shareable(), private=True, sticky=False)
    comp.fire_event(event_type=event_type, fl_ctx=fl_ctx)


def create_analytic_dxo(
    tag: str, value, data_type: AnalyticsDataType, step: int = None, sender: Tracker = Tracker.TORCH_TB, **kwargs
) -> DXO:
    """Creates the analytic DXO.

    Args:
        tag (str): the tag associated with this value.
        value: the analytic data.
        data_type: (AnalyticsDataType): analytic data type.
        step (int) : global step
        sender: (Tracker), syntax of the sender: such TensorBoard or MLFLow
        kwargs: additional arguments to be passed into the receiver side's function.

    Returns:
        A DXO object that contains the analytic data.
    """
    step = step if step else kwargs.get("global_step", None)
    data = AnalyticsData(key=tag, value=value, data_type=data_type, step=step, sender=sender, kwargs=kwargs)
    dxo = data.to_dxo()
    return dxo


class AnalyticsSender(Widget):
    def __init__(self, event_type=ANALYTIC_EVENT_TYPE):
        """Sends analytics data.

        Note::
            This class implements some common methods follows signatures from PyTorch SummaryWriter.
            It provides a convenient way for Learner to use.

        Args:
            event_type (str): event type to fire.
        """
        super().__init__()
        self.fl_ctx = None
        self.engine = None
        self.event_type = event_type

    def get_tracker_name(self) -> Tracker:
        return Tracker.TORCH_TB

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.engine = fl_ctx.get_engine()

    def _add(
        self,
        tag: str,
        value,
        data_type: AnalyticsDataType,
        global_step: Optional[int] = None,
        kwargs: Optional[dict] = None,
    ):
        kwargs = kwargs if kwargs else {}
        if global_step:
            if not isinstance(global_step, int):
                raise TypeError(f"Expect global step to be an instance of int, but got {type(global_step)}")
            kwargs["global_step"] = global_step
        dxo = create_analytic_dxo(
            tag=tag, value=value, data_type=data_type, step=global_step, sender=self.get_tracker_name(), **kwargs
        )
        with self.engine.new_context() as fl_ctx:
            send_analytic_dxo(self, dxo=dxo, fl_ctx=fl_ctx, event_type=self.event_type)

    def add_scalar(self, tag: str, scalar: float, global_step: Optional[int] = None, **kwargs):
        """Sends a scalar.

        Args:
            tag (str): Data identifier.
            scalar (float): Value to send.
            global_step (optional, int): Global step value.
            **kwargs: Additional arguments to pass to the receiver side.
        """
        self._add(tag=tag, value=scalar, data_type=AnalyticsDataType.SCALAR, global_step=global_step, kwargs=kwargs)

    def add_scalars(self, tag: str, scalars: dict, global_step: Optional[int] = None, **kwargs):
        """Sends scalars.

        Args:
            tag (str): The parent name for the tags.
            scalars (dict): Key-value pair storing the tag and corresponding values.
            global_step (optional, int): Global step value.
            **kwargs: Additional arguments to pass to the receiver side.
        """
        self._add(tag=tag, value=scalars, data_type=AnalyticsDataType.SCALARS, global_step=global_step, kwargs=kwargs)

    def add_text(self, tag: str, text: str, global_step: Optional[int] = None, **kwargs):
        """Sends a text.

        Args:
            tag (str): Data identifier.
            text (str): String to send.
            global_step (optional, int): Global step value.
            **kwargs: Additional arguments to pass to the receiver side.
        """
        self._add(tag=tag, value=text, data_type=AnalyticsDataType.TEXT, global_step=global_step, kwargs=kwargs)

    def add_image(self, tag: str, image, global_step: Optional[int] = None, **kwargs):
        """Sends an image.

        Args:
            tag (str): Data identifier.
            image: Image to send.
            global_step (optional, int): Global step value.
            **kwargs: Additional arguments to pass to the receiver side.
        """
        self._add(tag=tag, value=image, data_type=AnalyticsDataType.IMAGE, global_step=global_step, kwargs=kwargs)

    def flush(self):
        """Flushes out the message.

        This is doing nothing, it is defined for mimic the PyTorch SummaryWriter behavior.
        """
        pass

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
        self._save_lock = Lock()
        self._end = False

    @abstractmethod
    def initialize(self, fl_ctx: FLContext):
        """Initializes the receiver.

        Args:
            fl_ctx (FLContext): fl context.
        """
        pass

    @abstractmethod
    def save(self, fl_ctx: FLContext, shareable: Shareable, record_origin: str):
        """Saves the received data.

        Args:
            fl_ctx (FLContext): fl context.
            shareable (Shareable): the received message.
            record_origin (str): the sender of this message / record.
        """
        pass

    @abstractmethod
    def finalize(self, fl_ctx: FLContext):
        """Finalizes the receiver.

        Args:
            fl_ctx (FLContext): fl context.
        """
        pass

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.initialize(fl_ctx)
        elif event_type in self.events:
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

            # if fed event use peer name to save
            if fl_ctx.get_prop(FLContextKey.EVENT_SCOPE) == EventScope.FEDERATION:
                record_origin = data.get_peer_prop(ReservedKey.IDENTITY_NAME, None)
            else:
                record_origin = fl_ctx.get_identity_name()

            if record_origin is None:
                self.log_error(fl_ctx, "record_origin can't be None.", fire_event=False)
                return
            with self._save_lock:
                self.save(shareable=data, fl_ctx=fl_ctx, record_origin=record_origin)
        elif event_type == EventType.END_RUN:
            self._end = True
            self.finalize(fl_ctx)
