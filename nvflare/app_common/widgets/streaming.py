# Copyright (c) 2021, NVIDIA CORPORATION.
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
from typing import List, Optional

from nvflare.apis.analytix import Data as AnalyticsData
from nvflare.apis.analytix import DataType as AnalyticsDataType
from nvflare.apis.dxo import DXO
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FedEventHeader, FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.widgets.widget import Widget

EVENT = "log_analytics"


def send_analytic_dxo(comp: FLComponent, dxo: DXO, fl_ctx: FLContext):
    """
    Sends analytic dxo.

    Args:
        comp (FLComponent):
        dxo (DXO): analytic data in dxo.
        fl_ctx (FLContext): fl context info.
    """
    if not isinstance(comp, FLComponent):
        raise TypeError("expect comp to be FLComponent, but got {}".format(type(fl_ctx)))
    if not isinstance(dxo, DXO):
        raise TypeError("expect fl_ctx to be FLContext, but got {}".format(type(fl_ctx)))
    if not isinstance(fl_ctx, FLContext):
        raise TypeError("expect fl_ctx to be FLContext, but got {}".format(type(fl_ctx)))

    fl_ctx.set_prop(key=FLContextKey.EVENT_DATA, value=dxo.to_shareable(), private=True, sticky=False)
    comp.fire_event(event_type=EVENT, fl_ctx=fl_ctx)


def write(tag: str, value, data_type: AnalyticsDataType, kwargs=None) -> DXO:
    """
    Writes the analytic data.

    Args:
        tag (str): the tag associated with this value.
        value: the analytic data.
        data_type (AnalyticsDataType): analytic data type.
        kwargs (dict): additional arguments to be passed into the receiver side's function.

    Returns:
        A DXO object that contains the analytic data.
    """
    data = AnalyticsData(tag=tag, value=value, data_type=data_type, kwargs=kwargs)
    dxo = data.to_dxo()
    return dxo


def write_scalar(tag: str, scalar: float, **kwargs) -> DXO:
    """
    Writes a scalar.

    Args:
        tag (str): the tag associated with this value.
        scalar (float): a scalar to write.
    """
    return write(tag, scalar, data_type=AnalyticsDataType.SCALAR, kwargs=kwargs)


def write_scalars(tag: str, tag_scalar_dict: dict, **kwargs) -> DXO:
    """
    Writes scalars.

    Args:
        tag (str): the tag associated with this dict.
        tag_scalar_dict (dict): A dictionary that contains tag and scalars to write.
    """
    return write(tag, tag_scalar_dict, data_type=AnalyticsDataType.SCALARS, kwargs=kwargs)


def write_image(tag: str, image, **kwargs) -> DXO:
    """
    Writes an image.

    Args:
        tag (str): the tag associated with this value.
        image: the image to write.
    """
    return write(tag, image, data_type=AnalyticsDataType.IMAGE, kwargs=kwargs)


def write_text(tag: str, text: str, **kwargs) -> DXO:
    """
    Writes text.

    Args:
        tag (str): the tag associated with this value.
        text (str): the text to write.
    """
    return write(tag, text, data_type=AnalyticsDataType.TEXT, kwargs=kwargs)


class AnalyticsReceiver(Widget, ABC):
    def __init__(self, events: Optional[List[str]] = None):
        """
        Receives analytic data.

        Args:
            events (optional, List[str]): A list of event that this receiver will handled.
        """
        FLComponent.__init__(self)
        if events is None:
            events = [EVENT, f"fed.{EVENT}"]
        self.events = events

    @abstractmethod
    def initialize(self, fl_ctx: FLContext):
        """
        Initializes the receiver.

        Args:
            fl_ctx (FLContext): fl context.
        """
        pass

    @abstractmethod
    def save(self, fl_ctx: FLContext, shareable: Shareable, record_origin: str):
        """
        Saves the received data.

        Args:
            fl_ctx (FLContext): fl context.
            shareable (Shareable): the received message.
            record_origin (str): the sender of this message / record.
        """
        pass

    @abstractmethod
    def finalize(self, fl_ctx: FLContext):
        """
        Finalizes the receiver.

        Args:
            fl_ctx (FLContext): fl context.

        """
        pass

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.initialize(fl_ctx)
        elif event_type in self.events:
            data = fl_ctx.get_prop(FLContextKey.EVENT_DATA, None)
            if data is None:
                self.log_error(fl_ctx, "Missing event data.")
                return
            if not isinstance(data, Shareable):
                self.log_error(fl_ctx, f"Expect shareable but get {type(data)}")
                return
            record_origin = fl_ctx.get_identity_name()

            # if fed event use peer name to save
            if data.get_header(FedEventHeader.ORIGIN) is not None:
                peer_name = data.get_peer_prop(ReservedKey.IDENTITY_NAME, None)
                record_origin = peer_name

            if record_origin is None:
                self.log_error(fl_ctx, "record_origin can't be None.")
                return
            self.save(shareable=data, fl_ctx=fl_ctx, record_origin=record_origin)
        elif event_type == EventType.END_RUN:
            self.finalize(fl_ctx)
