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

from nvflare.apis.analytix import AnalyticsData, AnalyticsDataType
from nvflare.apis.dxo import DXO
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext

# TODO: api should not depend on app_common
from nvflare.app_common.tracking.tracker_types import ANALYTIC_EVENT_TYPE, LogWriterName


def send_analytic_dxo(
    comp: FLComponent, dxo: DXO, fl_ctx: FLContext, event_type: str = ANALYTIC_EVENT_TYPE, fire_fed_event: bool = False
):
    """Sends analytic dxo.

    Sends analytic dxo by firing an event (of type "analytix_log_stats" by default unless otherwise specified)
    with the dxo in the fl_ctx.

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

    shareable = dxo.to_shareable()
    fl_ctx.set_prop(key=FLContextKey.EVENT_DATA, value=shareable, private=True, sticky=False)
    if not fire_fed_event:
        comp.fire_event(event_type=event_type, fl_ctx=fl_ctx)
    else:
        comp.fire_fed_event(event_type=event_type, event_data=shareable, fl_ctx=fl_ctx)


def create_analytic_dxo(
    tag: str,
    value,
    data_type: AnalyticsDataType,
    writer: LogWriterName = LogWriterName.TORCH_TB,
    **kwargs,
) -> DXO:
    """Creates the analytic DXO.

    Args:
        tag (str): the tag associated with this value.
        value: the analytic data.
        data_type: (AnalyticsDataType): analytic data type.
        writer (LogWriterName): syntax of the sender: such as TensorBoard or MLflow
        kwargs: additional arguments to be passed into the receiver side's function.

    Returns:
        A DXO object that contains the analytic data.
    """
    data = AnalyticsData(key=tag, value=value, data_type=data_type, sender=writer, **kwargs)
    dxo = data.to_dxo()
    return dxo
