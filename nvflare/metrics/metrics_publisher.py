# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, ReservedTopic
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.fuel.data_event.data_bus import DataBus
from nvflare.metrics.metrics_keys import METRICS_EVENT_TYPE, MetricKeys


def publish_app_metrics(metrics: dict, metric_name: str, tags: dict, data_bus: DataBus) -> None:
    metrics_data = []
    filtered = [key for key in metrics if key != MetricKeys.type]

    for key in filtered:
        metrics_value = metrics.get(key)
        metrics_type = metrics.get(MetricKeys.type)
        metrics_data.append(
            {
                MetricKeys.metric_name: f"{metric_name}_{key}" if metric_name else key,
                MetricKeys.value: metrics_value,
                MetricKeys.type: metrics_type,
                MetricKeys.tags: tags,
            }
        )
    data_bus.publish([ReservedTopic.APP_METRICS], metrics_data)


def convert_metrics_to_event(
    comp: FLComponent,
    metrics: dict,
    metric_name: str,
    tags: dict,
    fl_ctx: FLContext,
) -> None:
    metrics_data = {MetricKeys.metric_name: metric_name, MetricKeys.value: metrics, MetricKeys.tags: tags}
    shareable = Shareable(data={"METRICS": metrics_data})

    with fl_ctx.get_engine().new_context() as fl_ctx2:
        fl_ctx2.set_prop(key=FLContextKey.EVENT_DATA, value=shareable, private=True, sticky=False)
        comp.fire_event(event_type=METRICS_EVENT_TYPE, fl_ctx=fl_ctx2)


def collect_metrics(
    comp: FLComponent,
    streaming_to_server: bool,
    metrics: dict,
    metric_name: str,
    tags: dict,
    data_bus: DataBus,
    fl_ctx: FLContext,
):

    if not streaming_to_server:
        publish_app_metrics(metrics, metric_name, tags, data_bus)
    else:
        convert_metrics_to_event(comp, metrics, metric_name, tags, fl_ctx)
