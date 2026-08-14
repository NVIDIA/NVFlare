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

from nvflare.apis.fl_constant import FilterKey, FLContextKey
from nvflare.fuel.utils.fobs.decomposers.via_downloader import (
    contains_lazy_download_ref,
    materialize_lazy_download_refs,
)


def get_filters(filters_name, fl_ctx, config_filters, task_name, direction):
    """Return the site- and job-configured filters active for a task direction."""
    filter_list = []
    scope_object = fl_ctx.get_prop(FLContextKey.SCOPE_OBJECT)
    if scope_object:
        filters = getattr(scope_object, filters_name)
        if filters:
            filter_list.extend(filters.get(direction, []))

    task_filter_list = config_filters.get(task_name + FilterKey.DELIMITER + direction)
    if task_filter_list:
        filter_list.extend(task_filter_list)
    return filter_list


def apply_filters(filters_name, filter_data, fl_ctx, config_filters, task_name, direction, abort_signal=None):
    fl_ctx.set_prop(FLContextKey.FILTER_DIRECTION, direction, private=True, sticky=False)
    filter_list = get_filters(filters_name, fl_ctx, config_filters, task_name, direction)

    if filter_list:
        if contains_lazy_download_ref(filter_data):
            engine = fl_ctx.get_engine()
            get_cell = getattr(engine, "get_cell", None) if engine is not None else None
            cell = get_cell() if callable(get_cell) else None
            filter_data = materialize_lazy_download_refs(filter_data, cell, abort_signal)

        for f in filter_list:
            filter_data = f.process(filter_data, fl_ctx)
    return filter_data
