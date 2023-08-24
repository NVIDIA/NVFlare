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


def _apply_filters(filter_list, filter_data, fl_ctx, config_filters, task_name, direction):
    task_filter_list = config_filters.get(task_name + FilterKey.DELIMITER + direction)
    if task_filter_list:
        filter_list.extend(task_filter_list)

    if filter_list:
        for f in filter_list:
            filter_data = f.process(filter_data, fl_ctx)
    return filter_data


def apply_data_filters(data_filters, task_data, fl_ctx, task_name, direction):
    # apply scope filters first
    scope_object = fl_ctx.get_prop(FLContextKey.SCOPE_OBJECT)
    filter_list = []
    if scope_object and scope_object.task_data_filters:
        filter_list.extend(scope_object.task_data_filters.get(direction, []))
    task_data = _apply_filters(filter_list, task_data, fl_ctx, data_filters, task_name, direction)
    return task_data


def apply_result_filters(result_filters, result, fl_ctx, task_name, direction):
    filter_list = []
    scope_object = fl_ctx.get_prop(FLContextKey.SCOPE_OBJECT)
    if scope_object and scope_object.task_result_filters:
        filter_list.extend(scope_object.task_result_filters.get(direction, []))
    result = _apply_filters(filter_list, result, fl_ctx, result_filters, task_name, direction)
    return result
