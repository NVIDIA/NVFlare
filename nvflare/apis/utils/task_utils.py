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

from nvflare.apis.fl_constant import FLContextKey
from nvflare.security.logging import secure_format_exception


def _apply_filters(filter_list, filter_error, filter_data, logger, fl_ctx):
    if filter_list:
        for f in filter_list:
            try:
                filter_data = f.process(filter_data, fl_ctx)
            except Exception as e:
                logger.error(
                    "processing error in task data filter {}: {}; "
                    "asked client to try again later".format(type(f), secure_format_exception(e)),
                )
                filter_error = True
                break
    return filter_error, filter_data


def apply_data_filters(task_filter_list, task_data, logger, fl_ctx):
    filter_error = False
    # apply scope filters first
    scope_object = fl_ctx.get_prop(FLContextKey.SCOPE_OBJECT)
    filter_list = []
    if scope_object and scope_object.task_data_filters:
        filter_list.extend(scope_object.task_data_filters)
    if task_filter_list:
        filter_list.extend(task_filter_list)
    filter_error, task_data = _apply_filters(filter_list, filter_error, task_data, logger, fl_ctx)
    return filter_error, task_data


def apply_result_filters(task_filter_list, result, logger, fl_ctx):
    filter_error = False
    filter_list = []
    scope_object = fl_ctx.get_prop(FLContextKey.SCOPE_OBJECT)
    if scope_object and scope_object.task_result_filters:
        filter_list.extend(scope_object.task_result_filters)
    if task_filter_list:
        filter_list.extend(task_filter_list)
    filter_error, result = _apply_filters(filter_list, filter_error, result, logger, fl_ctx)
    return filter_error, result
