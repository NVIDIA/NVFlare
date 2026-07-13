# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Any

from nvflare.collab.api.ctx import Context
from nvflare.collab.api.filter import CallFilter, ResultFilter
from nvflare.collab.sys.downloader import Downloader, download_tensors
from nvflare.fuel.utils.log_utils import get_obj_logger


class OutgoingModelCallFilter(CallFilter):

    def __init__(self, model_arg_name: str):
        super().__init__()
        self.model_arg_name = model_arg_name
        self.logger = get_obj_logger(self)

    def filter_call(self, func_kwargs: dict, context: Context):
        arg_value = func_kwargs.get(self.model_arg_name)
        if not arg_value:
            return func_kwargs

        num_receivers = context.target_group_size
        self.logger.info(f"target group size={num_receivers}")

        downloader = Downloader(
            num_receivers=num_receivers,
            timeout=5.0,
        )
        model = downloader.add_tensors(arg_value, 0)
        func_kwargs[self.model_arg_name] = model
        return func_kwargs


class IncomingModelCallFilter(CallFilter):

    def __init__(self, model_arg_name: str):
        super().__init__()
        self.model_arg_name = model_arg_name
        self.logger = get_obj_logger(self)

    def filter_call(self, func_kwargs: dict, context: Context):
        arg_value = func_kwargs.get(self.model_arg_name)
        if not arg_value:
            return func_kwargs

        err, model = download_tensors(ref=arg_value, per_request_timeout=5.0)
        if err:
            self.logger.error(f"error filtering call arg {arg_value}: {err}")
        else:
            func_kwargs[self.model_arg_name] = model
        return func_kwargs


class OutgoingModelResultFilter(ResultFilter):

    def filter_result(self, result: Any, context: Context):
        if not isinstance(result, dict):
            return result

        downloader = Downloader(
            num_receivers=1,
            timeout=5.0,
        )
        return downloader.add_tensors(result, 0)


class IncomingModelResultFilter(ResultFilter):

    def __init__(self):
        super().__init__()
        self.logger = get_obj_logger(self)

    def filter_result(self, result: Any, context: Context):
        err, model = download_tensors(ref=result, per_request_timeout=5.0)
        if err:
            self.logger.error(f"error filtering result {result}: {err}")
            return result
        else:
            return model
