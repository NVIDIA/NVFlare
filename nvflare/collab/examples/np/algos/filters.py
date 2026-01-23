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
import random

from nvflare.collab import collab
from nvflare.fuel.utils.log_utils import get_obj_logger


class AddNoiseToModel:

    def __init__(self):
        self.logger = get_obj_logger(self)

    @collab.call_filter
    def add_noise(self, func_kwargs: dict):
        direction = collab.filter_direction
        qual_func_name = collab.qual_func_name
        self.logger.debug(f"[{collab.call_info}] filtering call: {func_kwargs=} {direction=} {qual_func_name=}")
        weights_key = "weights"
        weights = func_kwargs.get(weights_key)
        if weights is None:
            # nothing to filter
            self.logger.info(f"nothing to filter in {func_kwargs}")
            return func_kwargs

        # add some noise to weights
        noise = random.random()
        self.logger.debug(f"[{collab.call_info}] adding noise {noise}")
        weights += noise
        func_kwargs[weights_key] = weights
        self.logger.info(f"[{collab.call_info}] weights after adding noise {noise}: {weights}")
        return func_kwargs


class Print:

    def __init__(self):
        self.logger = get_obj_logger(self)

    @collab.call_filter
    def print_call(self, func_kwargs: dict):
        self.logger.info(f"[{collab.call_info}] print_call on fox ctx {id(collab.context)}")
        direction = collab.filter_direction
        qual_func_name = collab.qual_func_name
        self.logger.info(
            f"[{collab.call_info}] printing call ctx {id(collab.context)}: {func_kwargs=} {direction=} {qual_func_name=}"
        )
        return func_kwargs

    @collab.result_filter
    def print_result(self, result, context):
        self.logger.info(f"[{collab.call_info}] print_result on  {id(context)} fox ctx {id(collab.context)}")
        direction = collab.filter_direction
        qual_func_name = collab.qual_func_name
        self.logger.info(
            f"[{collab.call_info}] printing result ctx {id(collab.context)}: {result=} {direction=} {qual_func_name=}"
        )
        return result
