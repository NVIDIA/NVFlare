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
import copy
import time

from .constants import CollabMethodArgName
from .ctx import Context
from .utils import check_optional_args


class Resp:

    def __init__(self, process_cb, cb_kwargs, context: Context):
        self.result = None
        self.exception = None
        self.resp_time = None
        self.process_cb = process_cb
        self.cb_kwargs = cb_kwargs
        self.context = context

    def set_result(self, result):
        if self.process_cb:
            ctx = copy.copy(self.context)

            # swap caller/callee
            original_caller = ctx.caller
            ctx.caller = ctx.callee
            ctx.callee = original_caller
            self.cb_kwargs[CollabMethodArgName.CONTEXT] = ctx
            check_optional_args(self.process_cb, self.cb_kwargs)
            result = self.process_cb(result, **self.cb_kwargs)
        self.result = result
        self.resp_time = time.time()

    def set_exception(self, ex):
        self.exception = ex
        self.resp_time = time.time()
