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
import threading
import time

from nvflare.fuel.utils.log_utils import get_obj_logger

from .constants import CollabMethodArgName
from .ctx import Context, set_call_context
from .utils import check_context_support


class ResultWaiter(threading.Event):

    def __init__(self, sites: list[str]):
        super().__init__()
        self.num_results_expected = len(sites)
        self.sites = sites
        self.results = {}
        self.lock = threading.Lock()

    def set_result(self, target_name: str, result):
        # target_name is either <site_name> or <site_name>.<obj_name>
        parts = target_name.split(".")
        site_name = parts[0]
        with self.lock:
            print(f"set result for {target_name} on site {site_name}")
            self.results[site_name] = result
            if self.num_results_received() >= self.num_results_expected:
                # print(f"received results from all {self.num_results_expected} sites!")
                self.set()

    def finalize_results(self):
        with self.lock:
            for s in self.sites:
                if s not in self.results:
                    self.results[s] = TimeoutError
        return self.results

    def num_results_received(self):
        return len(self.results)


class GroupCallContext:

    def __init__(self, app, target_name, func_name, process_cb, cb_kwargs, context: Context, waiter: ResultWaiter):
        """GroupCallContext contains contextual information about a group call to a target.

        Args:
            app: the calling app.
            target_name: name of the target to be called in the remote app.
            func_name: name of the function to be called in the remote app.
            process_cb: the callback function to be called to process response from the remote app.
            cb_kwargs: kwargs passed to the callback function.
            context: call context.
            waiter: the waiter to wait for result
        """
        self.app = app
        self.target_name = target_name
        self.func_name = func_name
        self.process_cb = process_cb
        self.cb_kwargs = cb_kwargs
        self.context = context
        self.waiter = waiter
        self.logger = get_obj_logger(self)

    def set_result(self, result):
        """This is called by the backend to set the result received from the remote app.
        If process_cb is available, it will be called with the result from the remote app.

        Args:
            result: the result received from the remote app.

        Returns: None

        """
        # filter incoming result
        ctx = copy.copy(self.context)

        # swap caller/callee
        original_caller = ctx.caller
        ctx.caller = ctx.callee
        ctx.callee = original_caller

        if not isinstance(result, Exception):
            result = self.app.apply_incoming_result_filters(self.target_name, self.func_name, result, ctx)

        if self.process_cb:
            # set the context for the process_cb only
            set_call_context(ctx)
            self.cb_kwargs[CollabMethodArgName.CONTEXT] = ctx
            check_context_support(self.process_cb, self.cb_kwargs)
            result = self.process_cb(result, **self.cb_kwargs)

            # set back to original context
            set_call_context(self.context)
        else:
            self.logger.info(f"{self.func_name} does not have process_cb!")

        self.waiter.set_result(self.target_name, result)

    def set_exception(self, ex):
        """This is called by the backend to set the exception received from the remote app.
        The process_cb will NOT be called.

        Args:
            ex: the exception received from the remote app.

        Returns:

        """
        self.waiter.set_result(self.target_name, ex)
