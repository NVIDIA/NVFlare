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
import queue
import threading

from nvflare.fuel.utils.log_utils import get_obj_logger

from .call_opt import CallOption
from .constants import CollabMethodArgName
from .ctx import Context, set_call_context
from .utils import check_context_support


class ResultQueue:
    def __init__(self, limit: int):
        if limit <= 0:
            raise ValueError(f"bad queue limit {limit}: must be > 0")
        self.limit = limit
        self.q = queue.Queue()
        self.consumed = 0
        self.num_items_received = 0

    def append(self, item, complete=True):
        if self.num_items_received == self.limit:
            raise RuntimeError(f"queue is full: {self.limit} items are already appended")
        self.q.put_nowait(item)

        if complete:
            self.num_items_received += 1
        return self.num_items_received == self.limit

    def __iter__(self):
        return self

    def __next__(self):
        if self.consumed == self.limit:
            raise StopIteration()
        else:
            i = self.q.get(block=True)
            self.consumed += 1
            return i

    def __len__(self):
        return self.num_items_received


class ResultWaiter(threading.Event):

    def __init__(self, sites: list[str]):
        super().__init__()
        self.sites = sites
        self.results = ResultQueue(len(sites))
        self.lock = threading.Lock()

    @staticmethod
    def _get_site_name(target_name: str):
        # target_name is either <site_name> or <site_name>.<obj_name>
        parts = target_name.split(".")
        return parts[0]

    def set_result(self, target_name: str, result):
        site_name = self._get_site_name(target_name)
        with self.lock:
            all_received = self.results.append((site_name, result))
            if all_received:
                self.set()

    def add_partial_result(self, target_name: str, partial_result):
        site_name = self._get_site_name(target_name)
        with self.lock:
            self.results.append((site_name, partial_result), complete=False)


class GroupCallContext:

    def __init__(
        self,
        app,
        target_name: str,
        call_opt: CallOption,
        func_name: str,
        process_cb,
        cb_kwargs,
        context: Context,
        waiter: ResultWaiter,
    ):
        """GroupCallContext contains contextual information about a group call to a target.

        Args:
            app: the calling app.
            target_name: name of the target to be called in the remote app.
            call_opt: call options.
            func_name: name of the function to be called in the remote app.
            process_cb: the callback function to be called to process response from the remote app.
            cb_kwargs: kwargs passed to the callback function.
            context: call context.
            waiter: the waiter to wait for result
        """
        self.app = app
        self.call_opt = call_opt
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
        try:
            # filter incoming result
            ctx = copy.copy(self.context)

            # swap caller/callee
            original_caller = ctx.caller
            ctx.caller = ctx.callee
            ctx.callee = original_caller

            if not isinstance(result, Exception):
                set_call_context(ctx)
                result = self.app.apply_incoming_result_filters(self.target_name, self.func_name, result, ctx)
                if self.process_cb:
                    self.cb_kwargs[CollabMethodArgName.CONTEXT] = ctx
                    check_context_support(self.process_cb, self.cb_kwargs)
                    result = self.process_cb(self, result, **self.cb_kwargs)

                # set back to original context
                set_call_context(self.context)
        except Exception as ex:
            result = ex
        finally:
            self.waiter.set_result(self.target_name, result)

    def set_exception(self, ex):
        """This is called by the backend to set the exception received from the remote app.
        The process_cb will NOT be called.

        Args:
            ex: the exception received from the remote app.

        Returns:

        """
        self.waiter.set_result(self.target_name, ex)

    def add_partial_result(self, partial_result):
        self.waiter.add_partial_result(self.target_name, partial_result)
