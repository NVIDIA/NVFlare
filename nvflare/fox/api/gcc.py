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

from nvflare.apis.fl_exception import RunAborted
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

        # num_whole_items_received is the number of WHOLE items received.
        # the queue could contain partial results.
        self.num_whole_items_received = 0

    def append(self, item, is_whole=True):
        """Append an item to the result queue.

        Args:
            item: the item to be appended.
            is_whole: whether the item is whole.

        Returns: whether the queue has received all whole items.

        """
        if self.num_whole_items_received == self.limit:
            raise RuntimeError(f"queue is full: {self.limit} items are already appended")
        self.q.put_nowait(item)

        if is_whole:
            # increment num_whole_items_received only if the item is whole!
            self.num_whole_items_received += 1
        return self.num_whole_items_received == self.limit

    def __iter__(self):
        return self

    def __next__(self):
        if not self.q.empty():
            return self.q.get()

        # queue is empty: do we expect more?
        if self.num_whole_items_received < self.limit:
            # there will be more items - wait until more item is received
            return self.q.get(block=True)
        else:
            # no more items
            raise StopIteration()

    def __len__(self):
        """Return the number of whole items that have been received.
        Note that this is NOT the current number of items in the queue!

        Returns: the number of whole items that have been received

        """
        return self.num_whole_items_received


class ResultWaiter(threading.Event):

    def __init__(self, sites: list[str]):
        super().__init__()
        self.sites = sites
        self.results = ResultQueue(len(sites))
        self.in_sending_count = 0
        self.lock = threading.Lock()
        self.sending_decreased = threading.Condition(self.lock)

    def inc_sending(self):
        with self.sending_decreased:
            self.in_sending_count += 1

    def dec_sending(self):
        with self.sending_decreased:
            self.in_sending_count -= 1
            self.sending_decreased.notify()

    def wait_for_sending_available(self, limit, timeout, abort_signal):
        """Wait for sending count to be decreased if current count exceeds limit

        Args:
            limit: to limit to check
            timeout: time to wait
            abort_signal: abort signal

        Returns: the sending count after waiting

        """
        while True:
            if abort_signal and abort_signal.triggered:
                raise RunAborted("run is aborted")

            with self.sending_decreased:
                if self.in_sending_count < limit:
                    return
                else:
                    self.sending_decreased.wait(timeout)

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
            self.results.append((site_name, partial_result), is_whole=False)


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
        self.send_complete_cb = None
        self.send_complete_cb_kwargs = {}
        self.logger = get_obj_logger(self)

    def set_send_complete_cb(self, cb, **cb_kwargs):
        if not callable(cb):
            raise ValueError("send_complete_cb must be callable")
        self.send_complete_cb = cb
        self.send_complete_cb_kwargs = cb_kwargs

    def send_completed(self):
        if self.send_complete_cb:
            self.send_complete_cb(**self.send_complete_cb_kwargs)

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
                try:
                    result = self.app.apply_incoming_result_filters(self.target_name, self.func_name, result, ctx)
                    if self.process_cb:
                        self.cb_kwargs[CollabMethodArgName.CONTEXT] = ctx
                        check_context_support(self.process_cb, self.cb_kwargs)
                        result = self.process_cb(self, result, **self.cb_kwargs)
                finally:
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
