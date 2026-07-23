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

from nvflare.collab.api.call_utils import check_context_support
from nvflare.collab.api.exceptions import CollabCallError, RunAborted
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.security.logging import secure_log_traceback

from .call_opt import CallOption
from .constants import CollabMethodArgName
from .context import Context, get_call_context, set_call_context

_SHORT_WAIT = 1.0
_FAILURE_MARKER = object()


class ResultQueue:
    def __init__(self, limit: int):
        if limit <= 0:
            raise ValueError(f"bad queue limit {limit}: must be > 0")

        self.limit = limit
        self.q = queue.Queue()

        # num_whole_items_received is the number of WHOLE items received.
        # the queue could contain partial results.
        self.num_whole_items_received = 0
        self.num_successes = 0
        self.failures = {}
        self.update_lock = threading.Lock()

    def append(self, item, is_whole=True):
        """Append an item to the result queue.

        Args:
            item: the item to be appended.
            is_whole: whether the item is whole.

        Returns: whether the queue has received all whole items.
        """
        with self.update_lock:
            if self.num_whole_items_received < self.limit:
                self.q.put_nowait(item)
                if is_whole:
                    # increment num_whole_items_received only if the item is whole!
                    # note: num_whole_items_received is not the number of all items received.
                    # partial items could be added to the queue but do not count as whole items.
                    self.num_whole_items_received += 1
                    self.num_successes += 1
                return self.num_whole_items_received == self.limit
            else:
                # do not allow any items (partial or whole) to be added to the queue if the queue
                # has already received all expected whole items.
                raise RuntimeError(f"queue is full: {self.limit} whole items are already appended")

    def append_failure(self, site_name: str, error: CollabCallError):
        """Record a terminal failure without yielding it as a successful result."""
        with self.update_lock:
            if self.num_whole_items_received >= self.limit:
                raise RuntimeError(f"queue is full: {self.limit} whole items are already appended")
            self.failures[site_name] = error
            self.num_whole_items_received += 1
            # Wake an iterator that may be waiting for the final site. The
            # marker is consumed internally and is never yielded to users.
            self.q.put_nowait(_FAILURE_MARKER)
            return self.num_whole_items_received == self.limit

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            # Check the queue and completion count atomically with append().
            with self.update_lock:
                try:
                    item = self.q.get_nowait()
                except queue.Empty:
                    item = None
                    all_received = self.num_whole_items_received >= self.limit

            if item is _FAILURE_MARKER:
                continue
            if item is not None:
                return item
            if all_received:
                raise StopIteration()

            # More outcomes are expected. Do not hold update_lock while
            # waiting, because append() needs it to enqueue the next item.
            item = self.q.get(block=True)
            if item is not _FAILURE_MARKER:
                return item

    def __len__(self):
        """Return the number of successful whole results received.

        Returns: the number of successful results

        """
        return self.num_successes


class ResultWaiter(threading.Event):

    def __init__(self, sites: list[str]):
        super().__init__()
        self.sites = sites
        self._expected_sites = {self._get_site_name(site) for site in sites}
        if len(self._expected_sites) != len(sites):
            raise ValueError(f"sites must be unique but got {sites}")
        self._received_sites = set()
        self._result_lock = threading.Lock()
        self.results = ResultQueue(len(self._expected_sites))
        self.standing_call_count = 0
        self.call_count_decreased = threading.Condition(threading.Lock())
        self.logger = get_obj_logger(self)

    def inc_call_count(self):
        """Increment standing call count by 1.

        Returns: None

        """
        with self.call_count_decreased:
            self.standing_call_count += 1

    def dec_call_count(self):
        """Decrease standing call count by 1, and notify other threads waiting for call count decreased.

        Returns: None

        """
        with self.call_count_decreased:
            self.standing_call_count -= 1
            self.call_count_decreased.notify()

    def wait_for_call_permission(self, limit, abort_signal):
        """Wait for the permission to make next call.
        The permission is granted when parallel call count is lower than the specified limit.

        Args:
            limit: to limit to check
            abort_signal: abort signal

        Returns: None

        """
        while True:
            with self.call_count_decreased:
                if abort_signal and abort_signal.triggered:
                    raise RunAborted("run is aborted while waiting for sending availability")

                if self.standing_call_count < limit:
                    return
                else:
                    self.call_count_decreased.wait(_SHORT_WAIT)

    def wait_for_responses(self, abort_signal):
        while True:
            if abort_signal.triggered:
                raise RunAborted("run is aborted while waiting for remote responses")

            done = self.wait(_SHORT_WAIT)
            if done:
                break

    @staticmethod
    def _get_site_name(target_name: str):
        # target_name is either <site_name> or <site_name>.<obj_name>
        parts = target_name.split(".")
        return parts[0]

    def set_result(self, target_name: str, result):
        return self._set_outcome(target_name, result=result)

    def set_failure(self, target_name: str, error: CollabCallError):
        return self._set_outcome(target_name, error=error)

    def _set_outcome(self, target_name: str, result=None, error: CollabCallError = None):
        site_name = self._get_site_name(target_name)
        with self._result_lock:
            if site_name not in self._expected_sites:
                self.logger.warning(f"ignored result from unexpected site '{site_name}'")
                return False
            if site_name in self._received_sites:
                self.logger.warning(f"ignored duplicate result from site '{site_name}'")
                return False

            if error:
                all_received = self.results.append_failure(site_name, error)
            else:
                all_received = self.results.append((site_name, result))
            self._received_sites.add(site_name)
            if all_received:
                self.set()
            return True

    def add_partial_result(self, target_name: str, partial_result):
        site_name = self._get_site_name(target_name)
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
        self.cb_kwargs = copy.copy(cb_kwargs) if cb_kwargs else {}
        self.context = context
        self.waiter = waiter
        self.completion_cb = None
        self.completion_cb_kwargs = {}
        self._completed = False
        self._completion_lock = threading.Lock()
        self.logger = get_obj_logger(self)

    def set_completion_cb(self, cb, **cb_kwargs):
        if not callable(cb):
            raise ValueError("completion_cb must be callable")
        self.completion_cb = cb
        self.completion_cb_kwargs = cb_kwargs

    def call_completed(self):
        with self._completion_lock:
            if self._completed:
                return
            self._completed = True
            cb = self.completion_cb
            cb_kwargs = self.completion_cb_kwargs

        if cb:
            cb(**cb_kwargs)

    def set_result(self, result):
        """This is called by the backend to set the result received from the remote app.
        If process_cb is available, it will be called with the result from the remote app.

        Args:
            result: the result received from the remote app.

        Returns: None

        """
        try:
            ctx = copy.copy(self.context)

            # swap caller/callee
            original_caller = ctx.caller
            ctx.caller = ctx.callee
            ctx.callee = original_caller

            if not isinstance(result, Exception):
                previous_ctx = get_call_context()
                set_call_context(ctx)
                try:
                    if self.process_cb:
                        callback_kwargs = copy.copy(self.cb_kwargs)
                        callback_kwargs[CollabMethodArgName.CONTEXT] = ctx
                        check_context_support(self.process_cb, callback_kwargs)
                        result = self.process_cb(self, result, **callback_kwargs)
                finally:
                    set_call_context(previous_ctx)
        except Exception as ex:
            secure_log_traceback(self.logger)
            result = ex
        finally:
            if isinstance(result, Exception):
                self.waiter.set_failure(self.target_name, self._make_call_error(result))
            else:
                self.waiter.set_result(self.target_name, result)

    def set_exception(self, ex):
        """This is called by the backend to set the exception received from the remote app.
        The process_cb will NOT be called.

        Args:
            ex: the exception received from the remote app.

        Returns:

        """
        self.waiter.set_failure(self.target_name, self._make_call_error(ex))

    def _make_call_error(self, ex) -> CollabCallError:
        if isinstance(ex, CollabCallError):
            return ex
        return CollabCallError(self.target_name, self.func_name, ex)

    def add_partial_result(self, partial_result):
        self.waiter.add_partial_result(self.target_name, partial_result)
