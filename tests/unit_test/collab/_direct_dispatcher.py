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
import threading
import time
from concurrent.futures import CancelledError

from nvflare.collab.api._invocation import _InvocationDispatcher
from nvflare.collab.api.app import App
from nvflare.collab.api.call_opt import CallOption
from nvflare.collab.api.call_utils import check_call_args
from nvflare.collab.api.constants import CollabMethodArgName
from nvflare.collab.api.context import get_call_context, set_call_context
from nvflare.collab.api.decorators import adjust_kwargs
from nvflare.collab.api.exceptions import RunAborted
from nvflare.collab.api.group_call_context import GroupCallContext


class _Waiter(threading.Event):

    def __init__(self):
        super().__init__()
        self.result = None


class _DirectDispatcher(_InvocationDispatcher):

    def __init__(self, target_obj_name: str, target_app: App, target_obj, abort_signal, thread_executor):
        _InvocationDispatcher.__init__(self, abort_signal)
        self.target_obj_name = target_obj_name
        self.target_app = target_app
        self.target_obj = target_obj
        self.executor = thread_executor

    def _get_func(self, func_name):
        return self.target_app.find_collab_method(self.target_obj, func_name)

    def _call_target(self, context, target_name: str, call_opt: CallOption, func_name: str, *args, **kwargs):
        func = self._get_func(func_name)
        if not func:
            raise AttributeError(f"{target_name} does not have method '{func_name}' or it is not collab")

        if not callable(func):
            raise AttributeError(f"the method '{func_name}' of {target_name} is not callable")

        expect_result = call_opt.expect_result
        timeout = call_opt.timeout

        waiter = None
        if expect_result:
            waiter = _Waiter()

        self.executor.submit(self._run_func, waiter, context, target_name, func_name, func, args, kwargs)
        if waiter:
            start_time = time.time()
            while True:
                if self.abort_signal.triggered:
                    waiter.result = RunAborted("job is aborted")
                    break

                ok = waiter.wait(0.1)
                if ok:
                    break

                waited = time.time() - start_time
                if waited > timeout:
                    # timed out
                    waiter.result = TimeoutError(f"function {func_name} timed out after {waited} seconds")
                    break

            return waiter.result
        else:
            return None

    def _preprocess(self, context, target_name, func_name, func, kwargs):
        caller_ctx = context
        my_ctx = self.target_app.new_context(caller_ctx.caller, caller_ctx.callee)

        # make sure the final kwargs conforms to func interface
        obj_itf = self.target_app.get_target_object_publish_interface(self.target_obj_name)
        if not obj_itf:
            raise RuntimeError(f"cannot find collab interface for object {self.target_obj_name}")

        func_itf = obj_itf.get_method(func_name)
        if func_itf is None:
            raise RuntimeError(f"cannot find interface for func '{func_name}' of object {self.target_obj_name}")

        check_call_args(func_name, func_itf, [], kwargs)
        kwargs[CollabMethodArgName.CONTEXT] = my_ctx
        adjust_kwargs(func, kwargs)
        return my_ctx, kwargs

    def _run_func(self, waiter: _Waiter, context, target_name, func_name, func, args, kwargs):
        previous_ctx = get_call_context()
        try:
            result = self._invoke(context, target_name, func_name, func, args, kwargs)
            if waiter:
                waiter.result = result
        except Exception as ex:
            if waiter:
                waiter.result = ex
        finally:
            set_call_context(previous_ctx)
            if waiter:
                waiter.set()

    def _invoke(self, context, target_name, func_name, func, args, kwargs):
        ctx, kwargs = self._preprocess(context, target_name, func_name, func, kwargs)
        return func(*args, **kwargs)

    def call_target_in_group(self, gcc: GroupCallContext, func_name: str, *args, **kwargs):
        target_name = gcc.target_name
        func = self._get_func(func_name)
        if not func:
            raise AttributeError(f"{target_name} does not have method '{func_name}' or it is not collab")

        if not callable(func):
            raise AttributeError(f"the method '{func_name}' of {target_name} is not callable")

        future = self.executor.submit(self._run_func_in_group, gcc, func_name, func, args, kwargs)
        timeout = gcc.call_opt.timeout
        timer = None
        if gcc.call_opt.expect_result and isinstance(timeout, (int, float)) and timeout > 0:
            timer = threading.Timer(timeout, self._group_call_timed_out, args=(gcc, func_name, timeout))
            timer.daemon = True
            timer.start()
        future.add_done_callback(lambda done: self._group_call_done(done, timer, gcc, func_name))

    @staticmethod
    def _group_call_done(future, timer: threading.Timer | None, gcc: GroupCallContext, func_name: str):
        if timer:
            timer.cancel()
        if future.cancelled():
            gcc.set_exception(CancelledError(f"function {func_name} was cancelled before execution"))
            gcc.send_completed()

    @staticmethod
    def _group_call_timed_out(gcc: GroupCallContext, func_name: str, timeout: float):
        gcc.set_exception(TimeoutError(f"function {func_name} timed out after {timeout} seconds"))
        gcc.send_completed()

    def _run_func_in_group(self, gcc: GroupCallContext, func_name, func, args, kwargs):
        previous_ctx = get_call_context()
        try:
            target_name = gcc.target_name
            # Execute directly in this worker. Calling _call_target here would
            # submit another task to the same pool and block this worker while
            # waiting, starving the pool for sufficiently large groups.
            result = self._invoke(gcc.context, target_name, func_name, func, args, kwargs)
            gcc.set_result(result)
        except Exception as ex:
            gcc.set_exception(ex)
        finally:
            set_call_context(previous_ctx)
            gcc.send_completed()
