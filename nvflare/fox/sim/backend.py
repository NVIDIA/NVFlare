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
from typing import Optional

from nvflare.apis.fl_exception import RunAborted
from nvflare.fox.api.app import App
from nvflare.fox.api.backend import Backend
from nvflare.fox.api.call_opt import CallOption
from nvflare.fox.api.constants import CollabMethodArgName
from nvflare.fox.api.dec import adjust_kwargs
from nvflare.fox.api.gcc import GroupCallContext
from nvflare.fox.api.utils import check_call_args


class _Waiter(threading.Event):

    def __init__(self):
        super().__init__()
        self.result = None


class SimBackend(Backend):
    """Backend for simulation that can work in-process or via subprocess.

    When subprocess_launcher is provided, calls are forwarded to a subprocess
    (e.g., for torchrun multi-GPU training). Otherwise, calls are executed
    directly in-process using the thread executor.

    This mirrors the FlareBackend pattern where FoxExecutor switches between
    in-process and subprocess modes based on configuration.
    """

    def __init__(
        self,
        target_obj_name: str,
        target_app: App,
        target_obj,
        abort_signal,
        thread_executor,
        subprocess_launcher=None,
    ):
        """Initialize SimBackend.

        Args:
            target_obj_name: Name of the target object.
            target_app: The App instance containing the target.
            target_obj: The actual target object with @fox.collab methods.
            abort_signal: Signal to abort execution.
            thread_executor: ThreadPoolExecutor for in-process execution.
            subprocess_launcher: Optional SubprocessLauncher for subprocess execution.
                                When provided, calls are forwarded to the subprocess.
        """
        Backend.__init__(self, abort_signal)
        self.target_obj_name = target_obj_name
        self.target_app = target_app
        self.target_obj = target_obj
        self.executor = thread_executor
        self.subprocess_launcher = subprocess_launcher

    def _get_func(self, func_name):
        return self.target_app.find_collab_method(self.target_obj, func_name)

    def call_target(self, context, target_name: str, call_opt: CallOption, func_name: str, *args, **kwargs):
        # If subprocess launcher is available, forward call to subprocess
        if self.subprocess_launcher:
            return self._call_via_subprocess(func_name, args, kwargs, call_opt.timeout)

        # Otherwise, execute in-process
        return self._call_inprocess(context, target_name, call_opt, func_name, args, kwargs)

    def _call_via_subprocess(self, func_name: str, args: tuple, kwargs: dict, timeout: float):
        """Forward call to subprocess via launcher."""
        if self.abort_signal.triggered:
            return RunAborted("job is aborted")

        if not self.subprocess_launcher.is_ready():
            raise RuntimeError(f"Subprocess is not ready for {self.target_obj_name}")

        try:
            result = self.subprocess_launcher.call(func_name, args=args, kwargs=kwargs)
            return result
        except Exception as e:
            return e

    def _call_inprocess(self, context, target_name: str, call_opt: CallOption, func_name: str, args, kwargs):
        """Execute call in-process using thread executor."""
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
        kwargs = self.target_app.apply_incoming_call_filters(target_name, func_name, kwargs, my_ctx)

        # make sure the final kwargs conforms to func interface
        obj_itf = self.target_app.get_target_object_collab_interface(self.target_obj_name)
        if not obj_itf:
            raise RuntimeError(f"cannot find collab interface for object {self.target_obj_name}")

        func_itf = obj_itf.get(func_name)
        if not func_itf:
            raise RuntimeError(f"cannot find interface for func '{func_name}' of object {self.target_obj_name}")

        check_call_args(func_name, func_itf, [], kwargs)
        kwargs[CollabMethodArgName.CONTEXT] = my_ctx
        adjust_kwargs(func, kwargs)
        return my_ctx, kwargs

    def _run_func(self, waiter: Optional[_Waiter], context, target_name, func_name, func, args, kwargs):
        try:
            ctx, kwargs = self._preprocess(context, target_name, func_name, func, kwargs)
            result = func(*args, **kwargs)

            # apply result filter
            result = self.target_app.apply_outgoing_result_filters(target_name, func_name, result, ctx)
            if waiter:
                waiter.result = result
        except Exception as ex:
            if waiter:
                waiter.result = ex
        finally:
            if waiter:
                waiter.set()

    def call_target_in_group(self, gcc: GroupCallContext, func_name: str, *args, **kwargs):
        target_name = gcc.target_name
        func = self._get_func(func_name)
        if not func:
            raise AttributeError(f"{target_name} does not have method '{func_name}' or it is not collab")

        if not callable(func):
            raise AttributeError(f"the method '{func_name}' of {target_name} is not callable")

        self.executor.submit(self._run_func_in_group, gcc, func_name, args, kwargs)

    def _run_func_in_group(self, gcc: GroupCallContext, func_name, args, kwargs):
        try:
            target_name = gcc.target_name
            result = self.call_target(gcc.context, target_name, gcc.call_opt, func_name, *args, **kwargs)
            gcc.send_completed()
            gcc.set_result(result)
        except Exception as ex:
            gcc.set_exception(ex)
