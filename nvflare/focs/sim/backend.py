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

from nvflare.apis.fl_exception import RunAborted
from nvflare.focs.api.app import App
from nvflare.focs.api.backend import Backend
from nvflare.focs.api.constants import CollabMethodArgName, CollabMethodOptionName
from nvflare.focs.api.dec import adjust_kwargs
from nvflare.focs.api.resp import Resp


class _Waiter(threading.Event):

    def __init__(self):
        super().__init__()
        self.result = None


class SimBackend(Backend):

    def __init__(self, target_app: App, target_obj, abort_signal, thread_executor):
        Backend.__init__(self, abort_signal)
        self.target_app = target_app
        self.target_obj = target_obj
        self.executor = thread_executor

    def _get_func(self, func_name):
        return self.target_app.find_collab_method(self.target_obj, func_name)

    def call_target(self, target_name: str, func_name: str, *args, **kwargs):
        func = self._get_func(func_name)
        if not func:
            raise AttributeError(f"{target_name} does not have method '{func_name}' or it is not collab")

        if not callable(func):
            raise AttributeError(f"the method '{func_name}' of {target_name} is not callable")

        blocking = kwargs.pop(CollabMethodOptionName.BLOCKING, True)
        timeout = kwargs.pop(CollabMethodOptionName.TIMEOUT, None)

        waiter = None
        if blocking:
            waiter = _Waiter()

        self.executor.submit(self._run_func, waiter, func, args, kwargs)
        if waiter:
            start_time = time.time()
            while True:
                if self.abort_signal.triggered:
                    waiter.result = RunAborted("job is aborted")

                ok = waiter.wait(0.1)
                if ok:
                    break

                waited = time.time() - start_time
                if waited > timeout:
                    # timed out
                    waiter.result = TimeoutError(f"function {func_name} timed out after {waited} seconds")
                    break

            return waiter.result

    def _augment_context(self, func, kwargs):
        ctx = kwargs.get(CollabMethodArgName.CONTEXT)
        if ctx:
            target_ctx = self.target_app.new_context(ctx.caller, ctx.callee)
            kwargs[CollabMethodArgName.CONTEXT] = target_ctx
        adjust_kwargs(func, kwargs)

    def _run_func(self, waiter: _Waiter, func, args, kwargs):
        try:
            self._augment_context(func, kwargs)
            result = func(*args, **kwargs)
            if waiter:
                waiter.result = result
        except Exception as ex:
            if waiter:
                waiter.result = ex
        finally:
            if waiter:
                waiter.set()

    def call_target_with_resp(self, resp: Resp, target_name: str, func_name: str, *args, **kwargs):
        # do not use the optional args - they are managed by the group
        kwargs.pop(CollabMethodOptionName.BLOCKING, None)
        kwargs.pop(CollabMethodOptionName.TIMEOUT, None)

        func = self._get_func(func_name)
        if not func:
            raise AttributeError(f"{target_name} does not have method '{func_name}' or it is not collab")

        if not callable(func):
            raise AttributeError(f"the method '{func_name}' of {target_name} is not callable")

        self.executor.submit(self._run_func_with_resp, resp, func, args, kwargs)

    def _run_func_with_resp(self, resp: Resp, func, args, kwargs):
        try:
            self._augment_context(func, kwargs)
            result = func(*args, **kwargs)
            resp.set_result(result)
        except Exception as ex:
            resp.set_exception(ex)
