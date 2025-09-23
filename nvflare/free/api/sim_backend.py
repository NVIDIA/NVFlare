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

from .app import App
from .backend import Backend
from .constants import CollabMethodArgName, CollabMethodOptionName
from .resp import Resp
from .utils import check_context_support


class _Waiter(threading.Event):

    def __init__(self):
        super().__init__()
        self.result = None
        self.exception = None


class SimBackend(Backend):

    def __init__(self, target_app: App, target_obj, abort_signal, thread_executor):
        Backend.__init__(self, abort_signal)
        self.target_app = target_app
        self.target_obj = target_obj
        self.executor = thread_executor

    def _get_func(self, func_name):
        func = getattr(self.target_obj, func_name, None)
        if func:
            return func

        if isinstance(self.target_obj, App):
            # see whether any targets have this method
            default_target = self.target_obj.get_default_target()
            if default_target:
                func = getattr(default_target, func_name, None)
                if func:
                    return func

            targets = self.target_obj.get_target_objects()
            for _, obj in targets:
                func = getattr(obj, func_name, None)
                if func:
                    return func
        return None

    def call_target(self, target_name: str, func_name: str, *args, **kwargs):
        func = self._get_func(func_name)
        if not func:
            raise AttributeError(f"{target_name} does not have {func_name}")

        if not callable(func):
            raise AttributeError(f"the {func_name} of {target_name} is not callable")

        blocking = kwargs.pop(CollabMethodOptionName.BLOCKING, True)
        timeout = kwargs.pop(CollabMethodOptionName.TIMEOUT, None)

        waiter = None
        if blocking:
            waiter = _Waiter()

        self.executor.submit(self._run_func, waiter, func, args, kwargs)
        if waiter:
            ok = waiter.wait(timeout)
            if not ok:
                # timed out
                raise TimeoutError(f"function {func_name} timed out after {timeout} seconds")
            if waiter.exception:
                raise waiter.exception
            return waiter.result

    def _augment_context(self, func, kwargs):
        ctx = kwargs.get(CollabMethodArgName.CONTEXT)
        if ctx:
            ctx.server = self.target_app.server
            ctx.clients = self.target_app.clients
        check_context_support(func, kwargs)

    def _run_func(self, waiter: _Waiter, func, args, kwargs):
        try:
            self._augment_context(func, kwargs)
            result = func(*args, **kwargs)
            if waiter:
                waiter.result = result
        except Exception as ex:
            if waiter:
                waiter.exception = ex
        finally:
            if waiter:
                waiter.set()

    def call_target_with_resp(self, resp: Resp, target_name: str, func_name: str, *args, **kwargs):
        func = self._get_func(func_name)
        if not func:
            raise AttributeError(f"{target_name} does not have {func_name}")

        if not callable(func):
            raise AttributeError(f"the {func_name} of {target_name} is not callable")

        self.executor.submit(self._run_func_with_resp, resp, func, args, kwargs)

    def _run_func_with_resp(self, resp: Resp, func, args, kwargs):
        try:
            self._augment_context(func, kwargs)
            result = func(*args, **kwargs)
            resp.set_result(result)
        except Exception as ex:
            resp.set_exception(ex)
