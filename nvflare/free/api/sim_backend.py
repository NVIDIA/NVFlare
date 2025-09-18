import threading

from .backend import Backend
from .resp import Resp
from .constants import CollabMethodOptionName, CollabMethodArgName
from .utils import check_optional_args


class _Waiter(threading.Event):

    def __init__(self):
        super().__init__()
        self.result = None
        self.exception = None


class SimBackend(Backend):

    def __init__(self, callable_obj, abort_signal, thread_executor):
        self.obj = callable_obj
        self.executor = thread_executor
        self.abort_signal = abort_signal

    def call_target(self, target_name: str, func_name: str, *args, **kwargs):
        func = getattr(self.obj, func_name, None)
        if not func:
            raise AttributeError(f"{target_name} does not have {func_name}")

        if not callable(func):
            raise AttributeError(f"the {func_name} of {target_name} is not callable")

        kwargs[CollabMethodArgName.ABORT_SIGNAL] = self.abort_signal

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

    @staticmethod
    def _run_func(waiter: _Waiter, func, args, kwargs):
        try:
            check_optional_args(func, kwargs)
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
        func = getattr(self.obj, func_name, None)
        if not func:
            raise AttributeError(f"{target_name} does not have {func_name}")

        if not callable(func):
            raise AttributeError(f"the {func_name} of {target_name} is not callable")

        self.executor.submit(self._run_func_with_resp, resp, func, args, kwargs)

    @staticmethod
    def _run_func_with_resp(resp: Resp, func, args, kwargs):
        try:
            check_optional_args(func, kwargs)
            result = func(*args, **kwargs)
            resp.set_result(result)
        except Exception as ex:
            resp.set_exception(ex)
