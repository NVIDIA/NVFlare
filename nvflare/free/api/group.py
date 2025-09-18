import copy
import time
from typing import List

from nvflare.apis.signal import Signal
from nvflare.apis.fl_exception import RunAborted

from .proxy import Proxy
from .ctx import Context
from .resp import Resp
from .constants import CollabMethodArgName


class Group:

    def __init__(
        self,
        abort_signal: Signal,
        proxies: List[Proxy],
        blocking: bool = True,
        timeout: float = None,
        min_resps: int = None,
        wait_after_min_resps: float = None,
        process_resp_cb=None,
        **cb_kwargs,
    ):
        self._abort_signal = abort_signal
        self._proxies = proxies
        self._blocking = blocking
        self._timeout = timeout
        self._min_resps = min_resps
        self._wait_after_min_resps = wait_after_min_resps
        self._process_resp_cb = process_resp_cb
        self._cb_kwargs = cb_kwargs

        if not min_resps:
            self._min_resps = len(proxies)

        if not wait_after_min_resps:
            self._wait_after_min_resps = 0

    def __getattr__(self, func_name):
        """
        This method is called when Python cannot find an invoked method func_name of this class.
        """

        def method(*args, **kwargs):
            resps = {}
            for p in self._proxies:
                kwargs_copy = copy.copy(kwargs)

                ctx = Context(p.caller_name, p.name)
                kwargs_copy[CollabMethodArgName.CONTEXT] = ctx

                resp = Resp(self._process_resp_cb, self._cb_kwargs)
                resps[p.name] = resp
                kwargs_copy[CollabMethodArgName.ABORT_SIGNAL] = self._abort_signal
                p.backend.call_target_with_resp(
                    resp, p.name, func_name, *args, **kwargs_copy
                )

            # wait for responses
            if not self._blocking:
                return

            start_time = time.time()
            min_received_time = None
            while True:
                if self._abort_signal.triggered:
                    raise RunAborted("run is aborted")

                # how many resps have been received?
                resps_received = 0
                for name, resp in resps.items():
                    if resp.resp_time:
                        resps_received += 1

                if resps_received == len(self._proxies):
                    break

                now = time.time()
                if resps_received >= self._min_resps:
                    if not min_received_time:
                        min_received_time = now
                    if now - min_received_time > self._wait_after_min_resps:
                        # waited long enough
                        break
                elif self._timeout and now - start_time > self._timeout:
                    # timed out
                    break
                else:
                    # still have not received min resps
                    time.sleep(0.1)

            # process results
            results = {}
            for name, resp in resps.items():
                if resp.resp_time:
                    if resp.exception:
                        result = resp.exception
                    else:
                        result = resp.result
                else:
                    result = TimeoutError()
                results[name] = result
            return results

        return method
