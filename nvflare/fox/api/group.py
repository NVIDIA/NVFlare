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
import time
from typing import List

from nvflare.apis.fl_exception import RunAborted
from nvflare.apis.signal import Signal
from nvflare.fuel.utils.log_utils import get_obj_logger

from .app import App
from .constants import CollabMethodArgName, CollabMethodOptionName
from .ctx import Context
from .proxy import Proxy
from .resp import Resp
from .utils import check_call_args


class Group:

    def __init__(
        self,
        app,
        abort_signal: Signal,
        proxies: List[Proxy],
        blocking: bool = True,
        timeout: float = 5.0,
        optional: bool = False,
        secure: bool = False,
        min_resps: int = None,
        wait_after_min_resps: float = None,
        process_resp_cb=None,
        **cb_kwargs,
    ):
        if not proxies:
            raise ValueError("no proxies to group")

        self._app = app
        self._abort_signal = abort_signal
        self._proxies = proxies
        self._blocking = blocking
        self._timeout = timeout
        self._optional = optional
        self._secure = secure
        self._min_resps = min_resps
        self._wait_after_min_resps = wait_after_min_resps
        self._process_resp_cb = process_resp_cb
        self._cb_kwargs = cb_kwargs
        self._logger = get_obj_logger(self)

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

            # filter once for all targets
            p = self._proxies[0]
            the_proxy, func_itf, adj_args, adj_kwargs = p.adjust_func_args(func_name, args, kwargs)
            ctx = the_proxy.app.new_context(the_proxy.app.name, the_proxy.name)

            self._logger.debug(f"[{ctx.header_str()}] calling {func_name} of group {[p.name for p in self._proxies]}")

            # apply outgoing call filters
            assert isinstance(self._app, App)
            adj_kwargs = self._app.apply_outgoing_call_filters(p.target_name, func_name, adj_kwargs, ctx)
            check_call_args(func_name, func_itf, adj_args, adj_kwargs)

            for p in self._proxies:
                the_proxy, func_itf, call_args, call_kwargs = p.adjust_func_args(func_name, adj_args, adj_kwargs)
                call_kwargs = copy.copy(call_kwargs)
                ctx = self._app.new_context(the_proxy.caller_name, the_proxy.name)
                call_kwargs[CollabMethodArgName.CONTEXT] = ctx

                # set the optional args to help backend decide how to call
                if self._timeout:
                    call_kwargs[CollabMethodOptionName.TIMEOUT] = self._timeout

                call_kwargs[CollabMethodOptionName.BLOCKING] = self._blocking
                call_kwargs[CollabMethodOptionName.SECURE] = self._secure
                call_kwargs[CollabMethodOptionName.OPTIONAL] = self._optional

                resp = Resp(
                    self._app,
                    p.target_name,
                    func_name,
                    self._process_resp_cb,
                    self._cb_kwargs,
                    ctx,
                )
                resps[p.name] = resp

                self._logger.debug(
                    f"[{ctx.header_str()}] group call: {func_name=} args={call_args} kwargs={call_kwargs}"
                )
                the_proxy.backend.call_target_with_resp(resp, the_proxy.name, func_name, *call_args, **call_kwargs)

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
                    result = resp.result
                else:
                    result = TimeoutError()
                results[name] = result
            return results

        return method


def group(
    ctx: Context,
    proxies: List[Proxy],
    blocking: bool = True,
    timeout: float = 5.0,
    optional: bool = False,
    secure: bool = False,
    min_resps: int = None,
    wait_after_min_resps: float = None,
    process_resp_cb=None,
    **cb_kwargs,
):
    return Group(
        ctx.app,
        ctx.abort_signal,
        proxies,
        blocking,
        timeout,
        optional,
        secure,
        min_resps,
        wait_after_min_resps,
        process_resp_cb,
        **cb_kwargs,
    )


def all_clients(
    ctx: Context,
    blocking: bool = True,
    timeout: float = 5.0,
    optional: bool = False,
    secure: bool = False,
    min_resps: int = None,
    wait_after_min_resps: float = None,
    process_resp_cb=None,
    **cb_kwargs,
):
    return Group(
        ctx.app,
        ctx.abort_signal,
        ctx.clients,
        blocking,
        timeout,
        optional,
        secure,
        min_resps,
        wait_after_min_resps,
        process_resp_cb,
        **cb_kwargs,
    )


def all_other_clients(
    ctx: Context,
    blocking: bool = True,
    timeout: float = 5.0,
    optional: bool = False,
    secure: bool = False,
    min_resps: int = None,
    wait_after_min_resps: float = None,
    process_resp_cb=None,
    **cb_kwargs,
):
    candidates = ctx.clients
    me = ctx.app.get_my_site()
    if me in candidates:
        candidates.remove(me)

    return Group(
        ctx.app,
        ctx.abort_signal,
        candidates,
        blocking,
        timeout,
        optional,
        secure,
        min_resps,
        wait_after_min_resps,
        process_resp_cb,
        **cb_kwargs,
    )


def all_children(
    ctx: Context,
    blocking: bool = True,
    timeout: float = 5.0,
    optional: bool = False,
    secure: bool = False,
    min_resps: int = None,
    wait_after_min_resps: float = None,
    process_resp_cb=None,
    **cb_kwargs,
):
    clients = ctx.app.get_children()
    if not clients:
        raise RuntimeError(f"app {ctx.app.name} has no child clients")

    return Group(
        ctx.app,
        ctx.abort_signal,
        clients,
        blocking,
        timeout,
        optional,
        secure,
        min_resps,
        wait_after_min_resps,
        process_resp_cb,
        **cb_kwargs,
    )


def all_leaf_clients(
    ctx: Context,
    blocking: bool = True,
    timeout: float = 5.0,
    optional: bool = False,
    secure: bool = False,
    min_resps: int = None,
    wait_after_min_resps: float = None,
    process_resp_cb=None,
    **cb_kwargs,
):
    clients = ctx.app.get_leaf_clients()
    if not clients:
        raise RuntimeError(f"app {ctx.app.name} has no leaf clients")

    return Group(
        ctx.app,
        ctx.abort_signal,
        clients,
        blocking,
        timeout,
        optional,
        secure,
        min_resps,
        wait_after_min_resps,
        process_resp_cb,
        **cb_kwargs,
    )
