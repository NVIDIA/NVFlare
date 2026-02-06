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
from typing import List

from nvflare.apis.signal import Signal
from nvflare.fuel.utils.log_utils import get_obj_logger

from .app import App
from .call_opt import CallOption
from .ctx import Context
from .gcc import GroupCallContext, ResultWaiter
from .proxy import Proxy
from .utils import check_call_args


class Group:

    def __init__(
        self,
        app,
        abort_signal: Signal,
        proxies: List[Proxy],
        call_opt: CallOption = None,
        process_resp_cb=None,
        **cb_kwargs,
    ):
        """A Group is a group of remote apps to be called.

        Args:
            app: the calling app.
            abort_signal: signal to abort execution.
            proxies: proxies of the remote apps to be called.
            call_opt: call option that specifies call behavior
            process_resp_cb: callback function to be called to process responses from remote apps.
            **cb_kwargs: kwargs passed to process_resp_cb.
        """
        if not proxies:
            raise ValueError("no proxies to group")

        self._app = app
        self._abort_signal = abort_signal
        self._proxies = proxies
        if not call_opt:
            call_opt = CallOption()
        self._call_opt = call_opt
        self._process_resp_cb = process_resp_cb
        self._cb_kwargs = cb_kwargs
        self._logger = get_obj_logger(self)

    @property
    def size(self):
        """Size of the group, which is the number of remote apps to be called.

        Returns: size of the group.

        """
        return len(self._proxies)

    @property
    def members(self):
        """
        Returns the members of the group, which is the list of all remote apps to be called.

        Returns: the members of the group

        """
        return self._proxies

    def _get_work_proxy(self, p, func_name):
        if self._call_opt.target:
            child = p.get_child(self._call_opt.target)
            if not child:
                raise RuntimeError(
                    f"site {p.name} does not have collab target named '{self._call_opt.target}': "
                    f"make sure to use correct target in the group call of '{func_name}'."
                )
            return child
        else:
            return p

    def __getattr__(self, func_name):
        """
        This method is called to invoke the specified collab function.

        If expect_result is False, then the call immediately returns None.

        If expect_result is True, a ResultQueue object is returned. Results from each site will be appended to
        the queue when they become available. If a site does not return result before timeout, the site's result
        is TimeoutError exception. Each item in the queue is a tuple of (site_name, result).

        The blocking flag is only meaningful when expect_result is True. If blocking is True, the call does not
        return until results are received from all sites (or timed out). If blocking is False, the call immediately
        returns. In both cases, the ResultQueue object is returned, and the application should iterate through it
        to process site results.

        """

        def method(*args, **kwargs):
            the_backend = None
            try:
                # filter once for all targets
                p = self._get_work_proxy(self._proxies[0], func_name)

                # func_proxy is the proxy that actually has the func.
                # the func_proxy is either "p" or a child of "p".
                func_proxy, func_itf, adj_args, adj_kwargs = p.adjust_func_args(func_name, args, kwargs)
                the_backend = p.backend

                with func_proxy.app.new_context(func_proxy.caller_name, func_proxy.name, target_group=self) as ctx:
                    self._logger.info(
                        f"[{ctx}] calling {func_name} {self._call_opt} of group {[p.name for p in self._proxies]}"
                    )

                    # apply outgoing call filters
                    assert isinstance(self._app, App)
                    adj_kwargs = self._app.apply_outgoing_call_filters(
                        func_proxy.target_name, func_name, adj_kwargs, ctx
                    )
                    check_call_args(func_name, func_itf, adj_args, adj_kwargs)

                    waiter = ResultWaiter([p.name for p in self._proxies])
                    max_parallel = self._call_opt.parallel
                    if max_parallel <= 0:
                        max_parallel = len(self._proxies)

                    for p in self._proxies:
                        p = self._get_work_proxy(p, func_name)
                        func_proxy, func_itf, call_args, call_kwargs = p.adjust_func_args(
                            func_name, adj_args, adj_kwargs
                        )
                        call_kwargs = copy.copy(call_kwargs)
                        ctx = self._app.new_context(
                            func_proxy.caller_name, func_proxy.name, target_group=self, set_call_ctx=False
                        )

                        gcc = GroupCallContext(
                            app=self._app,
                            target_name=func_proxy.target_name,
                            call_opt=self._call_opt,
                            func_name=func_name,
                            process_cb=self._process_resp_cb,
                            cb_kwargs=self._cb_kwargs,
                            context=ctx,
                            waiter=waiter,
                        )

                        # try to get permission to make next call
                        gcc.set_send_complete_cb(self._request_sent, gcc=gcc, proxy=func_proxy)
                        waiter.wait_for_call_permission(max_parallel, self._abort_signal)

                        # make next call
                        waiter.inc_call_count()
                        func_proxy.backend.call_target_in_group(gcc, func_name, *call_args, **call_kwargs)

                    if not self._call_opt.expect_result:
                        # do not wait for responses
                        return None

                    if not self._call_opt.blocking:
                        self._logger.debug(f"not blocking {func_name}")
                        return waiter.results

                    # wait for responses
                    waiter.wait_for_responses(self._abort_signal)
                    return waiter.results
            except Exception as ex:
                self._logger.error(f"exception {type(ex)} occurred: {ex}")
                if the_backend:
                    the_backend.handle_exception(ex)
                raise ex

        return method

    def _request_sent(self, gcc: GroupCallContext, proxy: Proxy):
        self._logger.debug(f"[{gcc.context}] call has been sent to '{proxy.name}' for func '{gcc.func_name}'")
        gcc.waiter.dec_call_count()


def group(
    ctx: Context,
    proxies: List[Proxy],
    call_opt: CallOption = None,
    process_resp_cb=None,
    **cb_kwargs,
):
    """This is a convenience method for creating a group.

    Args:
        ctx: call context.
        proxies: list of proxies.
        call_opt: call option that defines call behavior.
        process_resp_cb: callback to be called to process response from remote site.
        **cb_kwargs: kwargs to be passed to the CB.

    Returns: a Group object.

    """
    return Group(
        ctx.app,
        ctx.abort_signal,
        proxies,
        call_opt,
        process_resp_cb,
        **cb_kwargs,
    )
