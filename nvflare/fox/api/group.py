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

from nvflare.apis.fl_exception import RunAborted
from nvflare.apis.signal import Signal
from nvflare.fuel.utils.log_utils import get_obj_logger

from .app import App
from .constants import CollabMethodArgName, CollabMethodOptionName
from .ctx import Context, get_call_context, set_call_context
from .gcc import GroupCallContext, ResultWaiter
from .proxy import Proxy
from .utils import check_call_args


class Group:

    def __init__(
        self,
        app,
        abort_signal: Signal,
        proxies: List[Proxy],
        blocking: bool = True,
        expect_result: bool = True,
        timeout: float = 5.0,
        optional: bool = False,
        secure: bool = False,
        process_resp_cb=None,
        **cb_kwargs,
    ):
        """A Group is a group of remote apps to be called.

        Args:
            app: the calling app.
            abort_signal: signal to abort execution.
            proxies: proxies of the remote apps to be called.
            blocking: whether to block until responses are received from all remote apps.
            expect_result: whether to expect result from remote sites.
            timeout: how long to wait for responses.
            optional: whether the call is optional or not.
            secure: whether the call is secure or not.
            min_resps: minimum number of responses expected.
            wait_after_min_resps: how much longer to wait after min_resps are received.
            process_resp_cb: callback function to be called to process responses from remote apps.
            **cb_kwargs: kwargs passed to process_resp_cb.
        """
        if not proxies:
            raise ValueError("no proxies to group")

        self._app = app
        self._abort_signal = abort_signal
        self._proxies = proxies
        self._blocking = blocking
        self._expect_result = expect_result
        self._timeout = timeout
        self._optional = optional
        self._secure = secure
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

    def __getattr__(self, func_name):
        """
        This method is called when Python cannot find an invoked method func_name of this class.
        """

        def method(*args, **kwargs):
            the_backend = None
            orig_ctx = get_call_context()
            try:
                gccs = {}

                # filter once for all targets
                p = self._proxies[0]

                # func_proxy is the proxy that actually has the func.
                # the func_proxy is either "p" or a child of "p".
                func_proxy, func_itf, adj_args, adj_kwargs = p.adjust_func_args(func_name, args, kwargs)
                the_backend = p.backend
                ctx = func_proxy.app.new_context(func_proxy.caller_name, func_proxy.name, target_group=self)

                self._logger.debug(
                    f"[{ctx.header_str()}] calling {func_name} of group {[p.name for p in self._proxies]}"
                )

                # apply outgoing call filters
                assert isinstance(self._app, App)
                adj_kwargs = self._app.apply_outgoing_call_filters(func_proxy.target_name, func_name, adj_kwargs, ctx)
                check_call_args(func_name, func_itf, adj_args, adj_kwargs)

                waiter = ResultWaiter([p.name for p in self._proxies])
                for p in self._proxies:
                    func_proxy, func_itf, call_args, call_kwargs = p.adjust_func_args(func_name, adj_args, adj_kwargs)
                    call_kwargs = copy.copy(call_kwargs)
                    ctx = self._app.new_context(
                        func_proxy.caller_name, func_proxy.name, target_group=self, set_call_ctx=False
                    )

                    call_kwargs[CollabMethodArgName.CONTEXT] = ctx

                    # set the optional args to help backend decide how to call
                    if self._timeout:
                        call_kwargs[CollabMethodOptionName.TIMEOUT] = self._timeout

                    call_kwargs[CollabMethodOptionName.BLOCKING] = self._blocking
                    call_kwargs[CollabMethodOptionName.EXPECT_RESULT] = self._expect_result
                    call_kwargs[CollabMethodOptionName.SECURE] = self._secure
                    call_kwargs[CollabMethodOptionName.OPTIONAL] = self._optional

                    gcc = GroupCallContext(
                        self._app,
                        func_proxy.target_name,
                        func_name,
                        self._process_resp_cb,
                        self._cb_kwargs,
                        ctx,
                        waiter,
                    )
                    gccs[p.name] = gcc

                    self._logger.debug(
                        f"[{ctx.header_str()}] group call: {func_name=} args={call_args} kwargs={call_kwargs}"
                    )
                    func_proxy.backend.call_target_in_group(gcc, func_name, *call_args, **call_kwargs)

                if not self._expect_result:
                    # do not wait for responses
                    return None

                if not self._blocking:
                    self._logger.debug(f"not blocking {func_name}")
                    return waiter.results

                # wait for responses
                while True:
                    if self._abort_signal.triggered:
                        raise RunAborted("run is aborted")

                    # wait for a short time, so we can check other conditions
                    done = waiter.wait(0.1)
                    if done:
                        self._logger.info(f"all received from {waiter.results} for func {func_name}")
                        break

                return waiter.results
            except Exception as ex:
                if the_backend:
                    the_backend.handle_exception(ex)
            finally:
                if orig_ctx:
                    set_call_context(orig_ctx)

        return method


def group(
    ctx: Context,
    proxies: List[Proxy],
    blocking: bool = True,
    expect_result: bool = True,
    timeout: float = 5.0,
    optional: bool = False,
    secure: bool = False,
    process_resp_cb=None,
    **cb_kwargs,
):
    """This is a convenience method for creating a group.

    Args:
        ctx:
        proxies:
        blocking:
        expect_result:
        timeout:
        optional:
        secure:
        process_resp_cb:
        **cb_kwargs:

    Returns:

    """
    return Group(
        ctx.app,
        ctx.abort_signal,
        proxies,
        blocking,
        expect_result,
        timeout,
        optional,
        secure,
        process_resp_cb,
        **cb_kwargs,
    )


def all_clients(
    ctx: Context,
    blocking: bool = True,
    expect_result: bool = True,
    timeout: float = 5.0,
    optional: bool = False,
    secure: bool = False,
    process_resp_cb=None,
    **cb_kwargs,
):
    """This is a convenience method for creating a group with all clients.

    Args:
        ctx:
        blocking:
        expect_result:
        timeout:
        optional:
        secure:
        process_resp_cb:
        **cb_kwargs:

    Returns:

    """
    return Group(
        ctx.app,
        ctx.abort_signal,
        ctx.clients,
        blocking,
        expect_result,
        timeout,
        optional,
        secure,
        process_resp_cb,
        **cb_kwargs,
    )


def all_other_clients(
    ctx: Context,
    blocking: bool = True,
    expect_result: bool = True,
    timeout: float = 5.0,
    optional: bool = False,
    secure: bool = False,
    process_resp_cb=None,
    **cb_kwargs,
):
    """This is a convenience method for creating a group with all other clients (excluding myself).

    Args:
        ctx:
        blocking:
        expect_result:
        timeout:
        optional:
        secure:
        process_resp_cb:
        **cb_kwargs:

    Returns:

    """
    candidates = ctx.clients
    me = ctx.app.get_my_site()
    if me in candidates:
        candidates.remove(me)

    return Group(
        ctx.app,
        ctx.abort_signal,
        candidates,
        blocking,
        expect_result,
        timeout,
        optional,
        secure,
        process_resp_cb,
        **cb_kwargs,
    )


def all_children(
    ctx: Context,
    blocking: bool = True,
    expect_result: bool = True,
    timeout: float = 5.0,
    optional: bool = False,
    secure: bool = False,
    process_resp_cb=None,
    **cb_kwargs,
):
    """This is a convenience method for creating a group with all my child clients.

    Args:
        ctx:
        blocking:
        expect_result:
        timeout:
        optional:
        secure:
        process_resp_cb:
        **cb_kwargs:

    Returns:

    """
    clients = ctx.app.get_children()
    if not clients:
        raise RuntimeError(f"app {ctx.app.name} has no child clients")

    return Group(
        ctx.app,
        ctx.abort_signal,
        clients,
        blocking,
        expect_result,
        timeout,
        optional,
        secure,
        process_resp_cb,
        **cb_kwargs,
    )


def all_leaf_clients(
    ctx: Context,
    blocking: bool = True,
    expect_result: bool = True,
    timeout: float = 5.0,
    optional: bool = False,
    secure: bool = False,
    process_resp_cb=None,
    **cb_kwargs,
):
    """This is a convenience method for creating a group with all leaf clients.

    Args:
        ctx:
        blocking:
        expect_result:
        timeout:
        optional:
        secure:
        process_resp_cb:
        **cb_kwargs:

    Returns:

    """
    clients = ctx.app.get_leaf_clients()
    if not clients:
        raise RuntimeError(f"app {ctx.app.name} has no leaf clients")

    return Group(
        ctx.app,
        ctx.abort_signal,
        clients,
        blocking,
        expect_result,
        timeout,
        optional,
        secure,
        process_resp_cb,
        **cb_kwargs,
    )
