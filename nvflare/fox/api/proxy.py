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

from nvflare.fuel.utils.log_utils import get_obj_logger

from .backend import Backend
from .call_opt import CallOption
from .utils import check_call_args


class _ProxyCall:

    def __init__(
        self,
        proxy,
        expect_result: bool = True,
        timeout: float = 5.0,
        optional: bool = False,
        secure: bool = False,
        target: str = None,
    ):
        self.proxy = proxy
        self.call_opt = CallOption(
            expect_result=expect_result,
            blocking=expect_result,
            timeout=timeout,
            optional=optional,
            secure=secure,
            target=target,
        )

    def __getattr__(self, func_name):
        def method(*args, **kwargs):
            return self.proxy.call_func(self.call_opt, func_name, args, kwargs)

        return method


class Proxy:

    def __init__(self, app, target_name, target_fqn: str, backend: Backend, target_interface):
        """The Proxy represents a target in the App."""
        self.app = app
        self.target_name = target_name
        self.fqn = target_fqn  # fully qualified name of the target in hierarchy
        self.backend = backend
        self.caller_name = app.name
        self.target_interface = target_interface
        self.children = {}  # child proxies
        self.logger = get_obj_logger(self)

    def __call__(
        self,
        expect_result: bool = True,
        timeout: float = 5.0,
        optional: bool = False,
        secure: bool = False,
        target: str = None,
    ):
        """This is called when the proxy is used with call options.

        Args:
            expect_result:
            timeout:
            optional:
            secure:
            target:

        Returns:

        """
        return _ProxyCall(
            proxy=self,
            expect_result=expect_result,
            timeout=timeout,
            optional=optional,
            secure=secure,
            target=target,
        )

    @property
    def name(self):
        return self.target_name

    def add_child(self, name, p):
        self.children[name] = p
        setattr(self, name, p)

    def get_child(self, name):
        """Get the specified child proxy.

        Args:
            name: name of the child proxy.

        Returns: the child proxy if defined.

        """
        return self.children.get(name)

    def _find_interface(self, func_name):
        """Find interface for specified func name.

        Args:
            func_name: name of the func.

        Returns: the proxy that the func belongs to, the func interface.

        Notes: the proxy represents a remote object. The remote object could have sub-objects. In this case,
        the proxy will have child proxies, each representing a sub-object.

        We first try to find the interface from the proxy itself. If not found, we try to find it from child proxies.

        """
        # self.logger.debug(f"trying to find interface for {func_name}")
        args = self.target_interface.get(func_name) if self.target_interface else None
        if args:
            return self, args

        # try children
        the_args = None
        the_proxy = None
        the_name = None
        for n, c in self.children.items():
            args = c.target_interface.get(func_name) if c.target_interface else None
            if not args:
                continue

            # self.logger.debug(f"found interface for func {func_name}: defined in child {n}")

            if not the_proxy:
                the_name = n
                the_proxy = c
                the_args = args
            else:
                # already found a child proxy that has this func - ambiguity
                raise RuntimeError(
                    f"multiple collab objects ({the_name} and {n}) have {func_name}: please use qualified call"
                )
        return the_proxy, the_args

    def adjust_func_args(self, func_name, args, kwargs):
        """Based on specified args and kwargs, this method finds corresponding keywords for all positional
        args based on the interface of the func, and then moves the positional args into kwargs.

        Once done, all args will have keywords, which makes it easy for call filters to identify the args to process.

        Args:
            func_name: name of the func.
            args: positional arg values.
            kwargs: keyword arg values

        Returns: the proxy that the func belongs to, interface of the func, empty args, and new kwargs

        """
        call_args = args
        call_kwargs = kwargs

        # find the proxy and interface for the func
        p, func_itf = self._find_interface(func_name)
        if not p:
            raise RuntimeError(f"target {self.target_name} does not have method '{func_name}'")

        if func_itf:
            # check args and turn them to kwargs
            num_call_args = len(args) + len(kwargs)
            if num_call_args > len(func_itf):
                raise RuntimeError(
                    f"there are {num_call_args} call args ({args=} {kwargs=}), "
                    f"but function '{func_name}' only supports {len(func_itf)} args ({func_itf})"
                )
            call_kwargs = copy.copy(kwargs)
            call_args = []
            for i, arg_value in enumerate(args):
                call_kwargs[func_itf[i]] = arg_value

        return p, func_itf, call_args, call_kwargs

    def call_func(self, call_opt: CallOption, func_name, args, kwargs):
        """Call the specified function with call options.

        Args:
            call_opt: call option that controls the call behavior.
            func_name: name of func to be called.
            args: args to be passed to the func.
            kwargs: kwargs to be passed to the func.

        Returns: result of the function, or exception.

        """
        try:
            if call_opt.target:
                p = self.get_child(call_opt.target)
                if not p:
                    raise RuntimeError(
                        f"site {self.name} does not have collab target named '{call_opt.target}': "
                        f"make sure to use correct target when calling '{func_name}'."
                    )
            else:
                p = self

            p, func_itf, call_args, call_kwargs = p.adjust_func_args(func_name, args, kwargs)

            with p.app.new_context(self.caller_name, self.name) as ctx:
                # apply outgoing call filters
                call_kwargs = self.app.apply_outgoing_call_filters(p.target_name, func_name, call_kwargs, ctx)
                check_call_args(func_name, func_itf, call_args, call_kwargs)

                result = p.backend.call_target(ctx, p.target_name, call_opt, func_name, *call_args, **call_kwargs)
                if isinstance(result, Exception):
                    raise result

                if result is not None:
                    # filter incoming result filters
                    result = self.app.apply_incoming_result_filters(p.target_name, func_name, result, ctx)
                return result
        except Exception as ex:
            if self.backend:
                try:
                    self.backend.handle_exception(ex)
                except Exception as ex2:
                    # ignore exception from backend handling
                    self.logger.error(f"ignored backend's exception {type(ex2)}")

            # Must return the exception as the result of the func call.
            # Do NOT raise it!
            return ex

    def __getattr__(self, func_name):
        """
        This method is called when the proxy is invoked to perform the specified func without any call options.
        In this case, we use a CallOpt with default values.
        """

        def method(*args, **kwargs):
            return self.call_func(CallOption(), func_name, args, kwargs)

        return method
