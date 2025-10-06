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
from .constants import OPTION_ARGS, CollabMethodArgName
from .utils import check_call_args


class Proxy:

    def __init__(self, app, target_name, backend: Backend, target_interface):
        """The Proxy represents a target in the App."""
        self.app = app
        self.target_name = target_name
        self.backend = backend
        self.caller_name = app.name
        self.target_interface = target_interface
        self.children = {}  # child proxies
        self.logger = get_obj_logger(self)

    @property
    def name(self):
        return self.target_name

    def add_child(self, name, p):
        self.children[name] = p
        setattr(self, name, p)

    def get_target(self, name: str):
        obj = getattr(self, name, None)
        if not obj:
            return None
        if isinstance(obj, Proxy):
            return obj
        else:
            return None

    def _find_interface(self, func_name):
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
        call_args = args
        call_kwargs = kwargs

        # find the proxy for the func
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

    def __getattr__(self, func_name):
        """
        This method is called when Python cannot find an invoked method func_name of this class.
        """

        def method(*args, **kwargs):
            # remove option args
            option_args = {}
            for k in OPTION_ARGS:
                if k in kwargs:
                    option_args[k] = kwargs.pop(k)

            p, func_itf, call_args, call_kwargs = self.adjust_func_args(func_name, args, kwargs)
            ctx = p.app.new_context(self.caller_name, self.name)

            self.logger.debug(
                f"[{ctx.header_str()}] calling target {p.target_name} func {func_name}: {call_args=} {call_kwargs=}"
            )

            # apply outgoing call filters
            call_kwargs = self.app.apply_outgoing_call_filters(p.target_name, func_name, call_kwargs, ctx)
            check_call_args(func_name, func_itf, call_args, call_kwargs)

            call_kwargs[CollabMethodArgName.CONTEXT] = ctx

            # restore option args
            for k, v in option_args.items():
                call_kwargs[k] = v

            result = p.backend.call_target(p.target_name, func_name, *call_args, **call_kwargs)
            if isinstance(result, Exception):
                raise result

            # filter incoming result filters
            result = self.app.apply_incoming_result_filters(p.target_name, func_name, result, ctx)
            return result

        return method
