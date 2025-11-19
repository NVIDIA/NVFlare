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
from .constants import ContextKey
from .ctx import get_call_context
from .dec import algo as dec_algo
from .dec import call_filter as dec_call_filter
from .dec import classproperty
from .dec import collab as dec_collab
from .dec import final as dec_final
from .dec import init as dec_init
from .dec import result_filter as dec_result_filter
from .proxy_list import ProxyList


class facade:

    collab = dec_collab
    init = dec_init
    final = dec_final
    algo = dec_algo
    call_filter = dec_call_filter
    result_filter = dec_result_filter

    @classproperty
    def context(cls):
        return get_call_context()

    @classproperty
    def caller(cls):
        ctx = get_call_context()
        return ctx.caller

    @classproperty
    def callee(cls):
        ctx = get_call_context()
        return ctx.callee

    @classproperty
    def call_info(cls):
        ctx = get_call_context()
        return ctx.header_str()

    @classproperty
    def site_name(cls):
        ctx = get_call_context()
        return ctx.app.name

    @classproperty
    def server(cls):
        ctx = get_call_context()
        return ctx.app.server

    @classproperty
    def clients(cls):
        return cls.get_clients()

    @staticmethod
    def get_clients():
        ctx = get_call_context()
        return ProxyList(ctx.clients)

    @classproperty
    def other_clients(cls):
        ctx = get_call_context()
        candidates = ctx.clients
        me = ctx.app.get_my_site()
        if me in candidates:
            candidates.remove(me)
        return ProxyList(candidates)

    @classproperty
    def child_clients(cls):
        ctx = get_call_context()
        candidates = ctx.app.get_children()
        if not candidates:
            raise RuntimeError(f"app {ctx.app.name} has no child clients")
        return ProxyList(candidates)

    @classproperty
    def has_children(cls):
        ctx = get_call_context()
        return ctx.app.has_children()

    @classproperty
    def leaf_clients(cls):
        ctx = get_call_context()
        candidates = ctx.app.get_leaf_clients()
        if not candidates:
            raise RuntimeError(f"app {ctx.app.name} has no leaf clients")
        return ProxyList(candidates)

    @classproperty
    def env_type(cls):
        ctx = get_call_context()
        return ctx.env_type

    @classproperty
    def is_aborted(cls):
        ctx = get_call_context()
        return ctx.is_aborted()

    @classproperty
    def workspace(cls):
        ctx = get_call_context()
        return ctx.workspace

    @classproperty
    def filter_direction(cls):
        ctx = get_call_context()
        return ctx.get_prop(ContextKey.DIRECTION)

    @classproperty
    def qual_func_name(cls):
        ctx = get_call_context()
        return ctx.get_prop(ContextKey.QUALIFIED_FUNC_NAME)

    @staticmethod
    def fire_event(event_type: str, data):
        ctx = get_call_context()
        return ctx.app.fire_event(event_type, data, ctx)

    @staticmethod
    def register_event_handler(event_type: str, handler, **handler_kwargs):
        ctx = get_call_context()
        ctx.app.register_event_handler(event_type, handler, **handler_kwargs)

    @staticmethod
    def get_app_prop(name: str, default=None):
        ctx = get_call_context()
        return ctx.app.get_prop(name, default)

    @staticmethod
    def get_prop(name: str, default=None):
        ctx = get_call_context()
        return ctx.get_prop(name, default)

    @staticmethod
    def get_input(default=None):
        return facade.get_prop(ContextKey.INPUT, default)
