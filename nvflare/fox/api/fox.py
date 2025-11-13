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
from .ctx import get_call_context
from .dec import collab as dec_collab
from .proxy_list import ProxyList


class classproperty:
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, owner_instance, owner_class):
        return self.fget(owner_class)


class fox:

    @classproperty
    def context(cls):
        return get_call_context()

    @classproperty
    def collab(cls):
        return dec_collab

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
    def clients(cls):
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

    @classmethod
    def fire_event(cls, event_type: str, data):
        ctx = get_call_context()
        return ctx.app.fire_event(event_type, data, ctx)
