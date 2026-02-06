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
from .dec import in_call_filter as dec_in_call_filter
from .dec import in_result_filter as dec_in_result_filter
from .dec import init as dec_init
from .dec import out_call_filter as dec_out_call_filter
from .dec import out_result_filter as dec_out_result_filter
from .dec import result_filter as dec_result_filter
from .proxy_list import ProxyList


class facade:

    collab = dec_collab
    init = dec_init
    final = dec_final
    algo = dec_algo
    call_filter = dec_call_filter
    in_call_filter = dec_in_call_filter
    out_call_filter = dec_out_call_filter
    result_filter = dec_result_filter
    in_result_filter = dec_in_result_filter
    out_result_filter = dec_out_result_filter

    @classproperty
    def context(cls):
        """Get the call context.

        Returns: a context object

        """
        return get_call_context()

    @classproperty
    def caller(cls):
        """Get the site name of the caller

        Returns: name of the caller

        """
        ctx = get_call_context()
        return ctx.caller

    @classproperty
    def callee(cls):
        """Get the fully qualified collab object name of the invoked object: <site_name>[.<collab_obj_name>]

        Returns: fully qualified collab object name of the invoked object

        """
        ctx = get_call_context()
        return ctx.callee

    @classproperty
    def call_info(cls):
        """Get a string that represents call information

        Returns: a string that represents call information

        The string looks like:

            <current_site_name>:<caller>=><callee>

        """
        ctx = get_call_context()
        return str(ctx)

    @classproperty
    def site_name(cls):
        """Get the current site name, which is the name of the "app" object of the current site.

        Returns: the current site name

        """
        ctx = get_call_context()
        return ctx.app.name

    @classproperty
    def server(cls):
        """Get the server proxy.

        Returns: the server proxy

        """
        ctx = get_call_context()
        return ctx.server

    @classproperty
    def clients(cls):
        """Get all client proxies.

        Returns: all client proxies as a ProxyList

        """
        ctx = get_call_context()
        return ProxyList(ctx.clients)

    @classproperty
    def other_clients(cls):
        """Get all client proxies, excluding the site's own proxy.

        Returns: all client proxies, excluding the site's own proxy

        """
        ctx = get_call_context()

        # Note that ctx.clients returns a copy of client proxies, not the original client proxy list!
        # So it is safe to manipulate the candidates here.
        candidates = ctx.clients
        me = ctx.app.my_site
        if me in candidates:
            candidates.remove(me)
        return ProxyList(candidates)

    @classproperty
    def child_clients(cls):
        """Get all child client proxies.

        Returns: all child client proxies if the site has children. An exception is raised if no children.

        """
        ctx = get_call_context()
        candidates = ctx.app.get_children()
        if not candidates:
            raise RuntimeError(f"app {ctx.app.name} has no child clients")
        return ProxyList(candidates)

    @classproperty
    def has_children(cls):
        """Check whether the site has any child proxies.

        Returns: whether the site has any child proxies

        """
        ctx = get_call_context()
        return ctx.app.has_children()

    @classproperty
    def leaf_clients(cls):
        """Get all leaf client proxies.

        Returns: all leaf client proxies

        """
        ctx = get_call_context()
        candidates = ctx.app.get_leaf_clients()
        if not candidates:
            raise RuntimeError(f"app {ctx.app.name} has no leaf clients")
        return ProxyList(candidates)

    @classmethod
    def get_clients(cls, names: list[str]):
        """Get proxies for specified site names.

        Args:
            names: names of the sites for which to get proxies.

        Returns:

        """
        ctx = get_call_context()
        candidates = ctx.clients
        result = []
        for n in names:
            p = None
            for c in candidates:
                if c.name == n:
                    p = c
                    break
            if not p:
                # no proxy for this name
                raise RuntimeError(f"app {ctx.app.name} has no client '{n}'")
            result.append(p)
        return ProxyList(result)

    @classproperty
    def backend_type(cls):
        """Get the backend type of the current site.

        Returns: the backend type of the current site

        """
        ctx = get_call_context()
        return ctx.backend_type

    @classproperty
    def is_aborted(cls):
        """Check whether the job/experiment has been aborted.

        Returns: whether the job/experiment has been aborted

        """
        ctx = get_call_context()
        return ctx.is_aborted()

    @classproperty
    def workspace(cls):
        """Get the workspace object.

        Returns: the workspace object

        """
        ctx = get_call_context()
        return ctx.workspace

    @classproperty
    def filter_direction(cls):
        """Get the direction of filter call (incoming or outgoing). Only available to filter functions.

        Returns: the direction of filter call

        """
        ctx = get_call_context()
        return ctx.get_prop(ContextKey.DIRECTION)

    @classproperty
    def qual_func_name(cls):
        """Get the filter's qualified function name. Only available to filter functions.

        Returns: the filter's qualified function name

        """
        ctx = get_call_context()
        return ctx.get_prop(ContextKey.QUALIFIED_FUNC_NAME)

    @staticmethod
    def fire_event(event_type: str, data):
        """Fire an event to listening objects within the site.

        Args:
            event_type: type of the event
            data: data of the event

        Returns: results from event handlers.

        """
        ctx = get_call_context()
        return ctx.app.fire_event(event_type, data, ctx)

    @staticmethod
    def register_event_handler(event_type: str, handler, **handler_kwargs):
        """Register an event handler for a specified event type

        Args:
            event_type: type of the event
            handler: the handler function to be registered
            **handler_kwargs: kwargs to be passed to the handler

        Returns: None

        """
        ctx = get_call_context()
        ctx.app.register_event_handler(event_type, handler, **handler_kwargs)

    @staticmethod
    def get_app_prop(name: str, default=None):
        """Get a specified property from the site's app (usually for configuration properties).

        Args:
            name: name of the property.
            default: default value if the property does not exist.

        Returns: value of the specified app property, or default value if the property does not exist

        """
        ctx = get_call_context()
        return ctx.app.get_prop(name, default)

    @staticmethod
    def set_app_prop(name: str, value):
        """Set a specified property into the site's app.
        Properties in app are permanent during the job/experiment execution.

        Args:
            name: name of the property.
            value: value of the property.

        Returns:

        """
        ctx = get_call_context()
        return ctx.app.set_prop(name, value)

    @staticmethod
    def get_prop(name: str, default=None):
        """Get a specified property from the call context. Usually for sharing information during collab function
        processing.

        Args:
            name: name of the property.
            default: default value if the property does not exist.

        Returns:

        """
        ctx = get_call_context()
        return ctx.get_prop(name, default)

    @staticmethod
    def set_prop(name: str, value):
        """Set a specified property into the call context. Usually for sharing information during collab function
        processing.

        Args:
            name: name of the property.
            value: value of the property.

        Returns:

        """
        ctx = get_call_context()
        return ctx.set_prop(name, value)

    @staticmethod
    def get_result(default=None):
        """Get the last algo execution result from the call context.

        Args:
            default: the default value if the result does not exist in the call context.

        Returns: the last algo execution result from the call context

        """
        return facade.get_prop(ContextKey.RESULT, default)
