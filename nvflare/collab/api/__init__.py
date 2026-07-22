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

"""Public programming surface of the collab API layer.

User code should import these names from this package rather than from the
implementation modules. Exports are resolved lazily (PEP 562).
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import App, ClientApp, ServerApp
    from .call_opt import CallOption
    from .collab_workspace import CollabWorkspace
    from .constants import ContextKey
    from .context import Context
    from .exceptions import CollabCallError
    from .group_call_context import GroupCallContext
    from .module_wrapper import ModuleWrapper
    from .publish_interface import PublishInterface

__all__ = [
    "App",
    "CallOption",
    "ClientApp",
    "CollabWorkspace",
    "CollabCallError",
    "Context",
    "ContextKey",
    "GroupCallContext",
    "ModuleWrapper",
    "PublishInterface",
    "ServerApp",
]

_EXPORTS = {
    "App": ".app",
    "ClientApp": ".app",
    "ServerApp": ".app",
    "CallOption": ".call_opt",
    "CollabWorkspace": ".collab_workspace",
    "ContextKey": ".constants",
    "Context": ".context",
    "CollabCallError": ".exceptions",
    "GroupCallContext": ".group_call_context",
    "ModuleWrapper": ".module_wrapper",
    "PublishInterface": ".publish_interface",
}


def __getattr__(name):
    mod_path = _EXPORTS.get(name)
    if mod_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(mod_path, __package__)
    return getattr(module, name)


def __dir__():
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))
