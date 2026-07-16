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
import logging
from typing import TYPE_CHECKING

from .api.facade import Facade as collab

if TYPE_CHECKING:
    from .core.in_process_env import InProcessEnv
    from .core.multi_process_env import MultiProcessEnv
    from .core.recipe import CollabRecipe
    from .runtime.client_api import CollabClientAPI
    from .runtime.local.runner import InProcessRunner

__all__ = [
    "collab",
    "CollabClientAPI",
    "CollabRecipe",
    "InProcessEnv",
    "InProcessRunner",
    "MultiProcessEnv",
    "simple_logging",
]


def simple_logging(level=logging.INFO):
    """Configure basic stdout logging for example and user scripts."""
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


# The user-facing classes are exported here so users never need the core or
# runtime package paths. Resolution is lazy (PEP 562): CollabRecipe and the
# execution environments pull in FLARE job/runtime machinery, which client-side
# training scripts that only need the `collab` facade should not pay for.
_EXPORTS = {
    "CollabRecipe": ".core.recipe",
    "CollabClientAPI": ".runtime.client_api",
    "InProcessEnv": ".core.in_process_env",
    "MultiProcessEnv": ".core.multi_process_env",
    "InProcessRunner": ".runtime.local.runner",
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
