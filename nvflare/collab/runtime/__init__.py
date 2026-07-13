# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Collab runtime: backend implementations and per-mode execution machinery.

Exports are resolved lazily (PEP 562): backend implementations import from the
api layer, and the api layer imports the abstract Backend from this package, so
eager re-exports here would create a circular import.
"""

_EXPORTS = {
    "Backend": ".backend",
    "LocalBackend": ".local_backend",
    "SubprocessBackend": ".subprocess_backend",
    "FlareBackend": ".flare_backend",
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
