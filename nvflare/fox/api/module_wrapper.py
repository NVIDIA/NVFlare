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

"""Wrapper to use a module's functions as Fox collab/algo methods."""

import importlib
from types import ModuleType

from .dec import _ATTR_PARAM_NAMES, _FLAG_ALGO, _FLAG_COLLAB, _FLAG_SUPPORT_CTX, get_param_names, is_collab


def _is_algo(func):
    """Check if a function has the @fox.algo decorator."""
    return getattr(func, _FLAG_ALGO, False) is True


class ModuleWrapper:
    """Wraps a module so its @fox.collab and @fox.algo functions work with Fox.

    This allows you to use standalone functions instead of class methods:

        # my_module.py
        from nvflare.fox import fox

        @fox.collab
        def train(weights=None):
            ...

        @fox.algo
        def fed_avg():
            ...

        # main.py
        import my_module
        from nvflare.fox.api.module_wrapper import ModuleWrapper

        client = ModuleWrapper(my_module)
        server = ModuleWrapper(my_module)
        recipe = FoxRecipe(server=server, client=client, ...)

    Note: For FlareBackend (real distributed deployment), the module must be
    importable on all client machines (i.e., part of the installed package
    or included in job resources).
    """

    def __init__(self, module: ModuleType):
        """Initialize wrapper with a module containing @fox.collab/@fox.algo functions.

        Args:
            module: A Python module containing decorated functions.
        """
        # Store module name for pickling (not the module object itself)
        self._module_name = module.__name__
        self._setup_methods(module)

    def _setup_methods(self, module: ModuleType):
        """Find and wrap all collab and algo functions from the module."""
        for name in dir(module):
            if name.startswith("_"):
                continue
            func = getattr(module, name)
            if not callable(func):
                continue

            if is_collab(func):
                wrapped = self._create_collab_method(name, func)
                setattr(self, name, wrapped)
            elif _is_algo(func):
                wrapped = self._create_algo_method(name, func)
                setattr(self, name, wrapped)

    def _create_collab_method(self, name, original_func):
        """Create a method wrapper for an already-decorated @fox.collab function.

        The original function is already decorated, so we just need to make it
        callable as a bound method. We use a simple wrapper that delegates to
        the original and copy the collab flags.
        """

        # Create a simple wrapper that forwards to the original decorated function
        def method(self, *args, **kwargs):
            return original_func(*args, **kwargs)

        # Copy all the collab-related attributes from the original
        setattr(method, _FLAG_COLLAB, getattr(original_func, _FLAG_COLLAB, True))
        if hasattr(original_func, _FLAG_SUPPORT_CTX):
            setattr(method, _FLAG_SUPPORT_CTX, getattr(original_func, _FLAG_SUPPORT_CTX))

        original_params = get_param_names(original_func) or []
        setattr(method, _ATTR_PARAM_NAMES, original_params)

        return method.__get__(self, type(self))

    def _create_algo_method(self, name, original_func):
        """Create a method wrapper for an already-decorated @fox.algo function.

        The original function is already decorated, so we just need to make it
        callable as a bound method.
        """

        # Create a simple wrapper that forwards to the original decorated function
        def method(self):
            return original_func()

        # Copy the algo flag from the original
        setattr(method, _FLAG_ALGO, True)

        original_params = get_param_names(original_func) or []
        setattr(method, _ATTR_PARAM_NAMES, original_params)

        return method.__get__(self, type(self))

    def __deepcopy__(self, memo):
        """Support deepcopy for SimBackend."""
        module = importlib.import_module(self._module_name)
        return ModuleWrapper(module)

    def __getstate__(self):
        """Pickle support for FlareBackend - store only module name."""
        return {"_module_name": self._module_name}

    def __setstate__(self, state):
        """Unpickle support for FlareBackend - re-import and setup."""
        self._module_name = state["_module_name"]
        module = importlib.import_module(self._module_name)
        self._setup_methods(module)
