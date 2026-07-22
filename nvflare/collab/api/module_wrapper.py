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

"""Wrapper to use a module's functions as Collab publish/main methods."""

import importlib
import inspect
import os
import sys
from types import ModuleType
from typing import Optional, Union

from .decorators import (
    _ATTR_PARAM_NAMES,
    _FLAG_FINAL,
    _FLAG_INIT,
    _FLAG_MAIN,
    _FLAG_PUBLISH,
    _FLAG_SUPPORT_CTX,
    get_param_names,
    is_publish,
)


def _is_main(func):
    """Check if a function has the @collab.main decorator."""
    return getattr(func, _FLAG_MAIN, False) is True


def _is_init(func):
    """Check if a function has the @collab.init decorator."""
    return getattr(func, _FLAG_INIT, False) is True


def _is_final(func):
    """Check if a function has the @collab.final decorator."""
    return getattr(func, _FLAG_FINAL, False) is True


def get_caller_module(stack_level: int = 2) -> Optional[ModuleType]:
    """Get the module of the caller at the requested stack level."""
    frame = inspect.currentframe()
    try:
        for _ in range(stack_level):
            if frame is not None:
                frame = frame.f_back
        if frame is not None:
            module_name = frame.f_globals.get("__name__", "__main__")
            return sys.modules.get(module_name)
    finally:
        del frame
    return None


def resolve_server_client(server, client, stack_level: int = 2):
    """Use the caller's module for a missing server or client object."""
    if server is None or client is None:
        caller_module = get_caller_module(stack_level=stack_level + 1)
        if server is None:
            server = caller_module
        if client is None:
            client = caller_module
    return server, client


def get_importable_module_name(module: ModuleType) -> str:
    """Get an importable module name, handling __main__ case.

    When a script is run as `python script.py`, its __name__ is '__main__',
    which cannot be imported on remote machines. This function converts
    '__main__' to the actual importable module path based on the file location.

    Args:
        module: A Python module object

    Returns:
        An importable module name string

    Example:
        # When running: python path/to/my_module.py
        # module.__name__ = '__main__'
        # Returns: 'my_pkg.my_module' (or 'my_module')
    """
    module_name = module.__name__

    if module_name != "__main__":
        return module_name

    # Handle __main__ case: derive module name from file path
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise ValueError(
            "Cannot determine importable module name for __main__ module. "
            "Please import the module explicitly instead of running it as a script, "
            "or pass the module name string directly to ModuleWrapper."
        )

    # Convert file path to module name
    # e.g., /path/to/my_pkg/my_module.py -> my_pkg.my_module
    module_file = os.path.abspath(module_file)

    # Remove .py extension
    if module_file.endswith(".py"):
        module_file = module_file[:-3]

    # Find all possible module paths relative to sys.path entries
    # We prefer longer (more qualified) paths over shorter ones
    candidates = []

    for path_entry in sys.path:
        if not path_entry:
            path_entry = os.getcwd()
        path_entry = os.path.abspath(path_entry)

        if module_file.startswith(path_entry + os.sep):
            relative_path = module_file[len(path_entry) :].lstrip(os.sep)
            # Convert path separators to dots
            importable_name = relative_path.replace(os.sep, ".")
            # Verify it's actually importable
            try:
                importlib.import_module(importable_name)
                candidates.append(importable_name)
            except ImportError:
                continue

    if not candidates:
        raise ValueError(
            f"Cannot determine importable module name from file: {module_file}. "
            "Please ensure the module is in a package on sys.path, "
            "or pass the module name string directly to ModuleWrapper."
        )

    # Prefer the longest path (most qualified) - this ensures we get
    # 'my_pkg.my_module' (or 'my_module') instead of just 'test'
    # Also prioritize paths that start with 'nvflare.' as they're more likely
    # to be the correct package path for this project
    def score_candidate(name):
        # Higher score = better
        score = name.count(".")
        if name.startswith("nvflare."):
            score += 100  # Strong preference for nvflare package paths
        return score

    return max(candidates, key=score_candidate)


class ModuleWrapper:
    """Wrap a module's publish, main, init, and final functions for Collab.

    Collab creates this wrapper automatically whenever a module is supplied as
    a primary or named app object. This allows callers to pass modules containing
    standalone functions anywhere they could pass class instances:

        # my_module.py
        from nvflare.collab import collab

        @collab.publish
        def train(weights=None):
            ...

        @collab.main
        def fed_avg():
            ...

        # main.py
        import my_module

        recipe = CollabRecipe(server=my_module, client=my_module, ...)

    For a distributed deployment, the module must be
    importable on all client machines (i.e., part of the installed package
    or included in job resources).
    """

    def __init__(self, module: Union[ModuleType, str] = None):
        """Initialize wrapper with a module containing @collab.publish/@collab.main functions.

        Args:
            module: A Python module object OR a fully qualified module name string.
                    When a string is passed, the module will be imported.
                    When no argument is passed (None), the wrapper is in an uninitialized
                    state and will be set up by __setstate__ during unpickling.

        Note:
            For job config serialization, we store the module name as
            self._module which matches the 'module' parameter. FLARE's _get_args()
            looks for param or _param in __dict__, so _module matches 'module'.

            When running as __main__, we convert to an importable module path
            so that remote processes can import the same module.
        """
        if module is None:
            # Uninitialized state - will be set up by __setstate__
            self._module = None
            return

        if isinstance(module, ModuleType):
            # Direct module object.
            # Use get_importable_module_name to handle __main__ case
            self._module = get_importable_module_name(module)
            self._setup_methods(module)
        elif isinstance(module, str):
            # Module name string from job-config reconstruction.
            self._module = module
            actual_module = importlib.import_module(module)
            self._setup_methods(actual_module)
        else:
            raise TypeError(f"module must be a ModuleType or str, got {type(module)}")

    @property
    def module_name(self) -> str:
        """Get the module name."""
        return self._module

    def _setup_methods(self, module: ModuleType):
        """Find and wrap all decorated Collab functions from the module."""
        for name in dir(module):
            if name.startswith("_"):
                continue
            func = getattr(module, name)
            if not callable(func):
                continue

            if is_publish(func):
                wrapped = self._create_collab_method(name, func)
                setattr(self, name, wrapped)
            elif _is_main(func):
                wrapped = self._create_main_method(name, func)
                setattr(self, name, wrapped)
            elif _is_init(func):
                wrapped = self._create_lifecycle_method(name, func, _FLAG_INIT)
                setattr(self, name, wrapped)
            elif _is_final(func):
                wrapped = self._create_lifecycle_method(name, func, _FLAG_FINAL)
                setattr(self, name, wrapped)

    def _create_collab_method(self, name, original_func):
        """Create a method wrapper for an already-decorated @collab.publish function.

        The original function is already decorated, so we just need to make it
        callable as a bound method. We use a simple wrapper that delegates to
        the original and copy the collab flags.
        """

        # Create a simple wrapper that forwards to the original decorated function
        def method(self, *args, **kwargs):
            return original_func(*args, **kwargs)

        # Copy all the collab-related attributes from the original
        setattr(method, _FLAG_PUBLISH, getattr(original_func, _FLAG_PUBLISH, True))
        if hasattr(original_func, _FLAG_SUPPORT_CTX):
            setattr(method, _FLAG_SUPPORT_CTX, getattr(original_func, _FLAG_SUPPORT_CTX))

        original_params = get_param_names(original_func) or []
        setattr(method, _ATTR_PARAM_NAMES, original_params)

        return method.__get__(self, type(self))

    def _create_main_method(self, name, original_func):
        """Create a method wrapper for an already-decorated @collab.main function.

        The original function is already decorated, so we just need to make it
        callable as a bound method.
        """

        # Create a simple wrapper that forwards to the original decorated function
        def method(self):
            return original_func()

        # Copy the main flag from the original
        setattr(method, _FLAG_MAIN, True)

        original_params = get_param_names(original_func) or []
        setattr(method, _ATTR_PARAM_NAMES, original_params)

        return method.__get__(self, type(self))

    def _create_lifecycle_method(self, name, original_func, lifecycle_flag):
        """Create a bound wrapper for an @collab.init or @collab.final function."""

        def method(self, *args, **kwargs):
            return original_func(*args, **kwargs)

        setattr(method, lifecycle_flag, True)
        if hasattr(original_func, _FLAG_SUPPORT_CTX):
            setattr(method, _FLAG_SUPPORT_CTX, getattr(original_func, _FLAG_SUPPORT_CTX))

        original_params = get_param_names(original_func) or []
        setattr(method, _ATTR_PARAM_NAMES, original_params)
        return method.__get__(self, type(self))

    def __deepcopy__(self, memo):
        """Support direct-runner deepcopy."""
        module = importlib.import_module(self._module)
        return ModuleWrapper(module)

    def __getstate__(self):
        """Store only the importable module name when pickled."""
        return {"_module": self._module}

    def __setstate__(self, state):
        """Re-import and set up the module when unpickled."""
        self._module = state["_module"]
        module = importlib.import_module(self._module)
        self._setup_methods(module)


def wrap_if_module(obj):
    """Return a ModuleWrapper for a module, or leave any other object unchanged."""
    return ModuleWrapper(obj) if isinstance(obj, ModuleType) else obj
