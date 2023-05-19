# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
#


# Part of code is Adapted from from https://github.com/Project-MONAI/MONAI/blob/dev/monai/utils/module.py#L282
# which has the following license
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Part of code is Adapted from from https://github.com/Project-MONAI/MONAI/blob/dev/monai/utils/module.py#L282
"""

from importlib import import_module
from typing import Any, Tuple

from nvflare.security.logging import secure_format_exception

OPTIONAL_IMPORT_MSG_FMT = "{}"
OPS = ["==", ">=", ">", "<", "<="]


def get_module_version(this_pkg):
    return this_pkg.__version__.split(".")[:2]


def get_module_version_str(the_module):
    if the_module:
        module_version = ".".join(get_module_version(the_module))
    else:
        module_version = ""

    return module_version


def check_version(that_pkg, version: str = "", op: str = "==") -> bool:
    """
    compare module version with provided version
    """
    if not version or not hasattr(that_pkg, "__version__"):
        return True  # always valid version

    mod_version = tuple(int(x) for x in get_module_version(that_pkg))
    required = tuple(int(x) for x in version.split("."))
    result = True
    if op == "==":
        result = mod_version == required
    elif op == ">=":
        result = mod_version >= required
    elif op == ">":
        result = mod_version > required
    elif op == "<":
        result = mod_version < required
    elif op == "<=":
        result = mod_version <= required

    return result


class LazyImportError(ImportError):
    """
    Could not import APIs from an optional dependency.
    """


def optional_import(
    module: str,
    op: str = "==",
    version: str = "",
    name: str = "",
    descriptor: str = OPTIONAL_IMPORT_MSG_FMT,
    allow_namespace_pkg: bool = False,
) -> Tuple[Any, bool]:
    """
    Imports an optional module specified by `module` string.
    Any importing related exceptions will be stored, and exceptions raise lazily
    when attempting to use the failed-to-import module.
    Args:
        module: name of the module to be imported.
        op: version op it should be one of the followings: ==, <=, <, >, >=
        version: version string of the module if specified, it is used to the module version such
        that is satisfy the condition: <module>.__version__ <op> <version>.
        name: a non-module attribute (such as method/class) to import from the imported module.
        descriptor: a format string for the final error message when using a not imported module.
        allow_namespace_pkg: whether importing a namespace package is allowed. Defaults to False.
    Returns:
        The imported module and a boolean flag indicating whether the import is successful.
    Examples::
        >>> torch, flag = optional_import('torch')
        >>> print(torch, flag)
        <module 'torch' from '/..../lib/python3.8/site-packages/torch/__init__.py'> True

        >>> torch, flag = optional_import('torch', '1.1')
        >>> print(torch, flag)
        <module 'torch' from 'python/lib/python3.6/site-packages/torch/__init__.py'> True
        >>> the_module, flag = optional_import('unknown_module')
        >>> print(flag)
        False
        >>> the_module.method  # trying to access a module which is not imported
        OptionalImportError: import unknown_module (No module named 'unknown_module').
        >>> torch, flag = optional_import('torch', '42')
        >>> torch.nn  # trying to access a module for which there isn't a proper version imported
        OptionalImportError: import torch (requires version=='42').
        >>> conv, flag = optional_import('torch.nn.functional', ">=", '1.0', name='conv1d')
        >>> print(conv)
        <built-in method conv1d of type object at 0x11a49eac0>
        >>> conv, flag = optional_import('torch.nn.functional', ">=", '42', name='conv1d')
        >>> conv()  # trying to use a function from the not successfully imported module (due to unmatched version)
        OptionalImportError: from torch.nn.functional import conv1d (requires version>='42').
    """
    tb = None
    exception_str = ""

    if name:
        actual_cmd = f"from {module} import {name}"
    else:
        actual_cmd = f"import {module}"

    pkg = None
    try:
        if op not in OPS:
            raise ValueError(f"invalid op {op}, must be one of {OPS}")

        pkg = __import__(module)  # top level module
        the_module = import_module(module)
        if not allow_namespace_pkg:
            is_namespace = getattr(the_module, "__file__", None) is None and hasattr(the_module, "__path__")
            if is_namespace:
                raise AssertionError
        if name:  # user specified to load class/function/... from the module
            the_module = getattr(the_module, name)
    except Exception as import_exception:  # any exceptions during import
        tb = import_exception.__traceback__
        exception_str = secure_format_exception(import_exception)
    else:  # found the module
        if check_version(pkg, version, op):
            return the_module, True

    # preparing lazy error message
    msg = descriptor.format(actual_cmd)
    if version and tb is None:  # a pure version issue
        msg += f": requires '{module}{op}{version}'"
        if pkg:
            module_version = get_module_version_str(pkg)
            msg += f", current '{module}=={module_version}' "
    if exception_str:
        msg += f" ({exception_str})"

    class _LazyRaise:
        def __init__(self, attr_name, *_args, **_kwargs):
            self.attr_name = attr_name
            _default_msg = f"{msg}."
            if tb is None:
                self._exception = LazyImportError(_default_msg)
            else:
                self._exception = LazyImportError(_default_msg).with_traceback(tb)

        def __getattr__(self, attr_name):
            """
            Raises:
                OptionalImportError: When you call this method.
            """
            raise self._exception

        def __call__(self, *_args, **_kwargs):
            """
            Raises:
                OptionalImportError: When you call this method.
            """
            raise self._exception

    return _LazyRaise(name), False
