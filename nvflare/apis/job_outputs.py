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

"""Name the files a server component writes as part of its job's result.

A module that defines a result-producing server component registers, beside the class, a function that names the
files an instance writes, from that instance's own arguments::

    register_output_resolver(PTFileModelPersistor, lambda p: [in_server_app(p.global_model_file_name)])

When a job's ``meta.json`` is generated, each server component is looked up by its exact class and the paths become
the job's ``job_outputs`` (see :mod:`nvflare.job_config.job_outputs`). A subclass is never matched, since it may
write elsewhere. Paths are relative to the job's run folder; ``app_server/`` is the server app folder, and a
trailing ``/`` names a folder.

This module has no dependencies beyond ``nvflare.apis``, so component modules can import it at any layer.
"""

import posixpath
from typing import Any, Callable, Dict, List, NamedTuple, Optional

from nvflare.apis.job_def import SERVER_SITE_NAME

SERVER_APP_FOLDER = "app_" + SERVER_SITE_NAME  # the server app folder inside a job's run folder

OutputResolver = Callable[[Any], List[Optional[str]]]


class Resolution(NamedTuple):
    resolver: OutputResolver
    supporting: bool


_RESOLVERS: Dict[type, Resolution] = {}


def register_output_resolver(component_class: type, resolver: OutputResolver, supporting: bool = False) -> None:
    """Register how to name the files instances of ``component_class`` write as part of the job's result.

    Args:
        component_class: the exact class; its subclasses are not matched.
        resolver: called with the component when ``meta.json`` is generated; returns run-folder paths. Empty values
            are ignored.
        supporting: the files come with a result but are not one, such as training metrics. A job whose
            components name only supporting files declares no outputs: its main result was not named.
    """
    if not isinstance(component_class, type):
        raise TypeError(f"component_class must be a class, got {type(component_class).__name__}")
    if not callable(resolver):
        raise TypeError("resolver must be callable")
    _RESOLVERS[component_class] = Resolution(resolver, supporting)


def get_output_resolution(component) -> Optional[Resolution]:
    """The resolver registered for ``component``'s exact class, or None."""
    return _RESOLVERS.get(type(component))


def has_resolver_ancestor(component) -> bool:
    """Whether a base class of ``component``'s class, but not the class itself, has a resolver.

    Such a subclass may write elsewhere than its base class, so its files are unknown.
    """
    return any(base in _RESOLVERS for base in type(component).__mro__[1:])


def in_server_app(name: Optional[str]) -> Optional[str]:
    """The run-folder path of a file the server app saves in its own folder."""
    return posixpath.join(SERVER_APP_FOLDER, name) if name else None


def as_folder(path: Optional[str]) -> Optional[str]:
    """A folder entry for ``path``; None for the run folder itself, which a folder entry cannot name."""
    if not path:
        return None
    normalized = posixpath.normpath(path.replace("\\", "/"))
    return None if normalized == "." else f"{normalized}/"
