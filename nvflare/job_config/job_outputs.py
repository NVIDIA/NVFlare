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

"""Resolve a job's ``job_outputs`` from its server components when its ``meta.json`` is generated.

Each server component and workflow controller is looked up by its exact class in the resolvers registered with
:func:`nvflare.apis.job_outputs.register_output_resolver`, and its paths are read from its own arguments. Nothing is
declared, and the job downloads whole, when a model's location is unknown; a declaration without the model would
leave it out of an outputs-only download.
"""

import os
import posixpath
from typing import List

from nvflare.apis.job_def import MAX_JOB_OUTPUTS, validate_job_outputs
from nvflare.apis.job_outputs import get_output_resolution, has_resolver_ancestor

_ENVIRONMENT_FILE = "environment.json"  # the server app's config/environment.json


def resolve_job_outputs(server_app) -> List[str]:
    """The job's outputs, named by its server components; an empty list declares none.

    Nothing is declared when:

    - the server app ships a ``config/environment.json``, which can move where components write;
    - a model persistor has no resolver, such as a user's own persistor;
    - a component's class has none but a base class has one: the subclass may write elsewhere;
    - only supporting outputs, such as metrics, were named: the main result is unknown;
    - a resolver raises;
    - a path would not pass validation, or there are more than ``MAX_JOB_OUTPUTS`` entries.

    Args:
        server_app: the job's ``ServerAppConfig``.
    """
    from nvflare.app_common.abstract.model_persistor import ModelPersistor

    if ships_server_environment(server_app):
        return []
    main, supporting = [], []
    objects = list(server_app.components.values()) + [w.controller for w in server_app.workflows]
    for obj in objects:
        resolution = get_output_resolution(obj)
        if resolution is None:
            if isinstance(obj, ModelPersistor) or has_resolver_ancestor(obj):
                return []
            continue  # writes no result files
        try:
            paths = [p for p in resolution.resolver(obj) or [] if p]
        except Exception:
            return []  # a resolver that fails never fails the export: the job downloads whole
        (supporting if resolution.supporting else main).extend(paths)
    if not main:
        return []
    return outputs_from(main + supporting)


def outputs_from(paths: List[str]) -> List[str]:
    """The ``job_outputs`` entries for resolved paths.

    A path that would not pass validation (absolute, or outside the run folder) declares nothing: the file it names
    cannot be selected, and a shorter list would leave it out. The same holds for more than MAX_JOB_OUTPUTS entries.
    A job never fails to export because of its own declaration. Duplicates and files inside a declared folder are
    left out.
    """
    entries: List[str] = []
    for path in paths:
        entry = _normalize(path)
        if not entry:
            return []
        if entry not in entries:
            entries.append(entry)
    folders = [e for e in entries if e.endswith("/")]
    # a file inside a declared folder is already selected by the folder
    outputs = [e for e in entries if e.endswith("/") or not any(e.startswith(f) for f in folders)]
    return [] if len(outputs) > MAX_JOB_OUTPUTS else outputs


def ships_server_environment(server_app) -> bool:
    """Whether the server app ships a ``config/environment.json``.

    Such a file can move where server components write (for example a checkpoint folder), so the resolved paths may
    not hold.
    """
    for src_path, dest_dir, app_folder_type in getattr(server_app, "file_sources", []):
        if app_folder_type != "config" or (dest_dir and os.path.normpath(dest_dir) != "."):
            continue  # only a file placed directly in the app's config folder ("." included) is read
        if os.path.isdir(src_path):
            if os.path.isfile(os.path.join(src_path, _ENVIRONMENT_FILE)):
                return True
        elif os.path.basename(src_path) == _ENVIRONMENT_FILE:
            return True
    return False


def _normalize(path) -> str:
    """A path in the form validate_job_outputs accepts, or an empty string when it cannot be one."""
    if not isinstance(path, str) or not path:
        return ""
    folder = path.endswith("/")
    normalized = posixpath.normpath(path.replace("\\", "/"))  # "./output" -> "output"
    if normalized in (".", "") or normalized.startswith("../") or normalized == "..":
        return ""
    entry = f"{normalized}/" if folder else normalized
    try:
        validate_job_outputs([entry])
    except (TypeError, ValueError):
        return ""
    return entry
