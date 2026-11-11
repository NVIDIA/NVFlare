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

"""Declare a recipe job's outputs from the server components it configures.

A result-producing server component reports the files it writes through ``get_job_outputs(app_env, component_ids)``:
paths relative to the job's run folder, the root of its workspace archive, or None when it cannot tell. Each
component keeps that knowledge next to the code that writes the files; this module only asks. A component without
the method writes no outputs. When any component returns None, such as a model persistor that does not know its
own file, the job declares nothing and downloads whole, so a declared list never leaves the model out.
"""

import json
import os
import posixpath
from typing import Dict, List, Optional

from nvflare.apis.job_def import MAX_JOB_OUTPUTS, SERVER_SITE_NAME, JobMetaKey, validate_job_outputs

_ENVIRONMENT_FILE = "environment.json"  # the server app's config/environment.json
_UNREADABLE = object()  # an environment file the job ships but that cannot be read before export


def _normalize(path) -> str:
    """A declared path in the form validate_job_outputs accepts, or an empty string when it cannot be one."""
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


def outputs_of(components: Dict[str, object], app_env: Optional[dict] = None) -> List[str]:
    """The ``job_outputs`` entries the given server components (id -> component) write.

    A path that would not pass validation is skipped, so a recipe never fails because of its own declaration.
    Duplicates and files inside a declared folder are left out.
    """
    component_ids = set(components)
    entries: List[str] = []
    for obj in components.values():
        get_job_outputs = getattr(obj, "get_job_outputs", None)
        if not callable(get_job_outputs):
            continue
        paths = get_job_outputs(app_env, component_ids)
        if paths is None:
            return []  # this component's files are unknown
        for path in paths:
            entry = _normalize(path)
            if entry and entry not in entries:
                entries.append(entry)
    folders = [e for e in entries if e.endswith("/")]
    # a file inside a declared folder is already selected by the folder
    outputs = [e for e in entries if e.endswith("/") or not any(e.startswith(f) for f in folders)]
    return outputs[:MAX_JOB_OUTPUTS]


def server_components(job) -> Dict[str, object]:
    """The components and controllers of a FedJob's server app, by id, before export."""
    server_app = getattr(job, "_deploy_map", {}).get(SERVER_SITE_NAME)
    if server_app is None:
        return {}
    app_config = server_app.app_config
    components = dict(app_config.components)
    components.update({w.id: w.controller for w in getattr(app_config, "workflows", [])})
    return components


def server_environment(job):
    """The server app's ``config/environment.json`` as shipped with the job: a dict, None, or _UNREADABLE."""
    server_app = getattr(job, "_deploy_map", {}).get(SERVER_SITE_NAME)
    if server_app is None:
        return None
    path = None
    for src_path, dest_dir, app_folder_type in getattr(server_app.app_config, "file_sources", []):
        if app_folder_type != "config" or dest_dir:
            continue  # only a file placed directly in the app's config folder is read
        if os.path.isdir(src_path):
            candidate = os.path.join(src_path, _ENVIRONMENT_FILE)
            if os.path.isfile(candidate):
                path = candidate  # a later source overwrites an earlier one at export
        elif os.path.basename(src_path) == _ENVIRONMENT_FILE:
            path = src_path
    if path is None:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            environment = json.load(f)
    except (OSError, ValueError):
        return _UNREADABLE
    return environment if isinstance(environment, dict) else _UNREADABLE


def declare_job_outputs(job, previous: Optional[list]) -> Optional[list]:
    """Set the job's ``job_outputs`` from its server components and return the value set.

    A value the user set with ``set_recipe_meta`` is kept unchanged. ``previous`` is the value an earlier call set,
    so a recipe that is run or exported again recomputes its own declaration and still keeps a user's value.
    """
    job_config = job.job
    key = JobMetaKey.JOB_OUTPUTS.value
    meta_props = job_config.meta_props or {}
    current = meta_props.get(key)
    if current is not None and current is not previous:
        return None  # set by the user
    app_env = server_environment(job)
    # an environment file that cannot be read may move a model; declare nothing rather than guess
    outputs = [] if app_env is _UNREADABLE else outputs_of(server_components(job), app_env)
    if not outputs:
        if current is not None:
            meta_props.pop(key, None)
        return None
    if job_config.meta_props is None:
        job_config.meta_props = {}
    job_config.meta_props[key] = outputs
    return outputs
