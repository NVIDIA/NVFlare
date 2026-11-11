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

"""Turn the outputs a recipe recorded while building its job into the job's ``job_outputs``.

The recipe, its model wrapper or its job class records each result file it configures, in the module that knows the
file (see :mod:`nvflare.job_config.job_outputs`). This module holds no framework knowledge: it applies the record when
the recipe is exported or run, so components added after the recipe was built are included.
"""

import os
import posixpath
from typing import List, Optional

from nvflare.apis.job_def import MAX_JOB_OUTPUTS, SERVER_SITE_NAME, JobMetaKey, validate_job_outputs
from nvflare.job_config.job_outputs import recorded_job_outputs

_ENVIRONMENT_FILE = "environment.json"  # the server app's config/environment.json


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


def outputs_from(paths: List[str]) -> List[str]:
    """The ``job_outputs`` entries for recorded paths.

    A path that would not pass validation is skipped, so a recipe never fails because of its own declaration.
    Duplicates and files inside a declared folder are left out. More than MAX_JOB_OUTPUTS entries declare nothing,
    since a truncated list could leave the model out.
    """
    entries: List[str] = []
    for path in paths:
        entry = _normalize(path)
        if entry and entry not in entries:
            entries.append(entry)
    folders = [e for e in entries if e.endswith("/")]
    # a file inside a declared folder is already selected by the folder
    outputs = [e for e in entries if e.endswith("/") or not any(e.startswith(f) for f in folders)]
    return [] if len(outputs) > MAX_JOB_OUTPUTS else outputs


def ships_server_environment(job) -> bool:
    """Whether the job ships a ``config/environment.json`` for its server app.

    Such a file can move where server components write (for example a checkpoint folder), so the recorded paths may
    not hold.
    """
    server_app = getattr(job, "_deploy_map", {}).get(SERVER_SITE_NAME)
    if server_app is None:
        return False
    for src_path, dest_dir, app_folder_type in getattr(server_app.app_config, "file_sources", []):
        if app_folder_type != "config" or (dest_dir and os.path.normpath(dest_dir) != "."):
            continue  # only a file placed directly in the app's config folder ("." included) is read
        if os.path.isdir(src_path):
            if os.path.isfile(os.path.join(src_path, _ENVIRONMENT_FILE)):
                return True
        elif os.path.basename(src_path) == _ENVIRONMENT_FILE:
            return True
    return False


def declare_job_outputs(job, previous: Optional[list]) -> Optional[list]:
    """Set the job's ``job_outputs`` from what its builder recorded, and return the value set.

    A value the user set with ``set_recipe_meta`` is kept unchanged. ``previous`` is the value an earlier call set,
    so a recipe that is run or exported again recomputes its own declaration and still keeps a user's value. When
    some outputs are unknown, or the job ships a server environment file, nothing is declared and the job downloads
    whole.
    """
    job_config = job.job
    key = JobMetaKey.JOB_OUTPUTS.value
    meta_props = job_config.meta_props or {}
    current = meta_props.get(key)
    if current is not None and current is not previous:
        return None  # set by the user
    paths, unknown = recorded_job_outputs(job)
    outputs = [] if unknown or ships_server_environment(job) else outputs_from(paths)
    if not outputs:
        if current is not None:
            meta_props.pop(key, None)
        return None
    if job_config.meta_props is None:
        job_config.meta_props = {}
    job_config.meta_props[key] = outputs
    return outputs
