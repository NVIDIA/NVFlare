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

"""Record the outputs a job's builder knows it configures.

The code that adds a result-producing server component to a job (a recipe, a model wrapper, a job class) names that
component's files, so it records them here, in the component's own module. A recipe turns the record into the job's
``job_outputs`` when it is exported or run. Paths are relative to the job's run folder; ``app_server/`` is the server
app folder, and a trailing ``/`` names a folder.

Code that adds a component whose files it does not know, such as a user's own model persistor, marks the outputs
unknown. The job then declares nothing and downloads whole, so a declaration never leaves the model out.
"""

import posixpath
import weakref
from typing import List, Optional, Tuple

from nvflare.apis.job_def import SERVER_SITE_NAME

SERVER_APP_FOLDER = "app_" + SERVER_SITE_NAME  # the server app folder inside a job's run folder

_DECLARED = "_declared_job_outputs"
_UNKNOWN = "_job_outputs_unknown"

# components their builder created before the job existed, with the files they save
_TAGGED_COMPONENTS = weakref.WeakKeyDictionary()


def in_server_app(name: Optional[str]) -> Optional[str]:
    """The run-folder path of a file the server app saves in its own folder."""
    return posixpath.join(SERVER_APP_FOLDER, name) if name else None


def as_folder(path: Optional[str]) -> Optional[str]:
    """A folder entry for ``path``; None for the run folder itself, which a folder entry cannot name."""
    if not path:
        return None
    normalized = posixpath.normpath(path.replace("\\", "/"))
    return None if normalized == "." else f"{normalized}/"


def declare_job_outputs(job, *paths: Optional[str]) -> None:
    """Record server workspace paths that are part of the job's result. Empty values are ignored."""
    declared = getattr(job, _DECLARED, None)
    if declared is None:
        declared = []
        setattr(job, _DECLARED, declared)
    declared.extend(p for p in paths if p)


def mark_job_outputs_unknown(job) -> None:
    """Record that the job has a result-producing component whose files are not known."""
    setattr(job, _UNKNOWN, True)


def recorded_job_outputs(job) -> Tuple[List[str], bool]:
    """The recorded paths, and whether some outputs are unknown."""
    return list(getattr(job, _DECLARED, None) or []), bool(getattr(job, _UNKNOWN, False))


def tag_component_outputs(component, *paths: Optional[str]) -> None:
    """Record the files a component saves, for a component created before its job, such as a recipe's persistor
    passed on as ``model_persistor``. Declared when the component is added (see ``setup_custom_persistor``)."""
    _TAGGED_COMPONENTS[component] = [p for p in paths if p]


def tagged_component_outputs(component) -> Optional[List[str]]:
    """The files recorded for a component, or None when its builder recorded none."""
    try:
        return _TAGGED_COMPONENTS.get(component)
    except TypeError:  # not weak-referenceable or not hashable
        return None


def declare_metrics_outputs(job, writer) -> None:
    """Record the files a MetricsArtifactWriter writes: its results folder, or its two files when that folder is the
    run folder itself. Another metrics component may write elsewhere, so the outputs are then marked unknown."""
    from nvflare.app_common.widgets.metrics_artifact_writer import MetricsArtifactWriter

    if type(writer) is not MetricsArtifactWriter:
        mark_job_outputs_unknown(job)
        return
    folder = as_folder(writer.results_dir)
    if folder:
        declare_job_outputs(job, folder)
    else:
        declare_job_outputs(job, writer.summary_file_name, writer.round_file_name)


def declare_np_persistor_outputs(job, persistor) -> None:
    """Record the model file an NPModelPersistor saves: ``<model_dir>/<model_name>`` under the result root, with the
    ".npy" that np.save adds when missing."""
    model_name = persistor.model_name or ""
    if model_name and not model_name.endswith(".npy"):
        model_name += ".npy"
    declare_job_outputs(job, posixpath.join(persistor.model_dir or "", model_name) if model_name else None)
