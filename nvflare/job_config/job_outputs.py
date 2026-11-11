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
from typing import Callable, List, Optional, Tuple

from nvflare.apis.job_def import SERVER_SITE_NAME

SERVER_APP_FOLDER = "app_" + SERVER_SITE_NAME  # the server app folder inside a job's run folder

_DECLARED = "_declared_job_outputs"
_SUPPORTING = "_declared_supporting_job_outputs"
_UNKNOWN = "_job_outputs_unknown"
_ACCOUNTED = "_job_outputs_accounted_components"  # ids of components whose files were recorded

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


def declare_job_outputs(job, *paths: Optional[str], supporting: bool = False) -> None:
    """Record server workspace paths that are part of the job's result. Empty values are ignored.

    ``supporting`` marks outputs that come with a result but are not one, such as training metrics. A job whose
    record holds only supporting outputs declares nothing: its main result, such as a model saved by a controller,
    was not recorded, and a declaration without it would leave it out of an outputs-only download.
    """
    attribute = _SUPPORTING if supporting else _DECLARED
    declared = getattr(job, attribute, None)
    if declared is None:
        declared = []
        setattr(job, attribute, declared)
    declared.extend(p for p in paths if p)


def account_component(job, component) -> None:
    """Record that ``component``'s files are covered by the job's record.

    At export, a model persistor that no declaration accounted for makes the job declare nothing, so a missed
    declaration fails safe to a whole-job download.
    """
    accounted = getattr(job, _ACCOUNTED, None)
    if accounted is None:
        accounted = set()
        setattr(job, _ACCOUNTED, accounted)
    accounted.add(id(component))


def is_accounted(job, component) -> bool:
    """Whether ``component``'s files are covered by the job's record."""
    return id(component) in (getattr(job, _ACCOUNTED, None) or ())


def mark_job_outputs_unknown(job) -> None:
    """Record that the job has a result-producing component whose files are not known."""
    setattr(job, _UNKNOWN, True)


def recorded_job_outputs(job) -> Tuple[List[str], bool]:
    """The recorded paths, and whether some outputs are unknown.

    Supporting outputs are included only beside a main result; on their own they mean the main result is unknown.
    """
    main = list(getattr(job, _DECLARED, None) or [])
    supporting = list(getattr(job, _SUPPORTING, None) or [])
    unknown = bool(getattr(job, _UNKNOWN, False))
    if supporting and not main:
        return [], True
    return main + supporting, unknown


def tag_component_outputs(component, outputs: Callable[[object], List[Optional[str]]]) -> None:
    """Record how to name the files a component saves, for a component created before its job, such as a recipe's
    persistor passed on as ``model_persistor``.

    ``outputs`` is called with the component when it is added to the job (see ``setup_custom_persistor``), so the
    paths reflect its settings at that time. A copy of the component carries no tag, which leaves the job undeclared.
    """
    _TAGGED_COMPONENTS[component] = outputs


def tagged_component_outputs(component) -> Optional[List[str]]:
    """The files a tagged component saves, or None when its builder tagged none."""
    try:
        outputs = _TAGGED_COMPONENTS.get(component)
    except TypeError:  # not weak-referenceable or not hashable
        return None
    return None if outputs is None else [p for p in outputs(component) if p]


def declare_metrics_outputs(job, writer) -> None:
    """Record the files a MetricsArtifactWriter writes: its results folder, or its two files when that folder is the
    run folder itself. Another metrics component may write elsewhere, so the outputs are then marked unknown."""
    from nvflare.app_common.widgets.metrics_artifact_writer import MetricsArtifactWriter

    if type(writer) is not MetricsArtifactWriter:
        mark_job_outputs_unknown(job)
        return
    folder = as_folder(writer.results_dir)
    if folder:
        declare_job_outputs(job, folder, supporting=True)
    else:
        declare_job_outputs(job, writer.summary_file_name, writer.round_file_name, supporting=True)


def declare_np_persistor_outputs(job, persistor) -> None:
    """Record the model file an NPModelPersistor saves: ``<model_dir>/<model_name>`` under the result root, with the
    ".npy" that np.save adds when missing."""
    account_component(job, persistor)
    model_name = persistor.model_name or ""
    if model_name and not model_name.endswith(".npy"):
        model_name += ".npy"
    declare_job_outputs(job, posixpath.join(persistor.model_dir or "", model_name) if model_name else None)
