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

Each built-in result-producing server component writes to a path set by its own arguments. This module reads those
arguments and turns them into ``job_outputs`` entries: paths relative to the server run folder, which is the root of
the job's workspace archive. Components are matched by class name, so no framework (PyTorch, TensorFlow, XGBoost) is
imported to inspect a job. A component this module does not know adds nothing, and a path that would not pass
:func:`nvflare.apis.job_def.validate_job_outputs` is skipped: a recipe never fails because of its own declaration.
"""

import json
import os
import posixpath
from typing import Callable, Dict, List, Optional

from nvflare.apis.job_def import MAX_JOB_OUTPUTS, SERVER_SITE_NAME, JobMetaKey, validate_job_outputs

_SERVER_APP_DIR = "app_" + SERVER_SITE_NAME  # the server app folder inside the run folder
_ENVIRONMENT_FILE = "environment.json"  # the app's config/environment.json, read by PTFileModelPersistor
_UNREADABLE = object()  # an environment file the job ships but that cannot be read before export


class _Undeclarable(Exception):
    """A component's output paths cannot be determined, so the job declares nothing."""


class _JobContext:
    def __init__(self, ids, environment=None):
        self.ids = set(ids)  # the server component IDs
        self.environment = environment  # the server app's environment.json: a dict, None, or _UNREADABLE


def _in_app(name: str) -> str:
    return posixpath.join(_SERVER_APP_DIR, name) if name else ""


def _pt_persistor(obj, ctx: _JobContext) -> List[str]:
    names = [getattr(obj, "global_model_file_name", ""), getattr(obj, "best_global_model_file_name", "")]
    # The app's environment.json can name a checkpoint folder (APP_CKPT_DIR) inside the app folder.
    if ctx.environment is _UNREADABLE:
        raise _Undeclarable()
    folder = (ctx.environment or {}).get(getattr(obj, "ckpt_dir_env_key", "APP_CKPT_DIR"))
    if not folder:
        return [_in_app(n) for n in names]
    if not isinstance(folder, str):
        raise _Undeclarable()
    paths = [_in_app(posixpath.join(folder, n)) for n in names if n]
    if not all(_normalize(p) for p in paths):
        raise _Undeclarable()  # e.g. an absolute folder outside the app
    return paths


def _save_name_persistor(obj, ctx: _JobContext) -> List[str]:
    return [_in_app(getattr(obj, "save_name", ""))]


def _np_persistor(obj, ctx: _JobContext) -> List[str]:
    # saved under the result root, whose files sit at the archive root; np.save adds ".npy" when it is missing
    name = getattr(obj, "model_name", "") or ""
    if name and not name.endswith(".npy"):
        name += ".npy"
    return [posixpath.join(getattr(obj, "model_dir", "") or "", name)]


def _metrics_writer(obj, ctx: _JobContext) -> List[str]:
    return [_folder(getattr(obj, "results_dir", ""))]


def _stats_writer(obj, ctx: _JobContext) -> List[str]:
    return [getattr(obj, "output_path", "")]


def _fedavg_controller(obj, ctx: _JobContext) -> List[str]:
    # When its persistor_id names no component, FedAvg runs without a persistor and saves the model itself,
    # relative to the run folder.
    if getattr(obj, "_persistor_id", None) in ctx.ids:
        return []
    return [getattr(obj, "save_filename", None) or ""]


def _cross_site_eval(obj, ctx: _JobContext) -> List[str]:
    return [_folder(getattr(obj, "_cross_val_dir", ""))]


def _folder(path: str) -> str:
    return f"{path.rstrip('/')}/" if path else ""


# fully qualified class name -> the outputs one instance of it writes
_PRODUCERS: Dict[str, Callable[[object, _JobContext], List[str]]] = {
    "nvflare.app_opt.pt.file_model_persistor.PTFileModelPersistor": _pt_persistor,
    "nvflare.app_opt.tf.model_persistor.TFModelPersistor": _save_name_persistor,
    "nvflare.app_opt.sklearn.joblib_model_param_persistor.JoblibModelParamPersistor": _save_name_persistor,
    "nvflare.app_opt.xgboost.tree_based.model_persistor.XGBModelPersistor": _save_name_persistor,
    "nvflare.app_common.np.np_model_persistor.NPModelPersistor": _np_persistor,
    "nvflare.app_common.widgets.metrics_artifact_writer.MetricsArtifactWriter": _metrics_writer,
    "nvflare.app_common.statistics.json_stats_file_persistor.JsonStatsFileWriter": _stats_writer,
    "nvflare.app_common.workflows.fedavg.FedAvg": _fedavg_controller,
    "nvflare.app_common.workflows.cross_site_model_eval.CrossSiteModelEval": _cross_site_eval,
    "nvflare.app_common.workflows.cross_site_eval.CrossSiteEval": _cross_site_eval,
}


def _producer(obj) -> Optional[Callable[[object, _JobContext], List[str]]]:
    for cls in type(obj).__mro__:  # a subclass writes where its base class does
        producer = _PRODUCERS.get(f"{cls.__module__}.{cls.__qualname__}")
        if producer:
            return producer
    return None


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


def _is_model_persistor(obj) -> bool:
    from nvflare.app_common.abstract.model_persistor import ModelPersistor

    return isinstance(obj, ModelPersistor)


def outputs_of(components: Dict[str, object], environment=None) -> List[str]:
    """The ``job_outputs`` entries the given server components (id -> component) write.

    ``environment`` is the server app's ``environment.json`` (a dict, or None when the job ships none).

    Returns no entries when a model's location cannot be determined: a model persistor this module does not know,
    or a checkpoint folder it cannot resolve. A declaration without the model would leave it out of an
    outputs-only download, so the job then downloads whole unless the user declares its outputs. Duplicates and
    files inside a declared folder are left out.
    """
    ctx = _JobContext(components, environment)
    entries: List[str] = []
    for obj in components.values():
        producer = _producer(obj)
        if not producer:
            if _is_model_persistor(obj):
                return []
            continue
        try:
            paths = producer(obj, ctx)
        except _Undeclarable:
            return []
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
    outputs = outputs_of(server_components(job), server_environment(job))
    if not outputs:
        if current is not None:
            meta_props.pop(key, None)
        return None
    if job_config.meta_props is None:
        job_config.meta_props = {}
    job_config.meta_props[key] = outputs
    return outputs
