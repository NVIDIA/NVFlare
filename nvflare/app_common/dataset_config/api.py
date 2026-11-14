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

"""The site dataset configuration API, implementing nvflare.apis.dataset_config_spec.

It currently supports loading the site's configuration file. Only ``_read_document`` knows the file format (YAML
today); everything else validates and serves the parsed document, so a different format changes that one function.
"""

import hashlib
import os
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import unquote, urlsplit

import yaml

from nvflare.apis.dataset_config_spec import (
    DATASET_CONFIG_ENV,
    DATASET_CONFIG_FILE,
    DatasetConfig,
    DatasetConfigError,
    StudyDatasetConfig,
    UnknownDatasetError,
)

FORMAT_VERSION = 1
# the naming rule of Study runtime datasets (nvflare/app_opt/job_launcher/study_runtime.py)
_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,61}[a-z0-9])?$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*$")
_TOP_KEYS = frozenset({"format_version", "studies", "usage"})
_STUDY_KEYS = frozenset({"datasets"})
_DATASET_KEYS = frozenset({"type", "uri", "options", "secrets", "usage"})


def _freeze(value):
    """A read-only copy: mappings become read-only views, lists become tuples and sets frozensets, recursively."""
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(value)
    return value


def _names(keys) -> list:
    """Keys sorted for an error message; YAML keys need not all be strings, so sort by their text."""
    return sorted(keys, key=str)


def _error(source: str, message: str) -> DatasetConfigError:
    return DatasetConfigError(f"invalid dataset configuration '{source}': {message}")


def _mapping(value, label: str, source: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise _error(source, f"{label} must be a mapping")
    return value


def _check_uri(uri, label: str, source: str) -> str | None:
    """Validate a dataset URI; return the decoded local path of a file URI, else None."""
    if not isinstance(uri, str) or not uri:
        raise _error(source, f"{label}.uri is required")
    try:
        parts = urlsplit(uri)
    except ValueError:
        # e.g. an unclosed IPv6 host; the URI is not echoed, as it may hold credentials
        raise _error(source, f"{label}.uri is not a valid URI") from None
    if not parts.scheme or not _SCHEME.match(parts.scheme):
        raise _error(source, f"{label}.uri must be an absolute URI with a scheme")
    if parts.password is not None:
        raise _error(source, f"{label}.uri must not contain a password; put credentials in secrets")
    if parts.scheme.lower() != "file":
        return None
    if parts.netloc or parts.query or parts.fragment:
        raise _error(source, f"{label}.uri: a file URI has no host, query or fragment (use file:///path)")
    path = unquote(parts.path)
    if not path.startswith("/"):
        raise _error(source, f"{label}.uri: a file URI needs an absolute path")
    return path


def _dataset(study: str, name, entry, source: str) -> tuple[DatasetConfig, dict]:
    if not isinstance(name, str) or not _NAME.match(name):
        raise _error(source, f"study '{study}' has a dataset name that breaks the naming rule")
    label = f"studies.{study}.datasets.{name}"
    if not isinstance(entry, dict):
        raise _error(source, f"{label} must be a mapping")
    unknown = set(entry) - _DATASET_KEYS
    if unknown:
        raise _error(source, f"{label} has unsupported keys {_names(unknown)}")
    dataset_type = entry.get("type")
    if not isinstance(dataset_type, str) or not dataset_type:
        raise _error(source, f"{label}.type is required")
    path = _check_uri(entry.get("uri"), label, source)
    options = _mapping(entry.get("options"), f"{label}.options", source)
    secrets = _mapping(entry.get("secrets"), f"{label}.secrets", source)
    for key, variable in secrets.items():
        if not isinstance(key, str) or not key:
            raise _error(source, f"{label}.secrets keys must be non-empty strings")
        if not isinstance(variable, str) or not _ENV_NAME.match(variable):
            raise _error(source, f"{label}.secrets.{key} must name an environment variable, not hold a value")
    usage = _mapping(entry.get("usage"), f"{label}.usage", source)
    for component in usage:
        if not isinstance(component, str) or not component:
            raise _error(source, f"{label}.usage keys must be non-empty component names")
    return DatasetConfig(
        study=study,
        name=name,
        type=dataset_type,
        uri=entry["uri"],
        path=path,
        options=_freeze(options),
        secret_keys=tuple(secrets),
        usage=_freeze(usage),
    ), dict(secrets)


class DatasetConfigAPI(StudyDatasetConfig):
    """The dataset configuration API for one Study: immutable and validated when it is built."""

    def __init__(self, study: str, datasets: dict, secrets: dict, site_usage: Mapping, revision: str):
        self._study = study
        self._datasets = datasets  # name -> DatasetConfig, in file order
        self._secrets = secrets  # name -> {key: variable name}
        self._site_usage = site_usage
        self._revision = revision

    @property
    def study(self) -> str:
        return self._study

    @property
    def revision(self) -> str:
        return self._revision

    def list_datasets(self) -> list[str]:
        return list(self._datasets)

    def get_dataset(self, name: str) -> DatasetConfig:
        try:
            return self._datasets[name]
        except KeyError:
            raise UnknownDatasetError(f"study '{self._study}' has no dataset '{name}'") from None

    def site_usage(self, component: str) -> Mapping[str, Any] | None:
        return self._site_usage.get(component)

    def get_secret(self, name: str, key: str) -> str:
        self.get_dataset(name)
        variable = self._secrets[name].get(key)
        if variable is None:
            raise DatasetConfigError(f"dataset '{name}' has no secret '{key}'")
        value = os.environ.get(variable)
        if value is None:
            raise DatasetConfigError(f"secret '{key}' of dataset '{name}' is not set in this job's environment")
        return value


def parse_study_config(data, study: str, source: str, revision: str = "") -> DatasetConfigAPI:
    """Validate a whole parsed file and return the snapshot of one Study. Other Studies are validated too, so one
    broken entry anywhere makes the file invalid, as the launcher sees it."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise _error(source, "must be a mapping")
    unknown = set(data) - _TOP_KEYS
    if unknown:
        raise _error(source, f"unsupported keys {_names(unknown)}")
    version = data.get("format_version")
    if type(version) is not int or version != FORMAT_VERSION:
        raise _error(source, f"format_version must be {FORMAT_VERSION}")
    site_usage = _mapping(data.get("usage"), "usage", source)
    studies = _mapping(data.get("studies"), "studies", source)
    selected, selected_secrets = {}, {}
    for name, entry in studies.items():
        if not isinstance(name, str) or not _NAME.match(name):
            raise _error(source, "a Study name breaks the naming rule")
        entry = _mapping(entry, f"studies.{name}", source)
        unknown = set(entry) - _STUDY_KEYS
        if unknown:
            raise _error(
                source, f"studies.{name} has unsupported keys {_names(unknown)}; a Study entry holds only datasets"
            )
        datasets, secrets = {}, {}
        for dataset_name, dataset_entry in _mapping(entry.get("datasets"), f"studies.{name}.datasets", source).items():
            datasets[dataset_name], secrets[dataset_name] = _dataset(name, dataset_name, dataset_entry, source)
        if name == study:
            selected, selected_secrets = datasets, secrets
    return DatasetConfigAPI(study, selected, selected_secrets, _freeze(site_usage), revision)


def _read_document(raw: bytes, source: str):
    """Decode the file into plain mappings, lists and scalars. The only format-specific code in this module."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        raise _error(source, "not valid YAML") from None


def load_study_config(path: str, study: str, missing_ok: bool) -> DatasetConfigAPI:
    """Read one file. A missing file yields an empty snapshot only when missing_ok."""
    try:
        raw = Path(path).read_bytes()
    except FileNotFoundError:
        if missing_ok:
            return DatasetConfigAPI(study, {}, {}, MappingProxyType({}), "")
        raise DatasetConfigError(f"dataset configuration '{path}' does not exist") from None
    except OSError as e:
        raise DatasetConfigError(f"cannot read dataset configuration '{path}': {type(e).__name__}") from None
    return parse_study_config(_read_document(raw, path), study, path, hashlib.sha256(raw).hexdigest())


def load_job_study_config(fl_ctx) -> DatasetConfigAPI:
    """The snapshot for the job in fl_ctx. See nvflare.apis.dataset_config_spec.get_study_dataset_config."""
    from nvflare.apis.fl_constant import FLContextKey
    from nvflare.apis.job_def import get_job_meta_study

    study = get_job_meta_study(fl_ctx.get_prop(FLContextKey.JOB_META))
    delivered = os.environ.get(DATASET_CONFIG_ENV)
    if delivered is not None:
        if not delivered or not os.path.isabs(delivered):
            raise DatasetConfigError(f"{DATASET_CONFIG_ENV} must be an absolute path")
        return load_study_config(delivered, study, missing_ok=False)
    engine = fl_ctx.get_engine()
    workspace = engine.get_workspace() if engine is not None else None
    if workspace is None:
        raise DatasetConfigError("the site workspace is unavailable")
    return load_study_config(os.path.join(workspace.get_root_dir(), "local", DATASET_CONFIG_FILE), study, True)
