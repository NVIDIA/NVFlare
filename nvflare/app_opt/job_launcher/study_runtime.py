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
"""Site-owned per-study runtime configuration (v2): local/study_runtime.yaml.

Filename selects the format: this strict v2 parser never reads the frozen v1
local/study_data.yaml (see study_data.py). Both files present is a launcher
error, enforced by the launchers, not here.
"""

from __future__ import annotations

import logging
import os
import posixpath
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Optional

import yaml

from nvflare.apis.job_launcher_spec import JobProcessEnv
from nvflare.app_common.multinode import CONTRACT_ENV_NAMES
from nvflare.app_opt.job_launcher.study_data import MODE_RO, MODE_RW, StudyDatasetMount

STUDY_RUNTIME_FILE = "local/study_runtime.yaml"
SUPPORTED_FORMAT_VERSION = 2

DATASET_TYPE_MOUNT = "mount"
_RESERVED_DATASET_TYPES = {"databricks"}

_STUDY_KEYS = {
    "container",
    "pod_template",
    "docker_kwargs",
    "datasets",
    "env",
    "secret_env",
    "secret_mounts",
    "slurm",
}
_MOUNT_DATASET_KEYS = {"type", "source", "mode"}
_SECRET_ENV_KEYS = {"source", "key"}
_SECRET_MOUNT_KEYS = {"source", "mount_path", "mode", "items"}
_SLURM_KEYS = {"sandbox", "setup", "partition", "account", "qos"}
SLURM_SANDBOXES = frozenset({"apptainer", "pyxis", "none"})
_VALID_LAUNCHER_MODES = {"process", "docker", "k8s", "slurm"}

_VALID_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,61}[a-z0-9])?$")
_VALID_POSIX_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Env names the launchers own at job launch (PYTHONPATH, the workspace-transfer
# variables from workspace_cell_transfer, and the job bootstrap credentials). A study
# value would silently override or be overridden by the launcher-supplied one, so both
# env and secret_env reject them up front.
_RESERVED_ENV_NAMES = frozenset(
    {
        "PYTHONPATH",
        "NVFL_WORKSPACE_OWNER_FQCN",
        "NVFL_WORKSPACE_TRANSFER_TOKEN",
        JobProcessEnv.AUTH_TOKEN,
        JobProcessEnv.TOKEN_SIGNATURE,
        JobProcessEnv.SSID,
    }
)
SLURM_RESERVED_ENV_NAMES = _RESERVED_ENV_NAMES.union(
    {"BASHOPTS", "EUID", "SHELLOPTS", "UID", "NVFL_APPTAINER", "NVFL_SRUN"},
    CONTRACT_ENV_NAMES,
)
SLURM_RESERVED_ENV_PREFIXES = (
    "SLURM_",
    "APPTAINER_",
    "APPTAINERENV_",
    "SINGULARITY_",
    "SINGULARITYENV_",
    "NVFLARE_SLURM_",
    "_nvfl_",
)

# Docker containers.run kwargs owned by the launcher (the docker launcher and deploy
# validation enforce the same set for site-level default_job_container_kwargs).
RESERVED_DOCKER_KWARGS = frozenset(
    {
        "volumes",
        "mounts",
        "network",
        "environment",
        "command",
        "name",
        "detach",
        "auto_remove",
        "user",
        "working_dir",
        "image",
    }
)


@dataclass(frozen=True)
class SecretEnvRef:
    name: str
    source: str
    key: Optional[str] = None


@dataclass(frozen=True)
class SecretMount:
    study: str
    name: str
    source: str
    mount_path: str
    items: Optional[tuple] = None  # tuple of (key, path) pairs; None = full projection


@dataclass
class StudyRuntime:
    study: str
    datasets: list = field(default_factory=list)  # list[StudyDatasetMount]
    env: dict = field(default_factory=dict)
    secret_env: list = field(default_factory=list)  # list[SecretEnvRef]
    secret_mounts: list = field(default_factory=list)  # list[SecretMount]
    container_image: Optional[str] = None
    pod_template: Optional[dict] = None
    docker_kwargs: dict = field(default_factory=dict)
    slurm: dict = field(default_factory=dict)


def study_runtime_file_path(workspace_root: str) -> str:
    return os.path.join(workspace_root, *STUDY_RUNTIME_FILE.split("/"))


def _error(file_path: str, message: str) -> ValueError:
    return ValueError(f"study runtime file '{file_path}': {message}")


def _require_dict(value, label: str, file_path: str) -> dict:
    if not isinstance(value, dict):
        raise _error(file_path, f"{label} must be a mapping.")
    return value


def _require_known_keys(entry: dict, known: set, label: str, file_path: str) -> None:
    unknown = set(entry.keys()) - known
    if unknown:
        raise _error(file_path, f"{label} has unknown key(s) {sorted(unknown)}; allowed: {sorted(known)}.")


def _require_name(value, label: str, file_path: str) -> str:
    if not isinstance(value, str) or not _VALID_NAME.match(value):
        raise _error(file_path, f"{label} {value!r} is not a valid name.")
    return value


def _require_str(value, label: str, file_path: str) -> str:
    if not isinstance(value, str) or not value:
        raise _error(file_path, f"{label} must be a non-empty string.")
    return value


def _require_utf8_text(value, label: str, file_path: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not value and not allow_empty):
        requirement = "a string" if allow_empty else "a non-empty string"
        raise _error(file_path, f"{label} must be {requirement}.")
    if "\x00" in value:
        raise _error(file_path, f"{label} must not contain NUL.")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as e:
        raise _error(file_path, f"{label} must be valid Unicode text encodable as UTF-8.") from e
    return value


def _require_env_name(value, label: str, file_path: str, launcher_mode: str) -> str:
    if launcher_mode == "slurm":
        if not isinstance(value, str) or not _VALID_POSIX_ENV_NAME.fullmatch(value):
            raise _error(file_path, f"{label} {value!r} must be a valid POSIX environment variable name.")
        reserved = value in SLURM_RESERVED_ENV_NAMES or value.startswith(SLURM_RESERVED_ENV_PREFIXES)
    else:
        _require_str(value, label, file_path)
        reserved = value in _RESERVED_ENV_NAMES
    if reserved:
        raise _error(file_path, f"{label} {value!r} is launcher-owned and cannot be set in study config.")
    return value


def _require_slurm_absolute_path(value, label: str, file_path: str, mount_operand: bool = False) -> str:
    value = _require_utf8_text(value, label, file_path)
    if not posixpath.isabs(value):
        raise _error(file_path, f"{label} must be an absolute path.")
    if "\n" in value or "\r" in value or (mount_operand and any(c in value for c in (":", ","))):
        forbidden = "newline, ':' or ','" if mount_operand else "newline"
        raise _error(file_path, f"{label} must not contain {forbidden}.")
    return value


def _safe_relative_path(value: str, label: str, file_path: str) -> str:
    normalized = posixpath.normpath(value.replace(os.sep, "/"))
    parts = PurePosixPath(normalized)
    if parts.is_absolute() or ".." in parts.parts or normalized in ("", "."):
        raise _error(file_path, f"{label} must be a relative path under the config directory: {value!r}")
    return normalized


def _load_pod_template_file(template_path: str, file_path: str) -> dict:
    try:
        with open(template_path, "rt") as f:
            pod_template = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise _error(file_path, f"pod_template file '{template_path}' was not found.") from e
    except OSError as e:
        raise _error(file_path, f"could not read pod_template file '{template_path}': {e}") from e
    except yaml.YAMLError as e:
        raise _error(file_path, f"could not parse pod_template file '{template_path}': {e}") from e
    return _validate_pod_template(pod_template, f"pod_template file '{template_path}'", file_path)


def _validate_pod_template(pod_template, label: str, file_path: str) -> dict:
    pod_template = _require_dict(pod_template, label, file_path)
    kind = pod_template.get("kind")
    if kind and kind != "Pod":
        raise _error(file_path, f"{label} must define kind: Pod.")
    return pod_template


def _parse_pod_template(value, file_path: str, launcher_mode: str) -> dict:
    if launcher_mode != "k8s":
        raise _error(file_path, "pod_template is Kubernetes-only and is not supported by this launcher.")
    if isinstance(value, str):
        rel_path = _safe_relative_path(_require_str(value, "pod_template", file_path), "pod_template", file_path)
        template_path = os.path.join(os.path.dirname(file_path), *rel_path.split("/"))
        return _load_pod_template_file(template_path, file_path)
    return _validate_pod_template(value, "inline pod_template", file_path)


def _parse_docker_kwargs(study: str, entry, file_path: str, launcher_mode: str) -> dict:
    label = f"studies.{study}.docker_kwargs"
    if launcher_mode != "docker":
        raise _error(file_path, "docker_kwargs is Docker-only and is not supported by this launcher.")
    entry = _require_dict(entry, label, file_path)
    reserved = sorted(RESERVED_DOCKER_KWARGS & set(entry))
    if reserved:
        raise _error(file_path, f"{label} must not contain launcher-owned key(s) {reserved}.")
    for key in entry:
        _require_str(key, f"{label} key", file_path)
    return dict(entry)


def _parse_container(study: str, entry, file_path: str, launcher_mode: str) -> str:
    label = f"studies.{study}.container"
    entry = _require_dict(entry, label, file_path)
    _require_known_keys(entry, {"image"}, label, file_path)
    image = entry.get("image")
    if launcher_mode == "slurm":
        return _require_slurm_absolute_path(image, f"{label}.image", file_path)
    return _require_str(image, f"{label}.image", file_path)


def _parse_dataset(study: str, dataset: str, entry, file_path: str, launcher_mode: str) -> StudyDatasetMount:
    label = f"studies.{study}.datasets.{dataset}"
    _require_name(dataset, f"{label} dataset name", file_path)
    entry = _require_dict(entry, label, file_path)
    dataset_type = entry.get("type", DATASET_TYPE_MOUNT)
    if dataset_type in _RESERVED_DATASET_TYPES:
        raise _error(file_path, f"{label} type '{dataset_type}' is not yet supported.")
    if dataset_type != DATASET_TYPE_MOUNT:
        raise _error(file_path, f"{label} has unknown type {dataset_type!r}; allowed: '{DATASET_TYPE_MOUNT}'.")
    _require_known_keys(entry, _MOUNT_DATASET_KEYS, label, file_path)
    source = entry.get("source")
    if launcher_mode == "slurm":
        source = _require_slurm_absolute_path(source, f"{label}.source", file_path, mount_operand=True)
    else:
        source = _require_str(source, f"{label}.source", file_path)
    mode = entry.get("mode")
    if mode not in (MODE_RO, MODE_RW):
        raise _error(file_path, f"{label}.mode must be '{MODE_RO}' or '{MODE_RW}'.")
    return StudyDatasetMount(study=study, dataset=dataset, source=source, mode=mode)


def _parse_env(study: str, entry, file_path: str, launcher_mode: str) -> dict:
    entry = _require_dict(entry, f"studies.{study}.env", file_path)
    env = {}
    for name, value in entry.items():
        _require_env_name(name, f"studies.{study}.env variable name", file_path, launcher_mode)
        if isinstance(value, (dict, list)) or value is None:
            raise _error(file_path, f"studies.{study}.env.{name} must be a scalar value.")
        if isinstance(value, bool):
            # str(True) is "True"; YAML users writing `true` expect "true"
            value = "true" if value else "false"
        value = str(value)
        if launcher_mode == "slurm":
            value = _require_utf8_text(value, f"studies.{study}.env.{name}", file_path, allow_empty=True)
        elif not value:
            # empty values behave differently per launcher and can silently lose
            # against pod-template entries in the manifest merge
            raise _error(file_path, f"studies.{study}.env.{name} must not be empty; set a value or remove the key.")
        env[name] = value
    return env


def _parse_secret_env(study: str, entry, file_path: str, launcher_mode: str) -> list:
    entry = _require_dict(entry, f"studies.{study}.secret_env", file_path)
    refs = []
    for name, ref in entry.items():
        label = f"studies.{study}.secret_env.{name}"
        _require_env_name(name, f"{label} variable name", file_path, launcher_mode)
        ref = _require_dict(ref, label, file_path)
        _require_known_keys(ref, _SECRET_ENV_KEYS, label, file_path)
        source = ref.get("source")
        if launcher_mode == "slurm":
            source = _require_env_name(source, f"{label}.source", file_path, launcher_mode)
            key = ref.get("key")
            if key is not None:
                key = _require_str(key, f"{label}.key", file_path)
        else:
            source = _require_str(source, f"{label}.source", file_path)
            key = _require_str(ref.get("key"), f"{label}.key", file_path)
        refs.append(
            SecretEnvRef(
                name=name,
                source=source,
                key=key,
            )
        )
    return refs


def _parse_secret_mounts(
    study: str,
    entry,
    file_path: str,
    launcher_mode: str,
) -> list:
    entry = _require_dict(entry, f"studies.{study}.secret_mounts", file_path)
    mounts = []
    for name, mount in entry.items():
        label = f"studies.{study}.secret_mounts.{name}"
        _require_name(name, f"{label} mount name", file_path)
        mount = _require_dict(mount, label, file_path)
        _require_known_keys(mount, _SECRET_MOUNT_KEYS, label, file_path)
        mode = mount.get("mode", MODE_RO)
        if mode != MODE_RO:
            raise _error(file_path, f"{label}.mode must be '{MODE_RO}'; secret mounts are always read-only.")
        mount_path = mount.get("mount_path")
        if launcher_mode == "slurm":
            mount_path = _require_slurm_absolute_path(mount_path, f"{label}.mount_path", file_path, mount_operand=True)
        else:
            mount_path = _require_str(mount_path, f"{label}.mount_path", file_path)
            if not posixpath.isabs(mount_path):
                raise _error(file_path, f"{label}.mount_path must be an absolute path.")
        items = mount.get("items")
        if items is not None:
            if launcher_mode != "k8s":
                raise _error(
                    file_path,
                    f"{label}.items is Kubernetes-only and is not supported by this launcher; "
                    "point source at a directory containing only the intended files.",
                )
            items = _require_dict(items, f"{label}.items", file_path)
            if not items:
                raise _error(file_path, f"{label}.items must not be empty; omit it for full projection.")
            items = tuple(
                (
                    _require_str(key, f"{label}.items key", file_path),
                    _require_str(path, f"{label}.items.{key}", file_path),
                )
                for key, path in items.items()
            )
        source = mount.get("source")
        if launcher_mode == "slurm":
            source = _require_slurm_absolute_path(source, f"{label}.source", file_path, mount_operand=True)
        else:
            source = _require_str(source, f"{label}.source", file_path)
        mounts.append(
            SecretMount(
                study=study,
                name=name,
                source=source,
                mount_path=mount_path,
                items=items,
            )
        )
    return mounts


def _parse_slurm(study: str, entry, file_path: str) -> dict:
    label = f"studies.{study}.slurm"
    entry = _require_dict(entry, label, file_path)
    _require_known_keys(entry, _SLURM_KEYS, label, file_path)
    result = {}
    for key, value in entry.items():
        key_label = f"{label}.{key}"
        if key == "sandbox":
            if not isinstance(value, str) or value not in SLURM_SANDBOXES:
                raise _error(file_path, f"{key_label} must be one of {sorted(SLURM_SANDBOXES)}.")
        elif key == "setup":
            value = _require_utf8_text(value, key_label, file_path, allow_empty=True)
        else:
            if isinstance(value, bool) or not isinstance(value, (str, int)) or value == "":
                raise _error(file_path, f"{key_label} must be a non-empty string or integer.")
            if isinstance(value, str):
                value = _require_utf8_text(value, key_label, file_path)
                if any(character.isspace() for character in value):
                    raise _error(file_path, f"{key_label} must not contain whitespace.")
        result[key] = value
    return result


def _parse_study(
    study: str,
    entry,
    file_path: str,
    launcher_mode: str,
) -> StudyRuntime:
    _require_name(study, "study name", file_path)
    entry = _require_dict(entry, f"studies.{study}", file_path)
    _require_known_keys(entry, _STUDY_KEYS, f"studies.{study}", file_path)
    if "slurm" in entry and launcher_mode != "slurm":
        raise _error(file_path, "slurm is Slurm-only and is not supported by this launcher.")

    runtime = StudyRuntime(study=study)
    if "container" in entry:
        runtime.container_image = _parse_container(study, entry["container"], file_path, launcher_mode)
    if "pod_template" in entry:
        runtime.pod_template = _parse_pod_template(entry["pod_template"], file_path, launcher_mode)
    if "docker_kwargs" in entry:
        runtime.docker_kwargs = _parse_docker_kwargs(study, entry["docker_kwargs"], file_path, launcher_mode)
    if "datasets" in entry:
        datasets = _require_dict(entry["datasets"], f"studies.{study}.datasets", file_path)
        runtime.datasets = [
            _parse_dataset(study, dataset, ds_entry, file_path, launcher_mode) for dataset, ds_entry in datasets.items()
        ]
    if "env" in entry:
        runtime.env = _parse_env(study, entry["env"], file_path, launcher_mode)
    if "secret_env" in entry:
        runtime.secret_env = _parse_secret_env(study, entry["secret_env"], file_path, launcher_mode)
    duplicated = set(runtime.env) & {ref.name for ref in runtime.secret_env}
    if duplicated:
        raise _error(file_path, f"studies.{study} defines {sorted(duplicated)} in both env and secret_env.")
    if "secret_mounts" in entry:
        runtime.secret_mounts = _parse_secret_mounts(study, entry["secret_mounts"], file_path, launcher_mode)
    if "slurm" in entry:
        runtime.slurm = _parse_slurm(study, entry["slurm"], file_path)
        if runtime.slurm.get("sandbox") == "none":
            incompatible = [key for key in ("container", "datasets", "secret_mounts") if key in entry]
            if incompatible:
                raise _error(
                    file_path,
                    f"studies.{study}.slurm.sandbox 'none' does not support {sorted(incompatible)}.",
                )
    return runtime


def load_study_runtime_file(
    file_path: str,
    launcher_mode: str,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """Parse local/study_runtime.yaml (strict v2) for one launcher mode."""
    if launcher_mode not in _VALID_LAUNCHER_MODES:
        raise _error(file_path, f"unknown launcher mode {launcher_mode!r}.")
    try:
        with open(file_path, "rt") as f:
            config = yaml.safe_load(f)
    except OSError as e:
        raise ValueError(f"Could not read study runtime file '{file_path}': {e}") from e
    except yaml.YAMLError as e:
        raise ValueError(f"Could not parse study runtime file '{file_path}': {e}") from e

    config = _require_dict(config, "file content", file_path)
    _require_known_keys(config, {"format_version", "studies"}, "top level", file_path)
    format_version = config.get("format_version")
    if format_version != SUPPORTED_FORMAT_VERSION:
        raise _error(file_path, f"format_version must be {SUPPORTED_FORMAT_VERSION}, got {format_version!r}.")

    studies = config.get("studies")
    if studies is None:
        studies = {}
    studies = _require_dict(studies, "studies", file_path)
    if not studies and logger:
        logger.warning("study runtime file '%s' has no study entries; no study runtime will be configured", file_path)
    return {
        study: _parse_study(
            study,
            entry,
            file_path,
            launcher_mode=launcher_mode,
        )
        for study, entry in studies.items()
    }


def resolve_study_runtime(
    runtime_map: dict, study: Optional[str], file_path: str, logger: Optional[logging.Logger] = None
) -> StudyRuntime:
    """Return the study's runtime config, or an empty config when the study has no entry."""
    if study and study in runtime_map:
        return runtime_map[study]
    if study and runtime_map and logger:
        logger.warning(
            "study runtime file '%s' has no entry for study '%s'; no study runtime will be configured",
            file_path,
            study,
        )
    return StudyRuntime(study=study or "")
