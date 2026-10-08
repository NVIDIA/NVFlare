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

"""JSON bootstrap types shared by the job-based supervisor and task worker."""

import json
import math
import os
import stat
import uuid
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
_MAX_BOOTSTRAP_BYTES = 4 * 1024 * 1024
_MAX_IDENTITY_VALUE_LENGTH = 4096


def _freeze_json(value, depth=0):
    """Snapshot inert JSON values; reject executable objects and excessive nesting."""
    if depth > 64:
        raise ValueError("JSON nesting exceeds 64 levels")
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        return MappingProxyType({key: _freeze_json(item, depth + 1) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, depth + 1) for item in value)
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("worker records must contain only finite JSON-compatible values")


def _thaw_json(value):
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _load_json(encoded):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    value = json.loads(encoded, object_pairs_hook=pairs)
    return _thaw_json(_freeze_json(value))


def _require_text(name: str, value: str):
    if not isinstance(value, str) or not value or "\x00" in value or len(value) > _MAX_IDENTITY_VALUE_LENGTH:
        raise ValueError(f"{name} must be a nonempty string of at most {_MAX_IDENTITY_VALUE_LENGTH} characters")


def _require_absolute_directory_name(name: str, value: str):
    _require_text(name, value)
    if not os.path.isabs(value):
        raise ValueError(f"{name} must be an absolute path")


def _open_directory(path, create=False):
    """Anchor operations to real directories; never follow a replaced path segment."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open("/", flags)
    try:
        for part in os.path.abspath(path).split("/"):
            if part:
                if create:
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=fd)
                    except FileExistsError:
                        pass
                child = os.open(part, flags, dir_fd=fd)
                os.close(fd)
                fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _open_regular(path: str):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    directory_fd = _open_directory(os.path.dirname(os.path.abspath(path)))
    try:
        fd = os.open(os.path.basename(path), flags, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError(f"artifact must be a regular file: {path}")
    return os.fdopen(fd, "rb")


def _publish_exclusive(directory, name, writer):
    directory_fd = _open_directory(directory)
    temporary = f".{name}.{uuid.uuid4().hex}"
    try:
        fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600, dir_fd=directory_fd)
        try:
            with os.fdopen(fd, "w+b") as stream:
                writer(stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd, follow_symlinks=False)
            os.fsync(directory_fd)
        finally:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
    finally:
        os.close(directory_fd)


def _atomic_publish(path: str, encoded: bytes):
    directory = os.path.dirname(os.path.abspath(path))
    os.close(_open_directory(directory, create=True))
    _publish_exclusive(directory, os.path.basename(path), lambda stream: stream.write(encoded))


@dataclass(frozen=True)
class TaskAttemptIdentity:
    """Assignment plus authority-issued physical attempt; artifacts must match both.

    The client adapter receives attempt_id from the scheduling server, never
    invents it. Local unit-level supervisors can be driven by another trusted
    scheduling authority. A new physical attempt requires a new authority ID.
    """

    job_id: str
    site_name: str
    task_id: str
    task_name: str
    attempt_id: str

    def __post_init__(self):
        for name in ("job_id", "site_name", "task_id", "task_name", "attempt_id"):
            _require_text(name, getattr(self, name))
        for name in ("job_id", "site_name", "attempt_id"):
            value = getattr(self, name)
            if value in (".", "..") or "/" in value or "\\" in value:
                raise ValueError(f"{name} must be a single safe path component")

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "site_name": self.site_name,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "attempt_id": self.attempt_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        if not isinstance(value, Mapping):
            raise TypeError("attempt identity must be a mapping")
        expected = {"job_id", "site_name", "task_id", "task_name", "attempt_id"}
        if set(value) != expected:
            raise ValueError(f"attempt identity must contain exactly {sorted(expected)}")
        return cls(**{name: value[name] for name in expected})


@dataclass(frozen=True)
class ContextProperty:
    """One explicitly transported JSON-compatible FLContext property."""

    value: Any
    private: bool = True
    sticky: bool = False

    def __post_init__(self):
        object.__setattr__(self, "value", _freeze_json(self.value))
        if not isinstance(self.private, bool) or not isinstance(self.sticky, bool):
            raise TypeError("context property private and sticky flags must be bool")

    def to_dict(self) -> dict:
        return {"value": _thaw_json(self.value), "private": self.private, "sticky": self.sticky}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        if not isinstance(value, Mapping):
            raise TypeError("context property must be a mapping")
        if set(value) != {"value", "private", "sticky"}:
            raise ValueError("context property must contain exactly value, private, and sticky")
        return cls(value=value["value"], private=value["private"], sticky=value["sticky"])


@dataclass(frozen=True)
class WorkerBootstrap:
    """Trusted, inert configuration for one fresh, stateless task worker.

    ``executor`` and ``components`` retain normal NVFlare JSON component shapes.
    The worker builds only this selected compute graph; it does not reconstruct a
    live job engine or replay the supervising job's handler graph.
    """

    identity: TaskAttemptIdentity
    artifact_root: str
    workspace_root: str
    executor: Mapping[str, Any]
    components: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    context_properties: Mapping[str, ContextProperty] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if not isinstance(self.identity, TaskAttemptIdentity):
            raise TypeError("identity must be a TaskAttemptIdentity")
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported worker bootstrap schema version {self.schema_version}")
        _require_absolute_directory_name("artifact_root", self.artifact_root)
        _require_absolute_directory_name("workspace_root", self.workspace_root)
        if not isinstance(self.executor, Mapping):
            raise TypeError("executor must be a JSON component mapping")
        if not isinstance(self.components, Sequence) or isinstance(self.components, (str, bytes)):
            raise TypeError("components must be a sequence of JSON component mappings")
        component_ids = set()
        for component in self.components:
            if not isinstance(component, Mapping):
                raise TypeError("each component must be a JSON component mapping")
            component_id = component.get("id")
            if not isinstance(component_id, str) or not component_id or component_id in component_ids:
                raise ValueError("component IDs must be nonempty and unique")
            component_ids.add(component_id)
        if not isinstance(self.context_properties, Mapping):
            raise TypeError("context_properties must be a mapping")
        for name, prop in self.context_properties.items():
            _require_text("context property name", name)
            if not isinstance(prop, ContextProperty):
                raise TypeError(f"context property {name!r} must be a ContextProperty")
        object.__setattr__(self, "executor", _freeze_json(self.executor))
        object.__setattr__(self, "components", _freeze_json(self.components))
        object.__setattr__(self, "context_properties", MappingProxyType(dict(self.context_properties)))
        if len(json.dumps(self.to_dict(), allow_nan=False).encode("utf-8")) > _MAX_BOOTSTRAP_BYTES:
            raise ValueError("worker bootstrap is too large")

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "identity": self.identity.to_dict(),
            "artifact_root": self.artifact_root,
            "workspace_root": self.workspace_root,
            "executor": _thaw_json(self.executor),
            "components": _thaw_json(self.components),
            "context_properties": {name: prop.to_dict() for name, prop in self.context_properties.items()},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        if not isinstance(value, Mapping):
            raise TypeError("worker bootstrap must be a mapping")
        required = {"schema_version", "identity", "artifact_root", "workspace_root", "executor"}
        optional = {"components", "context_properties"}
        if not required.issubset(value) or set(value) - required - optional:
            raise ValueError("worker bootstrap has missing or unknown fields")
        context = value.get("context_properties", {})
        if not isinstance(context, Mapping):
            raise TypeError("context_properties must be a mapping")
        return cls(
            schema_version=value["schema_version"],
            identity=TaskAttemptIdentity.from_dict(value["identity"]),
            artifact_root=value["artifact_root"],
            workspace_root=value["workspace_root"],
            executor=value["executor"],
            components=tuple(value.get("components", ())),
            context_properties={name: ContextProperty.from_dict(prop) for name, prop in context.items()},
        )


def write_bootstrap(path: str, bootstrap: WorkerBootstrap):
    """Publish an immutable, mode-0600 worker bootstrap."""

    if not isinstance(bootstrap, WorkerBootstrap):
        raise TypeError("bootstrap must be a WorkerBootstrap")
    try:
        encoded = json.dumps(bootstrap.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    except (TypeError, ValueError) as e:
        raise ValueError(f"worker bootstrap must contain only JSON-compatible values: {e}") from e
    if len(encoded) > _MAX_BOOTSTRAP_BYTES:
        raise ValueError("worker bootstrap is too large")
    _atomic_publish(path, encoded)


def read_bootstrap(path: str) -> WorkerBootstrap:
    """Load and strictly validate a worker bootstrap."""

    with _open_regular(path) as stream:
        encoded = stream.read(_MAX_BOOTSTRAP_BYTES + 1)
    if len(encoded) > _MAX_BOOTSTRAP_BYTES:
        raise ValueError("worker bootstrap is too large")
    return WorkerBootstrap.from_dict(_load_json(encoded))
