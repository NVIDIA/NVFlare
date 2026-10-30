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

"""Immutable local artifacts for handoff between a supervisor and a task worker."""

import hashlib
import json
import logging
import math
import os
import re
import shutil
import stat
import time
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional

from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.decomposers.via_downloader import contains_lazy_download_ref
from nvflare.private.fed.resource_stats.contract import normalize_duration, normalize_measurement

from .protocol import SCHEMA_VERSION, TaskAttemptIdentity, _freeze_json, _load_json, _open_directory
from .protocol import _open_regular as _open_protocol_file
from .protocol import _publish_exclusive, _require_payload_limit, _thaw_json

_CHUNK_SIZE = 1024 * 1024
_MAX_RECORD_BYTES = 1024 * 1024
_MAX_RESOURCE_STATS_BYTES = 64 * 1024
_RESOURCE_STATS_TEMP = ".resource_stats.tmp"
_INPUT_KIND = "input"
_RESULT_KIND = "result"
_CHECKPOINT_KIND = "checkpoint"
_CHECKPOINT_INPUT_KIND = "checkpoint_input"
_LOGGER = logging.getLogger(__name__)


class IncompleteTaskArtifactError(RuntimeError):
    """The final commit record for an artifact does not exist."""


def _open_regular(path: str):
    try:
        return _open_protocol_file(path)
    except FileNotFoundError as e:
        raise IncompleteTaskArtifactError(f"artifact is incomplete: missing {os.path.basename(path)}") from e


class _PayloadWriter:
    """Hash the append-only FOBS stream while enforcing optional site policy."""

    def __init__(self, stream, max_payload_bytes):
        self.stream = stream
        self.max_payload_bytes = max_payload_bytes
        self.size = 0
        self.digest = hashlib.sha256()

    def write(self, value):
        if self.max_payload_bytes is not None and self.size + len(value) > self.max_payload_bytes:
            raise ValueError("artifact payload is too large")
        written = self.stream.write(value)
        self.digest.update(memoryview(value)[:written])
        self.size += written
        return written


class _TensorFileSink:
    """Externalize tensor leaves as owned files without retaining serialized tensors."""

    def __init__(self, directory, kind, max_payload_bytes):
        self.directory, self.kind = directory, kind
        self.max_payload_bytes = max_payload_bytes
        self.count = self.size = 0

    def add(self, value):
        from safetensors.torch import save

        from nvflare.app_opt.pt.lazy_tensor_dict import _LazyRef

        item_id = f"T{self.count}"
        file_name = f"{self.kind}.{item_id}.safetensors"
        remaining = None if self.max_payload_bytes is None else self.max_payload_bytes - self.size
        entry = None
        if isinstance(value, _LazyRef):
            with value.open() as tensors:
                if set(tensors.keys()) == {value.key}:
                    with _open_protocol_file(value.file_path) as stream:
                        size, digest = _fingerprint(stream, remaining)
                        destination = os.path.join(self.directory, file_name)
                        try:
                            _link_regular(value.file_path, destination, stream)
                        except OSError as e:
                            _LOGGER.warning("Cannot hard-link tensor %s; writing it instead: %s", file_name, e)
                        else:
                            try:
                                # Validate the pinned inode, not the mutable checkpoint slot.
                                with value.open(file_path=destination):
                                    pass
                            except BaseException:
                                os.unlink(destination)
                                raise
                            entry = {"file": file_name, "key": value.key, "size": size, "sha256": digest}
        if entry is None:
            tensor = value.materialize() if isinstance(value, _LazyRef) else value
            details = {}

            def write(stream):
                output = _PayloadWriter(stream, remaining)
                output.write(save({item_id: tensor}))
                details.update(size=output.size, sha256=output.digest.hexdigest())

            _publish_exclusive(self.directory, file_name, write)
            entry = {"file": file_name, "key": item_id, **details}
        self.count += 1
        self.size += entry["size"]
        return entry


class _TensorFileSource:
    def __init__(self, directory, reference, max_payload_bytes):
        self.directory, self.kind = directory, reference.kind
        self.max_payload_bytes = max_payload_bytes
        self.size = reference.size

    def get(self, entry):
        from safetensors import safe_open

        if (
            not isinstance(entry, dict)
            or set(entry) != {"file", "key", "size", "sha256"}
            or not isinstance(entry["file"], str)
            or not re.fullmatch(rf"{self.kind}\.T[0-9]+\.safetensors", entry["file"])
            or not isinstance(entry["key"], str)
            or type(entry["size"]) is not int
            or entry["size"] < 0
        ):
            raise ValueError("invalid tensor artifact descriptor")
        path = os.path.join(self.directory, entry["file"])
        remaining = None if self.max_payload_bytes is None else self.max_payload_bytes - self.size
        with _open_regular(path) as stream:
            if _fingerprint(stream, remaining) != (entry["size"], entry["sha256"]):
                raise ValueError("tensor artifact size or checksum mismatch")
        self.size += entry["size"]
        with safe_open(path, framework="pt") as tensors:
            return tensors.get_tensor(entry["key"])


def _fingerprint(stream, max_payload_bytes=None) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(_CHUNK_SIZE):
        digest.update(chunk)
        size += len(chunk)
        if max_payload_bytes is not None and size > max_payload_bytes:
            raise ValueError("artifact payload is too large")
    return size, digest.hexdigest()


def _write_json_exclusive(directory: str, name: str, value: Mapping[str, Any], *, publication_guard=nullcontext):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) > _MAX_RECORD_BYTES:
        raise ValueError(f"artifact record {name} is too large")
    _publish_exclusive(directory, name, lambda stream: stream.write(encoded), publication_guard=publication_guard)


def _read_json(path: str) -> dict:
    with _open_regular(path) as stream:
        encoded = stream.read(_MAX_RECORD_BYTES + 1)
    if len(encoded) > _MAX_RECORD_BYTES:
        raise ValueError(f"artifact record {os.path.basename(path)} is too large")
    value = _load_json(encoded)
    if not isinstance(value, dict):
        raise ValueError("artifact record must be a JSON object")
    return value


def _read_json_at(directory_fd, name, limit):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(name, flags, dir_fd=directory_fd)
    with os.fdopen(fd, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > limit:
            raise ValueError("resource record must be a bounded regular file")
        encoded = stream.read(limit + 1)
    if len(encoded) > limit:
        raise ValueError("resource record exceeds its size limit")
    value = _load_json(encoded)
    if not isinstance(value, dict):
        raise ValueError("resource record must be an object")
    return value


def _resource_record(identity, value):
    required = {"schema_version", "identity", "scope", "resource_capacity", "cpu_consumed", "message_traffic"}
    if not isinstance(value, Mapping) or not required <= set(value) <= required | {"duration_seconds"}:
        raise ValueError("invalid execution resource record fields")
    if value["schema_version"] != "1.0" or value["scope"] != "worker":
        raise ValueError("unsupported execution resource record")
    if TaskAttemptIdentity.from_dict(value["identity"]) != identity:
        raise ValueError("execution resource record identity mismatch")
    result = {"schema_version": "1.0", "identity": identity.to_dict(), "scope": "worker"}
    for name in ("resource_capacity", "cpu_consumed", "message_traffic"):
        result[name] = normalize_measurement(name, value[name])
    if "duration_seconds" in value:
        result["duration_seconds"] = normalize_duration(value["duration_seconds"])
    return result


@contextmanager
def _verified_checkpoint(path, max_payload_bytes):
    """Read a committed opaque checkpoint and the result it belongs to."""
    record = _read_json(path)
    required = {"schema_version", "identity", "checkpoint", "result"}
    if (
        not required.issubset(record)
        or set(record) - required - {"assignment"}
        or type(record["schema_version"]) is not int
        or record["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("invalid checkpoint record")
    identity = TaskAttemptIdentity.from_dict(record["identity"])
    if "assignment" in record:
        assignment = TaskAttemptIdentity.from_dict(record["assignment"])
        if replace(assignment, attempt_id=identity.attempt_id) != identity:
            raise ValueError("checkpoint assignment does not match its completion")
    reference = ArtifactReference.from_dict(record["checkpoint"])
    result = ArtifactReference.from_dict(record["result"])
    if reference.kind != _CHECKPOINT_KIND or result.kind != _RESULT_KIND:
        raise ValueError("invalid checkpoint/result reference")
    with _open_protocol_file(_checkpoint_payload_path(path, reference)) as stream:
        actual_size, digest = _fingerprint(stream, max_payload_bytes)
        if actual_size != reference.size or digest != reference.sha256:
            raise ValueError("checkpoint size or checksum mismatch")
        stream.seek(0)
        yield record, identity, reference, stream


def _checkpoint_payload_path(path, reference):
    return f"{path}.{reference.sha256}.bin"


def _link_regular(source, destination, stream):
    """Install a hard link to the verified inode without following path replacements."""
    source_fd = _open_directory(os.path.dirname(source))
    try:
        target_fd = _open_directory(os.path.dirname(destination), create=True)
        try:
            os.link(
                os.path.basename(source),
                os.path.basename(destination),
                src_dir_fd=source_fd,
                dst_dir_fd=target_fd,
                follow_symlinks=False,
            )
            try:
                actual = os.stat(os.path.basename(destination), dir_fd=target_fd, follow_symlinks=False)
                expected = os.fstat(stream.fileno())
                if not stat.S_ISREG(actual.st_mode) or (actual.st_dev, actual.st_ino) != (
                    expected.st_dev,
                    expected.st_ino,
                ):
                    raise ValueError("checkpoint payload changed during publication")
                os.fsync(target_fd)
            except BaseException:
                os.unlink(os.path.basename(destination), dir_fd=target_fd)
                raise
        finally:
            os.close(target_fd)
    finally:
        os.close(source_fd)


def _remove_checkpoint_payload(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError:
        try:
            _LOGGER.warning("Could not remove superseded checkpoint payload %s", path, exc_info=True)
        except Exception:
            pass


def _replace_checkpoint(path, writer, publication_guard):
    directory = os.path.dirname(path)
    directory_fd = _open_directory(directory, create=True)
    temporary = f".checkpoint.{uuid.uuid4().hex}"
    published = False
    try:
        try:
            _publish_exclusive(directory, temporary, writer)
            with publication_guard():
                os.replace(temporary, os.path.basename(path), src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
                # Replacement is the local commit point. A later sync/cleanup
                # error cannot turn its matching successful result into failure.
                published = True
                os.fsync(directory_fd)
        finally:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            finally:
                os.close(directory_fd)
    except OSError:
        if not published:
            raise
        try:
            _LOGGER.warning(
                "I/O error after published checkpoint %s; retaining it and its matching result. "
                "Crash durability could not be confirmed",
                path,
                exc_info=True,
            )
        except Exception:
            # A diagnostic handler must not split the committed state/result.
            pass


def require_eager_shareable(data: Shareable):
    """Reject data whose contents depend on a live federation download owner."""

    if not isinstance(data, Shareable):
        raise TypeError("task artifact data must be a Shareable")
    if data.get_header(ReservedHeaderKey.PASS_THROUGH):
        raise ValueError("durable task handoff requires eager data; pass-through Shareable is unsupported")
    if contains_lazy_download_ref(data):
        raise ValueError("durable task handoff requires eager data; lazy download references are unsupported")


@dataclass(frozen=True)
class ArtifactReference:
    kind: str
    file_name: str
    size: int
    sha256: str

    def __post_init__(self):
        if self.kind not in (_INPUT_KIND, _RESULT_KIND, _CHECKPOINT_KIND, _CHECKPOINT_INPUT_KIND):
            raise ValueError("invalid artifact kind")
        suffix = "bin" if self.kind in (_CHECKPOINT_KIND, _CHECKPOINT_INPUT_KIND) else "fobs"
        if self.file_name != f"{self.kind}.{suffix}":
            raise ValueError("artifact payload name does not match its kind")
        if type(self.size) is not int or self.size < 0:
            raise ValueError("artifact size must be a nonnegative integer")
        if not isinstance(self.sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("artifact sha256 must be a hexadecimal SHA-256 digest")

    def to_dict(self) -> dict:
        return {"kind": self.kind, "file_name": self.file_name, "size": self.size, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        if not isinstance(value, Mapping) or set(value) != {"kind", "file_name", "size", "sha256"}:
            raise ValueError("invalid artifact reference")
        return cls(**dict(value))


@dataclass(frozen=True)
class TaskCompletion:
    """Local compute finalization; this is independent of server result receipt."""

    identity: TaskAttemptIdentity
    result: ArtifactReference
    worker_pid: int
    worker_ppid: int
    started_at: float
    completed_at: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported task completion schema version {self.schema_version}")
        if not isinstance(self.identity, TaskAttemptIdentity):
            raise TypeError("completion identity must be a TaskAttemptIdentity")
        if not isinstance(self.result, ArtifactReference) or self.result.kind != _RESULT_KIND:
            raise TypeError("completion result must be a result ArtifactReference")
        if type(self.worker_pid) is not int or self.worker_pid <= 0:
            raise ValueError("worker_pid must be positive")
        if type(self.worker_ppid) is not int or self.worker_ppid < 0:
            raise ValueError("worker_ppid must be nonnegative")
        if not isinstance(self.started_at, (int, float)) or not isinstance(self.completed_at, (int, float)):
            raise TypeError("completion timestamps must be numeric")
        if any(type(t) is bool or not math.isfinite(t) for t in (self.started_at, self.completed_at)):
            raise ValueError("completion timestamps must be finite numbers")
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError("completion diagnostics must be a mapping")
        object.__setattr__(self, "diagnostics", _freeze_json(self.diagnostics))

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "identity": self.identity.to_dict(),
            "result": self.result.to_dict(),
            "worker_pid": self.worker_pid,
            "worker_ppid": self.worker_ppid,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "diagnostics": _thaw_json(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        expected = {
            "schema_version",
            "identity",
            "result",
            "worker_pid",
            "worker_ppid",
            "started_at",
            "completed_at",
            "diagnostics",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("invalid task completion record")
        return cls(
            schema_version=value["schema_version"],
            identity=TaskAttemptIdentity.from_dict(value["identity"]),
            result=ArtifactReference.from_dict(value["result"]),
            worker_pid=value["worker_pid"],
            worker_ppid=value["worker_ppid"],
            started_at=value["started_at"],
            completed_at=value["completed_at"],
            diagnostics=value["diagnostics"],
        )


class FileTaskArtifactStore:
    """Process-backend artifact store whose files outlive the worker process.

    The store never removes an attempt automatically. Under site policy, the
    CJ supervisor may call :meth:`release_payloads` while retaining the
    completion record, or :meth:`remove_attempt` after a full retention decision.
    """

    def __init__(self, root_dir: str, *, max_payload_bytes: int | None = None):
        if not isinstance(root_dir, str) or not os.path.isabs(root_dir):
            raise ValueError("artifact root must be an absolute path")
        _require_payload_limit(max_payload_bytes)
        self.root_dir = os.path.realpath(root_dir)
        self.max_payload_bytes = max_payload_bytes

    def attempt_dir(self, identity: TaskAttemptIdentity) -> str:
        if not isinstance(identity, TaskAttemptIdentity):
            raise TypeError("identity must be a TaskAttemptIdentity")
        candidate = os.path.join(self.root_dir, identity.attempt_id)
        if os.path.islink(candidate):
            raise ValueError("attempt directory is missing or invalid: symlink escapes identity")
        path = os.path.realpath(candidate)
        if os.path.commonpath([self.root_dir, path]) != self.root_dir:
            raise ValueError("attempt directory escapes artifact root")
        return path

    def bootstrap_path(self, identity: TaskAttemptIdentity) -> str:
        return os.path.join(self.attempt_dir(identity), "bootstrap.json")

    def create_attempt(self, identity: TaskAttemptIdentity) -> str:
        path = self.attempt_dir(identity)
        directory_fd = _open_directory(self.root_dir, create=True)
        try:
            os.mkdir(identity.attempt_id, mode=0o700, dir_fd=directory_fd)
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _write_json_exclusive(path, "identity.json", identity.to_dict())
        return path

    def _check_identity(self, identity):
        actual = TaskAttemptIdentity.from_dict(_read_json(os.path.join(self.attempt_dir(identity), "identity.json")))
        if actual != identity:
            raise ValueError("invalid or stale task artifact identity")

    @contextmanager
    def _resource_directory(self, identity):
        directory = self.attempt_dir(identity)
        directory_fd = _open_directory(directory)
        try:
            before = os.fstat(directory_fd)
            actual = TaskAttemptIdentity.from_dict(_read_json_at(directory_fd, "identity.json", _MAX_RECORD_BYTES))
            if actual != identity:
                raise ValueError("invalid or stale task artifact identity")
            yield directory_fd
            after = os.stat(directory, follow_symlinks=False)
            if not stat.S_ISDIR(after.st_mode) or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                raise ValueError("task resource directory was replaced")
        finally:
            os.close(directory_fd)

    def write_resource_stats(self, identity, value):
        """Publish one optional final sidecar independently of result commits."""

        checked = _resource_record(identity, value)
        encoded = json.dumps(checked, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(encoded) > _MAX_RESOURCE_STATS_BYTES:
            raise ValueError("execution resource record exceeds 64 KiB")
        with self._resource_directory(identity) as directory_fd:
            # A claimed attempt has one writer. A fixed exclusive temporary can
            # be retired after hard loss without scanning the attempt directory.
            temporary = _RESOURCE_STATS_TEMP
            fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600, dir_fd=directory_fd)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(
                    temporary,
                    "resource_stats.json",
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                os.fsync(directory_fd)
            finally:
                os.unlink(temporary, dir_fd=directory_fd)

    def read_resource_stats(self, identity):
        """Read only the owned sidecar; missing statistics are not result errors."""

        with self._resource_directory(identity) as directory_fd:
            try:
                value = _read_json_at(directory_fd, "resource_stats.json", _MAX_RESOURCE_STATS_BYTES)
            except FileNotFoundError:
                return None
            return _resource_record(identity, value)

    def remove_resource_stats(self, identity):
        """Retire settled statistics and interrupted publication, not result files."""

        with self._resource_directory(identity) as directory_fd:
            for name in ("resource_stats.json", _RESOURCE_STATS_TEMP):
                try:
                    os.unlink(name, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass

    def claim_worker(self, identity: TaskAttemptIdentity):
        """Fence duplicate execution before any application import or callback.

        An interrupted attempt is never reusable; a fresh physical execution
        requires a new authority-issued attempt identity.
        """
        self._check_identity(identity)
        directory = self.attempt_dir(identity)
        if any(name.startswith(("result.T", ".result.T")) for name in os.listdir(directory)):
            raise FileExistsError("attempt already contains worker artifacts")
        for name in ("result.fobs", "completion.json", "failure.json"):
            if os.path.lexists(os.path.join(directory, name)):
                raise FileExistsError("attempt already contains worker artifacts")
        _write_json_exclusive(directory, "worker.json", {"identity": identity.to_dict(), "worker_pid": os.getpid()})

    def _write_payload(self, identity: TaskAttemptIdentity, kind: str, data: Shareable) -> ArtifactReference:
        require_eager_shareable(data)
        sink = _TensorFileSink(self.attempt_dir(identity), kind, self.max_payload_bytes)
        reference = self._write_payload_stream(
            identity,
            kind,
            lambda stream: fobs.dump_to_stream(
                data,
                stream,
                max_value_size=_CHUNK_SIZE,
                fobs_ctx={"native": True, fobs.FOBSContextKey.TENSOR_FILE_SINK: sink},
            ),
        )
        if self.max_payload_bytes is not None and reference.size + sink.size > self.max_payload_bytes:
            raise ValueError("artifact payload is too large")
        return reference

    def _write_payload_stream(self, identity, kind, write):
        directory = self.attempt_dir(identity)
        if not os.path.isdir(directory) or os.path.islink(directory):
            raise ValueError("attempt directory is missing or invalid")
        self._check_identity(identity)
        suffix = "bin" if kind in (_CHECKPOINT_KIND, _CHECKPOINT_INPUT_KIND) else "fobs"
        file_name = f"{kind}.{suffix}"
        details = {}

        def writer(stream):
            payload_writer = _PayloadWriter(stream, self.max_payload_bytes)
            write(payload_writer)
            details["size"] = payload_writer.size
            details["sha256"] = payload_writer.digest.hexdigest()

        _publish_exclusive(directory, file_name, writer)
        return ArtifactReference(kind=kind, file_name=file_name, **details)

    @contextmanager
    def _verified_payload(self, identity, reference):
        self._check_identity(identity)
        if self.max_payload_bytes is not None and reference.size > self.max_payload_bytes:
            raise ValueError("artifact payload is too large")
        path = os.path.join(self.attempt_dir(identity), reference.file_name)
        with _open_regular(path) as stream:
            if os.fstat(stream.fileno()).st_size != reference.size:
                raise ValueError("artifact payload size or checksum mismatch")
            size, digest = _fingerprint(stream, self.max_payload_bytes)
            if size != reference.size or digest != reference.sha256:
                raise ValueError("artifact payload size or checksum mismatch")
            stream.seek(0)
            yield stream

    def _read_payload(self, identity: TaskAttemptIdentity, reference: ArtifactReference) -> Shareable:
        fobs_ctx = {
            "native": True,
            fobs.FOBSContextKey.TENSOR_FILE_SOURCE: _TensorFileSource(
                self.attempt_dir(identity), reference, self.max_payload_bytes
            ),
        }
        with self._verified_payload(identity, reference) as stream:
            data = fobs.load_from_stream(stream, fobs_ctx=fobs_ctx)
        require_eager_shareable(data)
        return data

    @staticmethod
    def _identity_from_record(record: Mapping[str, Any], expected: TaskAttemptIdentity):
        actual = TaskAttemptIdentity.from_dict(record.get("identity"))
        if actual != expected:
            raise ValueError("invalid or stale task artifact identity")

    def write_input(self, identity: TaskAttemptIdentity, data: Shareable) -> ArtifactReference:
        reference = self._write_payload(identity, _INPUT_KIND, data)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "identity": identity.to_dict(),
            "artifact": reference.to_dict(),
        }
        _write_json_exclusive(self.attempt_dir(identity), "input.json", manifest)
        return reference

    def read_input(self, identity: TaskAttemptIdentity) -> Shareable:
        manifest = _read_json(os.path.join(self.attempt_dir(identity), "input.json"))
        if set(manifest) != {"schema_version", "identity", "artifact"}:
            raise ValueError("invalid input artifact manifest")
        if type(manifest["schema_version"]) is not int or manifest["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported input artifact schema version")
        self._identity_from_record(manifest, identity)
        reference = ArtifactReference.from_dict(manifest["artifact"])
        if reference.kind != _INPUT_KIND:
            raise ValueError("input manifest refers to the wrong artifact kind")
        return self._read_payload(identity, reference)

    def commit_result(
        self,
        identity: TaskAttemptIdentity,
        data: Shareable,
        *,
        worker_pid: Optional[int] = None,
        worker_ppid: Optional[int] = None,
        started_at: Optional[float] = None,
        completed_at: Optional[float] = None,
        diagnostics: Optional[Mapping[str, Any]] = None,
    ) -> TaskCompletion:
        """Write the immutable result payload and install completion last."""

        reference = self.stage_result(identity, data)
        return self.commit_staged_result(
            identity,
            reference,
            worker_pid=worker_pid,
            worker_ppid=worker_ppid,
            started_at=started_at,
            completed_at=completed_at,
            diagnostics=diagnostics,
        )

    def stage_result(self, identity: TaskAttemptIdentity, data: Shareable) -> ArtifactReference:
        """Durably hand off a result without declaring successful worker completion."""
        return self._write_payload(identity, _RESULT_KIND, data)

    def commit_staged_result(
        self,
        identity: TaskAttemptIdentity,
        reference: ArtifactReference,
        *,
        worker_pid: Optional[int] = None,
        worker_ppid: Optional[int] = None,
        started_at: Optional[float] = None,
        completed_at: Optional[float] = None,
        diagnostics: Optional[Mapping[str, Any]] = None,
        publication_guard=nullcontext,
    ) -> TaskCompletion:
        """Verify the result record, then guard the final atomic completion publication."""
        if not isinstance(reference, ArtifactReference) or reference.kind != _RESULT_KIND:
            raise ValueError("staged result must be a result artifact reference")
        with self._verified_payload(identity, reference):
            pass
        now = time.time()
        completion = TaskCompletion(
            identity=identity,
            result=reference,
            worker_pid=os.getpid() if worker_pid is None else worker_pid,
            worker_ppid=os.getppid() if worker_ppid is None else worker_ppid,
            started_at=now if started_at is None else started_at,
            completed_at=now if completed_at is None else completed_at,
            diagnostics={} if diagnostics is None else diagnostics,
        )
        try:
            _write_json_exclusive(
                self.attempt_dir(identity),
                "completion.json",
                completion.to_dict(),
                publication_guard=publication_guard,
            )
        except (TypeError, ValueError) as e:
            raise ValueError(f"completion diagnostics must contain only JSON-compatible values: {e}") from e
        return completion

    def checkpoint_paths(self, identity):
        """Expose only this attempt's input copy and pending output file."""
        self._check_identity(identity)
        directory = self.attempt_dir(identity)
        manifest_path = os.path.join(directory, "checkpoint_input.json")
        input_path = None
        if os.path.lexists(manifest_path):
            manifest = _read_json(manifest_path)
            if set(manifest) != {"identity", "artifact"}:
                raise ValueError("invalid checkpoint input manifest")
            self._identity_from_record(manifest, identity)
            reference = ArtifactReference.from_dict(manifest["artifact"])
            if reference.kind != _CHECKPOINT_INPUT_KIND:
                raise ValueError("invalid checkpoint input reference")
            with self._verified_payload(identity, reference):
                input_path = os.path.join(directory, reference.file_name)
        return {"input": input_path, "output": os.path.join(directory, "checkpoint.pending")}

    def prepare_checkpoint(self, identity, checkpoint_file):
        """CJ snapshots the previously committed state before launching a worker."""
        if not os.path.lexists(checkpoint_file):
            return
        with _verified_checkpoint(checkpoint_file, self.max_payload_bytes) as (_, previous, _, stream):
            if (previous.job_id, previous.site_name, previous.task_name) != (
                identity.job_id,
                identity.site_name,
                identity.task_name,
            ):
                raise ValueError("checkpoint belongs to a different job, site or task")
            reference = self._write_payload_stream(
                identity, _CHECKPOINT_INPUT_KIND, lambda output: shutil.copyfileobj(stream, output, _CHUNK_SIZE)
            )
        _write_json_exclusive(
            self.attempt_dir(identity),
            "checkpoint_input.json",
            {"identity": identity.to_dict(), "artifact": reference.to_dict()},
        )

    def stage_checkpoint(self, identity):
        """Seal an optional application-written file after executor finalization."""
        self._check_identity(identity)
        path = os.path.join(self.attempt_dir(identity), "checkpoint.pending")
        if not os.path.lexists(path):
            return None
        with _open_protocol_file(path) as stream:
            size, digest = _fingerprint(stream, self.max_payload_bytes)
            os.fsync(stream.fileno())
            reference = ArtifactReference(_CHECKPOINT_KIND, "checkpoint.bin", size, digest)
            _link_regular(path, os.path.join(self.attempt_dir(identity), reference.file_name), stream)
        os.unlink(path)
        return reference

    def commit_checkpoint(self, completion, checkpoint_file, *, publication_guard=nullcontext, assignment=None):
        """CJ promotes state only from the verified, physically settled result."""
        config = completion.diagnostics.get("checkpoint")
        if config is None:
            return
        reference = ArtifactReference.from_dict(config)
        if reference.kind != _CHECKPOINT_KIND:
            raise ValueError("completion has an invalid checkpoint reference")
        # Idempotence covers re-observing completion; transport retries keep
        # using the saved result and never execute the script again.
        previous_payload = None
        if os.path.lexists(checkpoint_file):
            with _verified_checkpoint(checkpoint_file, self.max_payload_bytes) as (record, previous, previous_ref, _):
                previous_payload = _checkpoint_payload_path(checkpoint_file, previous_ref)
                if previous == completion.identity:
                    if record["checkpoint"] != reference.to_dict() or record["result"] != completion.result.to_dict():
                        raise ValueError("checkpoint completion was changed")
                    if assignment is not None and record.get("assignment") != assignment.to_dict():
                        raise ValueError("checkpoint assignment was changed")
                    return
        record = {
            "schema_version": SCHEMA_VERSION,
            "identity": completion.identity.to_dict(),
            "checkpoint": reference.to_dict(),
            "result": completion.result.to_dict(),
        }
        if assignment is not None:
            if replace(assignment, attempt_id=completion.identity.attempt_id) != completion.identity:
                raise ValueError("checkpoint assignment does not match its completion")
            record["assignment"] = assignment.to_dict()
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(encoded) > _MAX_RECORD_BYTES:
            raise ValueError("checkpoint record is too large")

        payload_path = _checkpoint_payload_path(checkpoint_file, reference)
        created = False
        try:
            with self._verified_payload(completion.identity, reference) as source:
                try:
                    _link_regular(
                        os.path.join(self.attempt_dir(completion.identity), reference.file_name), payload_path, source
                    )
                    created = True
                except FileExistsError:
                    with _open_protocol_file(payload_path) as existing:
                        if _fingerprint(existing, self.max_payload_bytes) != (reference.size, reference.sha256):
                            raise ValueError("existing checkpoint payload size or checksum mismatch")
            # Only the small manifest is replaced. Its immutable payload is
            # already durable, so this remains a single atomic commit point.
            _replace_checkpoint(checkpoint_file, lambda output: output.write(encoded), publication_guard)
        except BaseException:
            if created:
                _remove_checkpoint_payload(payload_path)
            raise
        if previous_payload is not None and previous_payload != payload_path:
            _remove_checkpoint_payload(previous_payload)

    def committed_checkpoint_completion(self, assignment, checkpoint_file):
        """Find the saved completion that already advanced this server assignment."""
        if not os.path.lexists(checkpoint_file):
            return None
        record = _read_json(checkpoint_file)
        if record.get("assignment") != assignment.to_dict():
            return None
        with _verified_checkpoint(checkpoint_file, self.max_payload_bytes) as (record, identity, _, _):
            if record.get("assignment") != assignment.to_dict():
                raise ValueError("checkpoint assignment changed during recovery")
            completion = self.read_completion(identity)
            if (
                completion.result.to_dict() != record["result"]
                or completion.diagnostics.get("checkpoint") != record["checkpoint"]
            ):
                raise ValueError("checkpoint does not match its saved completion")
            return completion

    def read_completion(self, identity: TaskAttemptIdentity) -> TaskCompletion:
        self._check_identity(identity)
        completion = TaskCompletion.from_dict(_read_json(os.path.join(self.attempt_dir(identity), "completion.json")))
        if completion.identity != identity:
            raise ValueError("invalid or stale task completion identity")
        return completion

    def read_result(self, identity: TaskAttemptIdentity) -> tuple[Shareable, TaskCompletion]:
        completion = self.read_completion(identity)
        return self._read_payload(identity, completion.result), completion

    def read_failure(self, identity: TaskAttemptIdentity):
        """Read a settled worker's diagnostic without treating it as completion."""
        self._check_identity(identity)
        try:
            with _open_protocol_file(os.path.join(self.attempt_dir(identity), "failure.json")) as stream:
                encoded = stream.read(_MAX_RECORD_BYTES + 1)
        except FileNotFoundError:
            return None
        if len(encoded) > _MAX_RECORD_BYTES:
            raise ValueError("task failure record is too large")
        record = _load_json(encoded)
        if record.get("schema_version") != SCHEMA_VERSION or record.get("identity") != identity.to_dict():
            raise ValueError("task failure record has an invalid identity or schema")
        if not all(isinstance(record.get(key), str) for key in ("error_type", "message")):
            raise ValueError("task failure record has an invalid error")
        return record

    def record_failure(
        self,
        identity: TaskAttemptIdentity,
        *,
        worker_pid: int,
        worker_ppid: int,
        started_at: float,
        failed_at: float,
        error_type: str,
        message: str,
    ):
        """Retain diagnostics without making a failed attempt look committed."""

        self._check_identity(identity)
        record = {
            "schema_version": SCHEMA_VERSION,
            "identity": identity.to_dict(),
            "worker_pid": worker_pid,
            "worker_ppid": worker_ppid,
            "started_at": started_at,
            "failed_at": failed_at,
            "error_type": error_type,
            "message": message,
        }
        _write_json_exclusive(self.attempt_dir(identity), "failure.json", record)

    def release_payloads(self, identity: TaskAttemptIdentity):
        """Release payloads and interrupted writes after writer/reader quiescence."""

        directory = self.attempt_dir(identity)
        if (
            os.path.islink(os.path.join(self.root_dir, identity.attempt_id))
            or os.path.islink(directory)
            or not os.path.isdir(directory)
        ):
            raise ValueError("attempt directory is missing or invalid")
        self._check_identity(identity)
        directory_fd = _open_directory(directory)
        try:
            payload_names = (
                "bootstrap.json",
                "input.json",
                "input.fobs",
                "result.fobs",
                "checkpoint_input.json",
                "checkpoint_input.bin",
                "checkpoint.pending",
                "checkpoint.bin",
            )
            temporary_prefixes = tuple(f".{name}." for name in payload_names) + (
                "input.T",
                ".input.T",
                "result.T",
                ".result.T",
            )
            names = set(payload_names)
            names.update(name for name in os.listdir(directory_fd) if name.startswith(temporary_prefixes))
            for name in names:
                try:
                    info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    if not stat.S_ISREG(info.st_mode):
                        raise ValueError(f"refusing to release invalid attempt artifact {name}")
                    os.unlink(name, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def remove_attempt(self, identity: TaskAttemptIdentity):
        """Remove all payloads and diagnostics after a full retention decision."""

        path = self.attempt_dir(identity)
        if os.path.islink(path):
            raise ValueError("refusing to remove a symlinked attempt directory")
        try:
            self._check_identity(identity)
        except IncompleteTaskArtifactError:
            # create_attempt may stop between mkdir and identity publication.
            # Full retention cleanup can remove this incomplete directory;
            # existing identities must still be valid and match the caller.
            pass
        shutil.rmtree(path)
