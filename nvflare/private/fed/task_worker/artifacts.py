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
import math
import os
import re
import shutil
import stat
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.decomposers.via_downloader import contains_lazy_download_ref

from .protocol import SCHEMA_VERSION, TaskAttemptIdentity, _freeze_json, _load_json, _open_directory
from .protocol import _open_regular as _open_protocol_file
from .protocol import _publish_exclusive, _thaw_json

_CHUNK_SIZE = 1024 * 1024
_MAX_RECORD_BYTES = 1024 * 1024
_MAX_PAYLOAD_BYTES = 4 * 1024 * 1024 * 1024
_INPUT_KIND = "input"
_RESULT_KIND = "result"
_ANALYTICS_KIND = "analytics"


class IncompleteTaskArtifactError(RuntimeError):
    """The final commit record for an artifact does not exist."""


def _open_regular(path: str):
    try:
        return _open_protocol_file(path)
    except FileNotFoundError as e:
        raise IncompleteTaskArtifactError(f"artifact is incomplete: missing {os.path.basename(path)}") from e


class _BoundedWriter:
    def __init__(self, stream):
        self.stream = stream

    def write(self, value):
        if self.stream.tell() + len(value) > _MAX_PAYLOAD_BYTES:
            raise ValueError("artifact payload is too large")
        return self.stream.write(value)

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _fingerprint(stream) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(_CHUNK_SIZE):
        digest.update(chunk)
        size += len(chunk)
        if size > _MAX_PAYLOAD_BYTES:
            raise ValueError("artifact payload is too large")
    return size, digest.hexdigest()


def _write_json_exclusive(directory: str, name: str, value: Mapping[str, Any]):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) > _MAX_RECORD_BYTES:
        raise ValueError(f"artifact record {name} is too large")
    _publish_exclusive(directory, name, lambda stream: stream.write(encoded))


def _read_json(path: str) -> dict:
    with _open_regular(path) as stream:
        encoded = stream.read(_MAX_RECORD_BYTES + 1)
    if len(encoded) > _MAX_RECORD_BYTES:
        raise ValueError(f"artifact record {os.path.basename(path)} is too large")
    value = _load_json(encoded)
    if not isinstance(value, dict):
        raise ValueError("artifact record must be a JSON object")
    return value


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
        if self.kind not in (_INPUT_KIND, _RESULT_KIND, _ANALYTICS_KIND):
            raise ValueError("artifact kind must be input, result or analytics")
        if self.file_name != f"{self.kind}.fobs":
            raise ValueError("artifact payload name does not match its kind")
        if type(self.size) is not int or not 0 <= self.size <= _MAX_PAYLOAD_BYTES:
            raise ValueError("artifact size must be a bounded nonnegative integer")
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
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must not precede started_at")
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

    def __init__(self, root_dir: str):
        if not isinstance(root_dir, str) or not os.path.isabs(root_dir):
            raise ValueError("artifact root must be an absolute path")
        self.root_dir = os.path.realpath(root_dir)

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

    def claim_worker(self, identity: TaskAttemptIdentity):
        """Fence duplicate execution before any application import or callback.

        An interrupted attempt is never reusable; a fresh physical execution
        requires a new authority-issued attempt identity.
        """
        self._check_identity(identity)
        directory = self.attempt_dir(identity)
        for name in ("result.fobs", "analytics.fobs", "completion.json", "failure.json"):
            if os.path.lexists(os.path.join(directory, name)):
                raise FileExistsError("attempt already contains worker artifacts")
        _write_json_exclusive(directory, "worker.json", {"identity": identity.to_dict(), "worker_pid": os.getpid()})

    def _write_payload(self, identity: TaskAttemptIdentity, kind: str, data: Shareable) -> ArtifactReference:
        require_eager_shareable(data)
        directory = self.attempt_dir(identity)
        if not os.path.isdir(directory) or os.path.islink(directory):
            raise ValueError("attempt directory is missing or invalid")
        self._check_identity(identity)
        file_name = f"{kind}.fobs"
        details = {}

        def writer(stream):
            fobs.dump_to_stream(data, _BoundedWriter(stream), max_value_size=_CHUNK_SIZE, fobs_ctx={"native": True})
            stream.flush()
            stream.seek(0)
            details["size"], details["sha256"] = _fingerprint(stream)

        _publish_exclusive(directory, file_name, writer)
        return ArtifactReference(kind=kind, file_name=file_name, **details)

    @contextmanager
    def _verified_payload(self, identity, reference):
        self._check_identity(identity)
        path = os.path.join(self.attempt_dir(identity), reference.file_name)
        with _open_regular(path) as stream:
            if os.fstat(stream.fileno()).st_size != reference.size:
                raise ValueError("artifact payload size or checksum mismatch")
            size, digest = _fingerprint(stream)
            if size != reference.size or digest != reference.sha256:
                raise ValueError("artifact payload size or checksum mismatch")
            stream.seek(0)
            yield stream

    def _read_payload(self, identity: TaskAttemptIdentity, reference: ArtifactReference) -> Shareable:
        with self._verified_payload(identity, reference) as stream:
            data = fobs.load_from_stream(stream, fobs_ctx={"native": True})
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
    ) -> TaskCompletion:
        """Install completion only after application code and finalization finish."""
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
            _write_json_exclusive(self.attempt_dir(identity), "completion.json", completion.to_dict())
        except (TypeError, ValueError) as e:
            raise ValueError(f"completion diagnostics must contain only JSON-compatible values: {e}") from e
        return completion

    def write_analytics(self, identity: TaskAttemptIdentity, records: list) -> ArtifactReference:
        return self._write_payload(identity, _ANALYTICS_KIND, Shareable({"records": records}))

    def read_analytics(self, identity: TaskAttemptIdentity, completion: TaskCompletion) -> list:
        if completion.identity != identity:
            raise ValueError("invalid or stale task completion identity")
        config = completion.diagnostics.get("analytics")
        if config is None:
            return []
        reference = ArtifactReference.from_dict(config)
        if reference.kind != _ANALYTICS_KIND:
            raise ValueError("analytics reference has the wrong artifact kind")
        records = self._read_payload(identity, reference).get("records")
        if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
            raise ValueError("analytics artifact must contain a list of records")
        return records

    def read_completion(self, identity: TaskAttemptIdentity) -> TaskCompletion:
        self._check_identity(identity)
        completion = TaskCompletion.from_dict(_read_json(os.path.join(self.attempt_dir(identity), "completion.json")))
        if completion.identity != identity:
            raise ValueError("invalid or stale task completion identity")
        return completion

    def read_result(self, identity: TaskAttemptIdentity) -> tuple[Shareable, TaskCompletion]:
        completion = self.read_completion(identity)
        return self._read_payload(identity, completion.result), completion

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
        """Release bulky attempt data while retaining completion diagnostics."""

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
            for name in ("bootstrap.json", "input.json", "input.fobs", "result.fobs", "analytics.fobs"):
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
        self._check_identity(identity)
        shutil.rmtree(path)
