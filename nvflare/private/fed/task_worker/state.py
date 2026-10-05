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

"""Local declared-state promotion, separate from disposable attempt retention."""

import fcntl
import json
import os
import re
import stat
import tempfile
from contextlib import contextmanager

from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.task_state import MAX_TASK_STATE_BYTES, TaskState

from .protocol import TaskAttemptIdentity


class StaleTaskStateError(RuntimeError):
    """An admitted attempt tried to replace a newer committed state revision."""


class FileTaskStateStore:
    """Promote explicit worker state with a successful, admitted result.

    Completion atomically names both immutable result and candidate state.
    Promotion verifies that pair and the input revision under a local file lock.
    A duplicate acknowledgement is idempotent; stale or conflicting attempts
    cannot overwrite state. This is Process/local-file support, not distributed
    consensus or recovery of server-side aggregation transactions.
    """

    def __init__(self, root_dir, names):
        if not isinstance(root_dir, str) or not os.path.isabs(root_dir):
            raise ValueError("state root must be an absolute path")
        self.root_dir = os.path.realpath(root_dir)
        self.names = TaskState.validate_names(names)
        self.path = os.path.join(self.root_dir, "current.json")

    @contextmanager
    def _locked(self):
        os.makedirs(self.root_dir, mode=0o700, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        fd = os.open(os.path.join(self.root_dir, "promotion.lock"), flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("state lock must be a regular file")
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def _read_current(self):
        try:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
            fd = os.open(self.path, flags)
        except FileNotFoundError:
            return None
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("state checkpoint must be a regular file")
            encoded = stream.read(MAX_TASK_STATE_BYTES + 65537)
        if len(encoded) > MAX_TASK_STATE_BYTES + 65536:
            raise ValueError("state checkpoint is too large")
        current = json.loads(encoded)
        expected = {"revision", "identity", "names", "records", "result_sha256", "state_sha256"}
        if not isinstance(current, dict) or set(current) != expected:
            raise ValueError("invalid state checkpoint")
        if type(current["revision"]) is not int or current["revision"] <= 0:
            raise ValueError("invalid state checkpoint revision")
        if TaskState.validate_names(current["names"]) != self.names:
            raise ValueError("state declarations changed within a job")
        TaskAttemptIdentity.from_dict(current["identity"])
        TaskState.from_wire(self.names, current["records"])
        for key in ("result_sha256", "state_sha256"):
            if not isinstance(current[key], str) or not re.fullmatch(r"[0-9a-f]{64}", current[key]):
                raise ValueError("invalid state checkpoint digest")
        return current

    def snapshot(self):
        with self._locked():
            current = self._read_current()
            return (0, {}) if current is None else (current["revision"], current["records"])

    def commit(self, identity, completion, artifact_store):
        if completion.identity != identity or completion.state is None:
            raise ValueError("state promotion requires this attempt's result/state completion")
        # Revalidate the durable result/state pair, not an in-memory claim.
        durable = artifact_store.read_completion(identity)
        if durable != completion:
            raise ValueError("state promotion completion differs from its durable record")
        with self._locked():
            current = self._read_current()
            if current is not None:
                previous = TaskAttemptIdentity.from_dict(current["identity"])
                if (previous.job_id, previous.site_name) != (identity.job_id, identity.site_name):
                    raise StaleTaskStateError("task state belongs to a different job/site")
            if current is not None and current["identity"] == identity.to_dict():
                if (
                    current["result_sha256"] != completion.result.sha256
                    or current["state_sha256"] != completion.state.sha256
                ):
                    raise StaleTaskStateError("an acknowledged attempt cannot replace its committed result/state")
                return current["revision"]
            revision = 0 if current is None else current["revision"]
            if revision != completion.state_revision:
                raise StaleTaskStateError("task state input revision is no longer current")
            result, _ = artifact_store.read_result(identity)
            if result.get_return_code(default=ReturnCode.OK) != ReturnCode.OK:
                raise ValueError("state promotion requires a successful task result")
            records = artifact_store.read_state(identity, completion.state)
            records = TaskState.from_wire(self.names, records).to_wire()
            checkpoint = {
                "revision": revision + 1,
                "identity": identity.to_dict(),
                "names": list(self.names),
                "records": records,
                "result_sha256": completion.result.sha256,
                "state_sha256": completion.state.sha256,
            }
            encoded = json.dumps(checkpoint, allow_nan=False, separators=(",", ":")).encode("utf-8")
            fd, temporary = tempfile.mkstemp(prefix=".state-", dir=self.root_dir)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(temporary, 0o600)
                os.replace(temporary, self.path)
                directory_fd = os.open(self.root_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            return revision + 1
