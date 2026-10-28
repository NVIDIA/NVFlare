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

"""Storage contracts for exact assignment bases and out-of-order results.

Custom stores can offload payloads without changing scheduling or aggregation.
Stores belong to one run and must make close() safe after partial initialization.
"""

from typing import Protocol


class SnapshotStore(Protocol):
    """Retain immutable bases until acceptance/discard; handles stay server-local."""

    def retain(self, version, model, adapter):
        """Acquire one reference to a version, returning an opaque handle."""

    def load(self, handle):
        """Borrow the retained model; loading may materialize offloaded state."""

    def release(self, handle):
        """End assignment ownership; durable stores may retain checkpoint files."""

    def close(self):
        """Release run resources, including after failure; durable artifacts may remain."""


class MemorySnapshotStore:
    """Share one owned snapshot per active base version in RAM."""

    def __init__(self):
        self._versions = {}

    def retain(self, version, model, adapter):
        if version not in self._versions:
            self._versions[version] = [adapter.snapshot(model), 0]
        self._versions[version][1] += 1
        return version

    def load(self, handle):
        return self._versions[handle][0]

    def release(self, handle):
        entry = self._versions[handle]
        entry[1] -= 1
        if not entry[1]:
            del self._versions[handle]

    def close(self):
        self._versions.clear()


class ResultStore(Protocol):
    """Own early raw Collab results until their scheduled acceptance.

    put/take/close run on the workflow thread. release can also run on a result
    reader after abort and must be thread-safe, including after close. take()
    transfers ownership to the caller; a durable checkpoint may remain on disk.
    """

    def put(self, assignment_id, result):
        """Take ownership of one completed result."""

    def take(self, assignment_id):
        """Return the selected result, materializing from storage when needed."""

    def release(self, result):
        """Release a taken/discarded result, including lazy backing files."""

    def close(self):
        """Release all stored results and any run-owned offload context."""


class MemoryResultStore:
    """Hold at most K completed results in RAM; ownership ends at take/close."""

    def __init__(self):
        self._results = {}

    def put(self, assignment_id, result):
        if assignment_id in self._results:
            raise ValueError("duplicate stored result")
        self._results[assignment_id] = result

    def take(self, assignment_id):
        return self._results.pop(assignment_id)

    def release(self, result):
        pass

    def close(self):
        self._results.clear()
