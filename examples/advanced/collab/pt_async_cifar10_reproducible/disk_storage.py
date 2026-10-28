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

"""Append-only local checkpoints for research replay; no reference counting."""

import pickle
from pathlib import Path

from nvflare.collab import collab


class _DiskStore:
    directory_name = None

    def __init__(self, root_dir: str | None = None):
        self.root_dir = root_dir
        self._directory = None

    def initialize(self):
        if self.root_dir is None:
            run_dir = Path(collab.workspace.get_run_dir(collab.fl_ctx.get_job_id()))
            directory = run_dir / self.directory_name
        else:
            directory = Path(self.root_dir)
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            raise FileExistsError(f"Research artifacts already exist in {directory}; use a fresh directory")
        self._directory = directory

    def _path(self, identity):
        return self._directory / f"{identity:08d}.pkl"

    @staticmethod
    def _save(path, value):
        # These are run-owned local checkpoints, not client-supplied file paths.
        with path.open("wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load(path):
        with Path(path).open("rb") as stream:
            return pickle.load(stream)


class DiskSnapshotStore(_DiskStore):
    """One immutable file per base version; assignment handles are file paths."""

    directory_name = "assignment_bases"

    def retain(self, version, model, adapter):
        path = self._path(version)
        if not path.exists():
            self._save(path, adapter.snapshot(model))
        return str(path)

    def load(self, handle):
        return self._load(handle)


class DiskResultStore(_DiskStore):
    """Persist each raw Collab result by ID; acceptance reloads it from disk."""

    directory_name = "assignment_results"

    def put(self, assignment_id, result):
        path = self._path(assignment_id)
        if path.exists():
            raise ValueError(f"Result {assignment_id} was already stored")
        self._save(path, result)

    def take(self, assignment_id):
        return self._load(self._path(assignment_id))
