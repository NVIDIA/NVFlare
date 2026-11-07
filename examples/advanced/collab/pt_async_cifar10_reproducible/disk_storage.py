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

"""Append-only local checkpoints for the research example."""

import pickle
from pathlib import Path


def load_checkpoint(path):
    # Only load run-owned local checkpoints, never client-supplied file paths.
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


class DiskStore:
    """One file per identity in a fresh directory; reads retain every file."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise FileExistsError(f"Research artifacts already exist in {self.directory}; use a fresh directory")

    def path(self, identity):
        return self.directory / f"{identity:08d}.pkl"

    def put(self, identity, value):
        with self.path(identity).open("xb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, identity):
        return load_checkpoint(self.path(identity))
