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

"""Temporary tensor files owned by one research run."""

from dataclasses import dataclass, field
from pathlib import Path

import torch


class TensorStore:
    """Share a tensor file across its users and unlink it at last use."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise FileExistsError(f"Research artifacts already exist in {self.directory}; use a fresh workspace")
        self._users: dict[int, int] = {}

    def path(self, identity):
        return self.directory / f"{identity:08d}.pt"

    def retain(self, identity, value, users=1):
        if users < 1:
            raise ValueError("Tensor file must have at least one user")
        path = self.path(identity)
        if identity not in self._users:
            # Register ownership before serialization so close can clean up partial writes.
            with path.open("xb") as stream:
                self._users[identity] = 0
                torch.save(value, stream)
        self._users[identity] += users
        return path

    def load(self, identity):
        return torch.load(str(self.path(identity)), map_location="cpu", weights_only=True, mmap=True)

    def release(self, identity):
        users = self._users[identity] - 1
        if users:
            self._users[identity] = users
        else:
            self.path(identity).unlink(missing_ok=True)
            del self._users[identity]

    def close(self):
        for identity in self._users:
            self.path(identity).unlink(missing_ok=True)
        self._users.clear()


@dataclass(frozen=True)
class DiskAssignment:
    assignment_id: int
    client_name: str | None
    logical_name: str
    base_version: int
    base_path: str
    metadata: dict = field(default_factory=dict)

    @property
    def base_model(self):
        return torch.load(self.base_path, map_location="cpu", weights_only=True, mmap=True)
