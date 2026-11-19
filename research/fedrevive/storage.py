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

"""Append-only tensor checkpoints owned by one fresh research run."""

from dataclasses import dataclass, field
from pathlib import Path

import torch


class TensorStore:
    """Retain every base/result; no reference counts or in-run reclamation."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise FileExistsError(f"Research artifacts already exist in {self.directory}; use a fresh workspace")

    def path(self, identity):
        return self.directory / f"{identity:08d}.pt"

    def put(self, identity, value):
        # Only run-owned local files are loaded. Never accept paths from uploads.
        with self.path(identity).open("xb") as stream:
            torch.save(value, stream)

    def load(self, identity):
        return torch.load(str(self.path(identity)), map_location="cpu", weights_only=True, mmap=True)


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
