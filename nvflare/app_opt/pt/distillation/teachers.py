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

"""Bounded teacher state snapshots, independent of models and training policy."""

from collections import deque
from copy import copy, deepcopy
from dataclasses import dataclass, field

import torch
from torch import nn


def _snapshot_state(state):
    """Copy tensor state to CPU, preserving PyTorch state_dict metadata."""
    result = copy(state)
    for name, value in state.items():
        result[name] = value.detach().cpu().clone() if isinstance(value, torch.Tensor) else deepcopy(value)
    if hasattr(state, "_metadata"):
        result._metadata = deepcopy(state._metadata)
    return result


@dataclass(frozen=True)
class TeacherSnapshot:
    """Borrowed read-only model state and optional application metadata.

    Frozen fields do not make tensor/dict contents immutable. Do not mutate
    snapshots returned by the buffer; use state_dict() for an owned export.
    Architecture, class ordering and model factories are caller-owned.
    """

    model_state: dict = field(repr=False)
    metadata: dict = field(default_factory=dict)


class TeacherBuffer:
    """Keep at most max_size CPU snapshots, newest first.

    add() copies weights, buffers and metadata, so later changes to the supplied
    model cannot change history. Capacity eviction releases the oldest snapshot.
    The buffer performs no model construction, inference or random sampling.
    """

    def __init__(self, max_size: int):
        if isinstance(max_size, bool) or not isinstance(max_size, int) or max_size < 1:
            raise ValueError("max_size must be a positive integer")
        self.max_size = max_size
        self._teachers = deque(maxlen=max_size)

    def add(self, model: nn.Module | dict, metadata: dict | None = None) -> None:
        """Capture a module or PyTorch state_dict; metadata is optional."""
        state = model.state_dict() if isinstance(model, nn.Module) else model
        if not isinstance(state, dict):
            raise TypeError("model must be a module or state_dict")
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary")
        self._teachers.appendleft(TeacherSnapshot(_snapshot_state(state), deepcopy(metadata or {})))

    def __len__(self):
        return len(self._teachers)

    def __iter__(self):
        return iter(self._teachers)

    def state_dict(self) -> dict:
        """Export independent CPU states, newest first; no model objects."""
        return {
            "max_size": self.max_size,
            "teachers": [
                {"model_state": _snapshot_state(item.model_state), "metadata": deepcopy(item.metadata)} for item in self
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore into a buffer with the same capacity; never silently evict."""
        entries = state["teachers"]
        if state["max_size"] != self.max_size or len(entries) > self.max_size:
            raise ValueError("checkpoint teacher capacity does not match this buffer")
        restored = []
        for item in entries:
            if not isinstance(item["model_state"], dict) or not isinstance(item["metadata"], dict):
                raise TypeError("checkpoint model_state and metadata must be dictionaries")
            restored.append(TeacherSnapshot(_snapshot_state(item["model_state"]), deepcopy(item["metadata"])))
        self._teachers = deque(restored, maxlen=self.max_size)
