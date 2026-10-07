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

"""Explicit, bounded application state; never a snapshot of a Python process."""

import base64
import copy
import json
import math
import re
from collections.abc import MutableMapping

TASK_STATE_KEY = "__task_state__"
MAX_TASK_STATE_BYTES = 1024 * 1024
MAX_TASK_STATE_NAMES = 64


def _validate_json(value, depth=0):
    if depth > 32:
        raise ValueError("task state nesting exceeds 32 levels")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _validate_json(item, depth + 1)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _validate_json(item, depth + 1)
        return
    raise ValueError("task state records must be JSON-compatible data or explicitly serialized bytes")


class TaskState(MutableMapping):
    """Named records explicitly declared by the application.

    Values are JSON-compatible data or bytes serialized by the application.
    Reads return copies: modify a value and assign it back to stage an update.
    A worker's updates become candidates only after successful finalization;
    the supervising runtime promotes them with the admitted result. State is
    local to a job/site and independent of transient attempt-payload retention.
    """

    def __init__(self, names=(), records=None):
        self._names = self.validate_names(names)
        self._records = {}
        for name, value in ({} if records is None else records).items():
            self[name] = value

    @property
    def names(self):
        return self._names

    @staticmethod
    def validate_names(names):
        if not isinstance(names, (tuple, list)):
            raise ValueError("task_state.names must be a list of unique record names")
        if len(names) > MAX_TASK_STATE_NAMES:
            raise ValueError(f"task_state.names exceeds the {MAX_TASK_STATE_NAMES}-record limit")
        if any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,127}", name) for name in names):
            raise ValueError("task_state.names contains an invalid record name")
        if len(set(names)) != len(names):
            raise ValueError("task_state.names must contain unique record names")
        return tuple(names)

    def __getitem__(self, name):
        return copy.deepcopy(self._records[name])

    def __setitem__(self, name, value):
        if name not in self.names:
            raise KeyError(f"task state record {name!r} was not declared")
        if type(value) is not bytes:
            _validate_json(value)
        old = self._records.get(name)
        existed = name in self._records
        self._records[name] = copy.deepcopy(value)
        try:
            self.to_wire()
        except BaseException:
            if existed:
                self._records[name] = old
            else:
                self._records.pop(name, None)
            raise

    def __delitem__(self, name):
        del self._records[name]

    def __iter__(self):
        return iter(self._records)

    def __len__(self):
        return len(self._records)

    def to_wire(self):
        records = {}
        for name, value in self._records.items():
            if type(value) is bytes:
                records[name] = {"encoding": "bytes", "value": base64.b64encode(value).decode("ascii")}
            else:
                _validate_json(value)
                records[name] = {"encoding": "json", "value": copy.deepcopy(value)}
        encoded = json.dumps(records, allow_nan=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) > MAX_TASK_STATE_BYTES:
            raise ValueError(f"task state exceeds the {MAX_TASK_STATE_BYTES}-byte encoded limit")
        return records

    @classmethod
    def from_wire(cls, names, records):
        if not isinstance(records, dict):
            raise ValueError("task state records must be a mapping")
        state = cls(names)
        for name, record in records.items():
            if not isinstance(record, dict) or set(record) != {"encoding", "value"}:
                raise ValueError("invalid task state record")
            value = record["value"]
            if record["encoding"] == "bytes":
                if not isinstance(value, str):
                    raise ValueError("invalid serialized task state")
                try:
                    value = base64.b64decode(value, validate=True)
                except ValueError as e:
                    raise ValueError("invalid serialized task state") from e
            elif record["encoding"] != "json":
                raise ValueError("unsupported task state encoding")
            else:
                _validate_json(value)
            state[name] = value
        state.to_wire()
        return state


def get_task_state(fl_ctx):
    """Get the declared state bound to an Executor's current local context."""
    state = fl_ctx.get_prop(TASK_STATE_KEY)
    if not isinstance(state, TaskState):
        raise RuntimeError("declared task state is not available in this runtime")
    return state
