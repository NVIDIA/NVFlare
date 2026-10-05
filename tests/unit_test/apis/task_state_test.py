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

import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.apis.task_state import (
    MAX_TASK_STATE_BYTES,
    MAX_TASK_STATE_NAMES,
    TASK_STATE_KEY,
    TaskState,
    get_task_state,
)


def test_record_declarations_are_canonical_bounded_and_read_only():
    names = ["optimizer", "metrics.round-1", "A" + "x" * 127]
    state = TaskState(names)
    names.append("other")
    assert state.names == ("optimizer", "metrics.round-1", "A" + "x" * 127)
    assert len(TaskState([f"record{i}" for i in range(MAX_TASK_STATE_NAMES)]).names) == MAX_TASK_STATE_NAMES
    assert not TaskState().names
    with pytest.raises(AttributeError):
        state.names = ("other",)


@pytest.mark.parametrize(
    "names",
    [
        None,
        "optimizer",
        {"optimizer"},
        ["optimizer", "optimizer"],
        [None],
        [""],
        ["1optimizer"],
        ["../optimizer"],
        ["with space"],
        ["x" * 129],
        ["métrics"],
        [f"record{i}" for i in range(MAX_TASK_STATE_NAMES + 1)],
    ],
)
def test_invalid_record_declarations_are_rejected(names):
    with pytest.raises(ValueError, match="task_state.names"):
        TaskState(names)


@pytest.mark.parametrize("value", [None, True, 17, -4.5, "text", [1, {"ok": True}], {"a": [None, "b"]}, b"\x00\xff"])
def test_json_and_explicitly_serialized_records_round_trip(value):
    state = TaskState(["record"], {"record": value})
    restored = TaskState.from_wire(state.names, state.to_wire())
    assert restored["record"] == value
    assert dict(restored) == {"record": value}
    assert list(restored) == ["record"]
    assert len(restored) == 1
    del restored["record"]
    assert len(restored) == 0
    with pytest.raises(KeyError):
        restored["record"]


def test_input_reads_and_wire_views_cannot_mutate_staged_records():
    original = {"nested": [1]}
    state = TaskState(["record"], {"record": original})
    original["nested"].append(2)
    read = state["record"]
    read["nested"].append(3)
    wire = state.to_wire()
    wire["record"]["value"]["nested"].append(4)
    assert state["record"] == {"nested": [1]}
    state["record"] = read
    assert state["record"] == {"nested": [1, 3]}


@pytest.mark.parametrize(
    "value", [object(), (1, 2), {1: "integer-key"}, {"nested": b"bytes"}, float("nan"), float("inf")]
)
def test_python_objects_and_non_json_data_are_not_snapshotted(value):
    state = TaskState(["record"], {"record": "old"})
    with pytest.raises(ValueError, match="JSON-compatible"):
        state["record"] = value
    assert state["record"] == "old"


def test_deep_and_cyclic_data_are_rejected_without_recursion_failure():
    deep = None
    for _ in range(33):
        deep = [deep]
    cyclic = []
    cyclic.append(cyclic)
    state = TaskState(["record"])
    for value in (deep, cyclic):
        with pytest.raises(ValueError, match="32 levels"):
            state["record"] = value
    assert not state


@pytest.mark.parametrize("oversize", ["x" * MAX_TASK_STATE_BYTES, b"x" * MAX_TASK_STATE_BYTES])
def test_aggregate_encoded_size_limit_rolls_back_new_and_replacement_writes(oversize):
    state = TaskState(["first", "second"], {"first": "old"})
    with pytest.raises(ValueError, match="encoded limit"):
        state["second"] = oversize
    assert dict(state) == {"first": "old"}
    with pytest.raises(ValueError, match="encoded limit"):
        state["first"] = oversize
    assert state["first"] == "old"


def test_only_declared_records_can_be_written_or_restored():
    state = TaskState(["optimizer"])
    with pytest.raises(KeyError, match="not declared"):
        state["other"] = 1
    with pytest.raises(KeyError, match="not declared"):
        TaskState.from_wire(["optimizer"], {"other": {"encoding": "json", "value": 1}})


def test_size_limit_applies_to_all_records_together():
    state = TaskState(["first", "second"], {"first": "x" * (MAX_TASK_STATE_BYTES // 2)})
    with pytest.raises(ValueError, match="encoded limit"):
        state["second"] = "y" * (MAX_TASK_STATE_BYTES // 2)
    assert list(state) == ["first"]


@pytest.mark.parametrize(
    "records",
    [
        None,
        [],
        {"record": []},
        {"record": {"encoding": "json"}},
        {"record": {"encoding": "json", "value": 1, "extra": True}},
        {"record": {"encoding": "pickle", "value": "unsafe"}},
        {"record": {"encoding": "bytes", "value": 1}},
        {"record": {"encoding": "bytes", "value": "!invalid!"}},
        {"record": {"encoding": "json", "value": b"not-json"}},
        {"record": {"encoding": "json", "value": float("nan")}},
    ],
)
def test_invalid_wire_records_are_rejected(records):
    with pytest.raises(ValueError):
        TaskState.from_wire(["record"], records)


def test_executor_access_requires_a_task_state_context_binding():
    fl_ctx = FLContext()
    with pytest.raises(RuntimeError, match="not available"):
        get_task_state(fl_ctx)
    fl_ctx.set_prop(TASK_STATE_KEY, {}, private=True, sticky=False)
    with pytest.raises(RuntimeError, match="not available"):
        get_task_state(fl_ctx)
    state = TaskState(["optimizer"])
    fl_ctx.set_prop(TASK_STATE_KEY, state, private=True, sticky=False)
    assert get_task_state(fl_ctx) is state
