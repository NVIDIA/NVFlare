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

"""Task reduction policies use controlled records, without launching workers."""

from copy import deepcopy
from dataclasses import replace
from unittest.mock import Mock

import pytest

from nvflare.private.fed.resource_stats import task_rollup
from nvflare.private.fed.resource_stats.contract import U64_MAX, U128_MAX
from nvflare.private.fed.task_worker.protocol import TaskAttemptIdentity


def _identity(attempt="first"):
    return TaskAttemptIdentity("job", "site", "task", "train", attempt)


def _report(identity, *, seconds="10", cpu_seconds="8", model="CPU", **changes):
    record = {
        "schema_version": "1.0",
        "identity": identity.to_dict(),
        "scope": "worker",
        "duration_seconds": seconds,
        "resource_capacity": {
            "cpu": {"groups": [{"units": "2", "model": model}]},
            "memory": {"bytes": "1024"},
            "gpu": {"groups": [{"kind": "cuda_device_unclassified", "count": "1"}]},
        },
        "cpu_consumed": {"cpu_seconds": cpu_seconds},
        "message_traffic": {"sent_bytes": "10", "received_bytes": "20"},
    }
    record.update(changes)
    return record


def _store(record):
    return Mock(
        read_resource_stats=Mock(side_effect=lambda _identity: deepcopy(record)),
        remove_resource_stats=Mock(),
    )


def _fold(reducer, identity, record):
    assert reducer.register(identity)
    reducer.bind(identity, "backend:" + identity.attempt_id)
    store = _store(record)
    reducer.settle(identity, store, started=True, settled=True)
    return store


def test_no_workers_adds_nothing_and_never_started_work_does_not_create_a_gap():
    reducer = task_rollup.TaskResourceReducer()
    assert reducer.snapshot() == {
        "resource_time": None,
        "cpu_consumed": None,
        "message_traffic": None,
        "duration_contained": True,
    }
    identity = _identity()
    assert reducer.register(identity)
    store = Mock()
    reducer.settle(identity, store, started=False)
    store.read_resource_stats.assert_not_called()
    store.remove_resource_stats.assert_not_called()
    assert reducer.snapshot()["resource_time"] is None
    assert reducer._pending_bytes == 0


def test_distinct_physical_workers_sum_their_own_lifetimes():
    reducer = task_rollup.TaskResourceReducer()
    first, second = _identity(), _identity("second")
    _fold(reducer, first, _report(first))
    _fold(reducer, second, _report(second, seconds="5", cpu_seconds="4"))
    values = reducer.snapshot()
    assert values["resource_time"] == {
        "cpu": {"groups": [{"model": "CPU", "unit_seconds": "30"}]},
        "memory": {"byte_seconds": "15360"},
        "gpu": {"groups": [{"kind": "cuda_device_unclassified", "instance_seconds": "15"}]},
    }
    assert values["cpu_consumed"] == {"cpu_seconds": "12"}
    assert values["message_traffic"] == {"sent_bytes": "20", "received_bytes": "40"}
    assert values["duration_contained"]
    assert reducer._pending_bytes == 0


def test_equivalent_repeated_reports_do_not_add_compute_or_traffic():
    reducer = task_rollup.TaskResourceReducer()
    identity = _identity()
    store = _fold(reducer, identity, _report(identity))
    expected = reducer.snapshot()
    store.read_resource_stats.side_effect = None
    store.read_resource_stats.return_value = _report(identity, seconds="10.0", cpu_seconds="8.0")
    reducer.settle(identity, store, started=True, settled=True)
    assert reducer.snapshot() == expected
    assert not reducer.register(identity)


def test_conflicting_repeat_retains_first_subtotal_and_marks_a_gap():
    reducer = task_rollup.TaskResourceReducer()
    identity = _identity()
    store = _fold(reducer, identity, _report(identity))
    store.read_resource_stats.side_effect = None
    store.read_resource_stats.return_value = _report(identity, seconds="20", cpu_seconds="16")
    reducer.settle(identity, store, started=True, settled=True)
    values = reducer.snapshot()
    assert values["resource_time"]["gpu"]["groups"][0]["instance_seconds"] == "10"
    assert values["cpu_consumed"] == {
        "status": "partial",
        "issues": ["observation_incomplete"],
        "cpu_seconds": "8",
    }
    assert values["message_traffic"]["sent_bytes"] == "10"


@pytest.mark.parametrize("settled", [False, True])
def test_missing_or_unsettled_worker_keeps_existing_values_and_marks_quality(settled):
    reducer = task_rollup.TaskResourceReducer()
    identity = _identity()
    _fold(reducer, identity, _report(identity))
    missing = _identity("missing")
    assert reducer.register(missing)
    store = _store(None)
    reducer.settle(missing, store, started=True, settled=settled)
    values = reducer.snapshot()
    assert values["resource_time"]["gpu"]["groups"][0]["instance_seconds"] == "10"
    assert values["resource_time"]["status"] == "partial"
    assert values["cpu_consumed"]["cpu_seconds"] == "8"
    assert values["duration_contained"] is settled
    assert store.read_resource_stats.call_count == int(settled)


def test_late_confirmed_settlement_can_resolve_a_temporary_scope_gap():
    reducer = task_rollup.TaskResourceReducer()
    identity = _identity()
    store = _store(_report(identity))
    assert reducer.register(identity)
    reducer.settle(identity, store, started=True, settled=False)
    assert not reducer.snapshot()["duration_contained"]
    reducer.settle(identity, store, started=True, settled=True)
    values = reducer.snapshot()
    assert values["duration_contained"]
    assert values["cpu_consumed"] == {"cpu_seconds": "8"}
    assert "status" not in values["resource_time"]


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": "2.0"},
        {"scope": "unregistered-rank"},
        {"identity": _identity("other").to_dict()},
        {"unexpected": "not a private protocol field"},
    ],
)
def test_invalid_report_cannot_supply_values_or_escape_its_owned_identity(changes):
    reducer = task_rollup.TaskResourceReducer()
    identity = _identity()
    record = _report(identity)
    record.update(changes)
    store = _fold(reducer, identity, record)
    assert reducer.snapshot()["resource_time"] == {
        "status": "unavailable",
        "issues": ["observation_incomplete"],
    }
    store.remove_resource_stats.assert_called_once_with(identity)


def test_partial_cpu_does_not_discard_complete_capacity_or_traffic():
    reducer = task_rollup.TaskResourceReducer()
    identity = _identity()
    _fold(
        reducer,
        identity,
        _report(
            identity,
            cpu_consumed={"status": "partial", "issues": ["attribution_incomplete"], "cpu_seconds": "8"},
        ),
    )
    values = reducer.snapshot()
    assert "status" not in values["resource_time"]
    assert values["cpu_consumed"]["issues"] == ["attribution_incomplete"]
    assert "status" not in values["message_traffic"]


@pytest.mark.parametrize("reason", ["unsupported", "malformed_source"])
def test_unavailable_capacity_keeps_prior_capacity_and_this_workers_cpu_and_traffic(reason):
    reducer = task_rollup.TaskResourceReducer()
    first, second = _identity(), _identity("second")
    _fold(reducer, first, _report(first))
    _fold(
        reducer,
        second,
        _report(second, resource_capacity={"status": "unavailable", "issues": [reason]}),
    )
    values = reducer.snapshot()
    assert values["resource_time"]["gpu"]["groups"][0]["instance_seconds"] == "10"
    assert values["resource_time"]["issues"] == ["observation_incomplete"]
    assert values["cpu_consumed"] == {"cpu_seconds": "16"}
    assert values["message_traffic"] == {"sent_bytes": "20", "received_bytes": "40"}


def test_unregistered_or_mismatched_claims_never_add_a_new_execution():
    reducer = task_rollup.TaskResourceReducer()
    identity = _identity()
    store = _store(_report(identity))
    reducer.settle(identity, store, started=True, settled=True)
    store.read_resource_stats.assert_not_called()
    assert reducer.snapshot()["resource_time"] is None
    assert reducer.register(identity)
    reducer.bind(replace(identity, task_name="different"), "unexpected")
    assert reducer.snapshot()["resource_time"]["status"] == "unavailable"


def test_key_limit_freezes_without_evicting_or_recounting_existing_sources(monkeypatch):
    monkeypatch.setattr(task_rollup, "_MAX_SCOPES", 1)
    reducer = task_rollup.TaskResourceReducer()
    identity = _identity()
    store = _fold(reducer, identity, _report(identity))
    assert not reducer.register(_identity("new"))
    reducer.settle(identity, store, started=True, settled=True)
    assert len(reducer._scopes) == 1
    assert reducer.snapshot()["cpu_consumed"]["cpu_seconds"] == "8"
    assert reducer.snapshot()["cpu_consumed"]["status"] == "partial"


def test_pending_budget_is_shared_and_is_released_after_fold(monkeypatch):
    monkeypatch.setattr(task_rollup, "_MAX_PENDING_BYTES", task_rollup.MAX_SIDECAR_BYTES)
    reducer = task_rollup.TaskResourceReducer()
    first, second = _identity(), _identity("second")
    assert reducer.register(first)
    assert not reducer.register(second)
    reducer.settle(first, _store(_report(first)), started=True, settled=True)
    assert reducer.register(second)
    assert reducer._pending_bytes == task_rollup.MAX_SIDECAR_BYTES
    assert reducer.snapshot()["cpu_consumed"]["status"] == "partial"


def test_serialized_ledger_limit_rejects_growth_before_storing_a_contribution(monkeypatch):
    monkeypatch.setattr(task_rollup, "_MAX_STATE_BYTES", 2000)
    reducer = task_rollup.TaskResourceReducer()
    identity = _identity()
    record = _report(identity)
    record["resource_capacity"]["cpu"]["groups"] = [
        {"units": "1", "model": f"CPU variant {index:03}"} for index in range(50)
    ]
    _fold(reducer, identity, record)
    assert reducer._frozen
    assert reducer._scopes[task_rollup._key(identity)].record is None
    assert reducer.snapshot()["cpu_consumed"]["status"] == "unavailable"
    assert reducer._pending_bytes == 0


def test_hardware_group_limit_keeps_other_dimensions_and_disjoint_cpu_work(monkeypatch):
    monkeypatch.setattr(task_rollup, "_MAX_GROUPS", 1)
    reducer = task_rollup.TaskResourceReducer()
    first, second = _identity(), _identity("second")
    _fold(reducer, first, _report(first, model="A"))
    _fold(reducer, second, _report(second, model="B"))
    values = reducer.snapshot()
    assert values["resource_time"]["cpu"]["groups"] == [{"model": "A", "unit_seconds": "20"}]
    assert values["resource_time"]["memory"]["byte_seconds"] == "20480"
    assert values["resource_time"]["gpu"]["groups"][0]["instance_seconds"] == "20"
    assert values["resource_time"]["status"] == "partial"
    assert values["cpu_consumed"] == {"cpu_seconds": "16"}


def test_overflow_preserves_valid_dimensions_and_the_previous_cpu_subtotal():
    reducer = task_rollup.TaskResourceReducer()
    first, second = _identity(), _identity("second")
    _fold(reducer, first, _report(first, seconds="1", cpu_seconds=str(U128_MAX)))
    record = _report(second, seconds="1000000000000000000000000000000", cpu_seconds="1")
    record["resource_capacity"]["memory"]["bytes"] = str(U64_MAX)
    _fold(reducer, second, record)
    values = reducer.snapshot()
    assert values["resource_time"]["gpu"]["groups"][0]["instance_seconds"] == "1000000000000000000000000000001"
    assert values["resource_time"]["memory"]["byte_seconds"] == "1024"
    assert values["resource_time"]["status"] == "partial"
    assert values["cpu_consumed"]["cpu_seconds"] == str(U128_MAX)
    assert values["cpu_consumed"]["status"] == "partial"


def test_cleanup_failure_keeps_values_and_makes_collection_incomplete():
    reducer = task_rollup.TaskResourceReducer()
    identity = _identity()
    store = _store(_report(identity))
    store.remove_resource_stats.side_effect = OSError("workspace unavailable")
    assert reducer.register(identity)
    reducer.settle(identity, store, started=True, settled=True)
    assert reducer.snapshot()["cpu_consumed"]["cpu_seconds"] == "8"
    assert reducer.snapshot()["cpu_consumed"]["status"] == "partial"
    assert reducer._pending_bytes == task_rollup.MAX_SIDECAR_BYTES


def test_failed_reads_and_retirement_retain_reservation_until_successful_retry(monkeypatch):
    monkeypatch.setattr(task_rollup, "_MAX_PENDING_BYTES", task_rollup.MAX_SIDECAR_BYTES)
    reducer = task_rollup.TaskResourceReducer()
    identity = _identity()
    store = _store(None)
    store.read_resource_stats.side_effect = OSError("cannot read worker file")
    store.remove_resource_stats.side_effect = OSError("cannot remove worker file")
    assert reducer.register(identity)
    reducer.settle(identity, store, started=True, settled=True)
    reducer.settle(identity, store, started=True, settled=True)
    assert reducer._pending_bytes == task_rollup.MAX_SIDECAR_BYTES
    assert not reducer.register(_identity("second"))
    assert not reducer.register(identity)

    store.remove_resource_stats.side_effect = None
    reducer.settle(identity, store, started=True, settled=True)
    reducer.settle(identity, store, started=True, settled=True)
    assert reducer._pending_bytes == 0
    assert reducer.register(_identity("second"))


def test_fork_reset_discards_parent_ledger_without_acquiring_an_inherited_lock():
    reducer = task_rollup.TaskResourceReducer()
    task_rollup.install_task_resource_reducer(reducer)
    assert task_rollup.get_task_resource_reducer() is reducer
    old_lock = task_rollup._ACTIVE_LOCK
    task_rollup._reset_after_fork()
    assert task_rollup.get_task_resource_reducer() is None
    assert task_rollup._ACTIVE_LOCK is not old_lock
