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

"""Raw traffic totals stay outside CellNet and never affect delivery."""

from unittest.mock import Mock

import pytest

from nvflare.private.fed.resource_stats.message_traffic import (
    record_external_trainer_traffic,
    register_external_trainer,
    retire_external_trainer,
    retire_job_traffic,
    snapshot_message_traffic,
)


def _cell(sent=0, received=0):
    cell = Mock()
    cell.get_traffic_stats.return_value = {
        "server.job-1": {"sent": sent, "received": received},
        "site-2.job-1": {"sent": 3, "received": 5},
    }
    return cell


def test_sum_actual_peers_without_persisting_cell_names():
    cell = _cell(7, 11)
    assert snapshot_message_traffic(cell) == {"sent_bytes": "10", "received_bytes": "16"}
    assert snapshot_message_traffic(cell) == {"sent_bytes": "10", "received_bytes": "16"}


def test_successful_zero_is_distinct_from_missing_cell():
    cell = _cell()
    cell.get_traffic_stats.return_value["site-2.job-1"] = {"sent": 0, "received": 0}
    assert snapshot_message_traffic(cell) == {"sent_bytes": "0", "received_bytes": "0"}
    assert snapshot_message_traffic(None) == {"status": "unavailable", "issues": ["observation_incomplete"]}


@pytest.mark.parametrize("bad", [True, -1, "7", None])
def test_malformed_transport_counter_is_unavailable(bad):
    cell = _cell(bad)
    assert snapshot_message_traffic(cell) == {"status": "unavailable", "issues": ["observation_incomplete"]}


def test_trainer_snapshots_are_cumulative_and_old_duplicates_cannot_double_count():
    cell = _cell(7, 11)
    register_external_trainer(cell, "trainer-1")
    record_external_trainer_traffic(cell, "trainer-1", {"sent_bytes": "100", "received_bytes": "200"})
    record_external_trainer_traffic(cell, "trainer-1", {"sent_bytes": "150", "received_bytes": "300"})
    record_external_trainer_traffic(cell, "trainer-1", {"sent_bytes": "100", "received_bytes": "200"})
    assert snapshot_message_traffic(cell) == {"sent_bytes": "160", "received_bytes": "316"}
    retire_external_trainer(cell, "trainer-1")
    retire_external_trainer(cell, "trainer-1")
    record_external_trainer_traffic(cell, "trainer-1", {"sent_bytes": "999", "received_bytes": "999"})
    assert snapshot_message_traffic(cell) == {"sent_bytes": "160", "received_bytes": "316"}


def test_missing_trainer_snapshot_keeps_observed_bytes_and_one_simple_issue():
    cell = _cell(7, 11)
    register_external_trainer(cell, "trainer-1")
    record_external_trainer_traffic(cell, "trainer-1", {"sent_bytes": "100", "received_bytes": "200"})
    register_external_trainer(cell, "trainer-2")
    record_external_trainer_traffic(cell, "trainer-2", {"bad": "data"})
    retire_external_trainer(cell, "trainer-2")
    assert snapshot_message_traffic(cell) == {
        "sent_bytes": "110",
        "received_bytes": "216",
        "status": "partial",
        "issues": ["observation_incomplete"],
    }


def test_distinct_jobs_and_unknown_sources_cannot_mix():
    cell, other = _cell(), _cell()
    register_external_trainer(cell, "trainer-1")
    record_external_trainer_traffic(other, "trainer-1", {"sent_bytes": "100", "received_bytes": "200"})
    record_external_trainer_traffic(cell, "unknown", {"sent_bytes": "100", "received_bytes": "200"})
    assert snapshot_message_traffic(cell)["status"] == "partial"
    assert snapshot_message_traffic(other) == {"sent_bytes": "3", "received_bytes": "5"}


def test_no_transport_or_trainer_observation_cannot_invent_zero():
    cell = _cell()
    cell.get_traffic_stats.side_effect = RuntimeError("unavailable")
    register_external_trainer(cell, "trainer-1")
    assert snapshot_message_traffic(cell) == {"status": "unavailable", "issues": ["observation_incomplete"]}
    record_external_trainer_traffic(cell, "trainer-1", {"sent_bytes": "100", "received_bytes": "200"})
    assert snapshot_message_traffic(cell) == {
        "sent_bytes": "100",
        "received_bytes": "200",
        "status": "partial",
        "issues": ["observation_incomplete"],
    }


def test_unavailable_final_snapshot_preserves_subtotal_but_marks_gap():
    cell = _cell()
    register_external_trainer(cell, "trainer-1")
    record_external_trainer_traffic(cell, "trainer-1", {"sent_bytes": "100", "received_bytes": "200"})
    record_external_trainer_traffic(cell, "trainer-1", {"status": "unavailable", "issues": ["observation_incomplete"]})
    retire_external_trainer(cell, "trainer-1")
    result = snapshot_message_traffic(cell)
    assert result["sent_bytes"] == "103"
    assert result["received_bytes"] == "205"
    assert result["status"] == "partial"


def test_live_source_subtotal_stays_partial_until_existing_settlement():
    cell = _cell()
    register_external_trainer(cell, "trainer-1")
    record_external_trainer_traffic(cell, "trainer-1", {"sent_bytes": "100", "received_bytes": "200"}, incomplete=True)
    assert snapshot_message_traffic(cell)["status"] == "partial"
    record_external_trainer_traffic(cell, "trainer-1", {"sent_bytes": "150", "received_bytes": "300"})
    retire_external_trainer(cell, "trainer-1")
    assert snapshot_message_traffic(cell) == {"sent_bytes": "153", "received_bytes": "305"}


def test_cleanup_of_unsettled_source_cannot_claim_complete():
    cell = _cell()
    register_external_trainer(cell, "trainer-1")
    record_external_trainer_traffic(cell, "trainer-1", {"sent_bytes": "100", "received_bytes": "200"})
    retire_external_trainer(cell, "trainer-1", incomplete=True)
    assert snapshot_message_traffic(cell)["status"] == "partial"


@pytest.mark.parametrize("sent,received", [(100, 200), (50, 100), (150, 100)])
def test_delayed_or_equal_snapshot_cannot_clear_newer_incomplete_source(sent, received):
    cell = _cell()
    register_external_trainer(cell, "trainer-1")
    record_external_trainer_traffic(cell, "trainer-1", {"sent_bytes": "100", "received_bytes": "200"}, incomplete=True)
    record_external_trainer_traffic(cell, "trainer-1", {"sent_bytes": str(sent), "received_bytes": str(received)})
    assert snapshot_message_traffic(cell)["status"] == "partial"


def test_empty_peer_map_is_observed_zero():
    cell = _cell()
    cell.get_traffic_stats.return_value = {}
    assert snapshot_message_traffic(cell) == {"sent_bytes": "0", "received_bytes": "0"}


def test_retire_job_traffic_matches_known_roots_and_preserves_shared_or_other_job_peers():
    cell = Mock()
    cell.get_fqcn.return_value = "relay-1.site-1"
    removed = {
        "server.job-1",
        "server.job-1.trainer",
        "server.ws_transfer_job-1",
        "relay-1.site-1.job-1",
        "relay-1.site-1.job-1.trainer",
        "relay-1.site-1.ws_transfer_job-1",
        "relay-2.site-2.job-1",
        "relay-2.site-2.ws_transfer_job-1",
        "site-3.job-1",
    }
    retained = {
        "server",
        "relay-1.site-1",
        "relay-1.site-1.-client_api_stable",
        "relay-1.site-1.job-10",
        "relay-1.site-1.ws_transfer_job-10",
        "relay-1.site-1.job-2.trainer",
        "unknown.job-1",
    }
    cell.get_traffic_stats.return_value = {peer: {"sent": 1, "received": 2} for peer in removed | retained}

    retire_job_traffic(
        cell,
        "job-1",
        [{"name": "site-2", "fqcn": "relay-2.site-2"}, {"name": "site-3"}],
    )

    assert {args[0] for args, _kwargs in cell.pop_traffic_stats.call_args_list} == removed


@pytest.mark.parametrize("method", ["get_fqcn", "get_traffic_stats", "pop_traffic_stats"])
def test_retire_job_traffic_failure_does_not_affect_completion(method):
    cell = _cell()
    cell.get_fqcn.return_value = "site-1"
    getattr(cell, method).side_effect = RuntimeError("unavailable")
    retire_job_traffic(cell, "job-1")


def test_retire_job_traffic_continues_after_one_peer_cannot_be_removed():
    cell = Mock()
    cell.get_fqcn.return_value = "site-1"
    cell.get_traffic_stats.return_value = {"site-1.job-1": {}, "site-1.job-1.trainer": {}}
    cell.pop_traffic_stats.side_effect = [RuntimeError("unavailable"), {}]

    retire_job_traffic(cell, "job-1")

    assert cell.pop_traffic_stats.call_count == 2
