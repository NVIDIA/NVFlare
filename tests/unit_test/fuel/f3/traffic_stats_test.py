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

from concurrent.futures import ThreadPoolExecutor

import pytest

from nvflare.fuel.f3.traffic_stats import MAX_TRAFFIC_PEER_NAME_LENGTH, PeerTrafficCounters


def test_totals_are_cumulative_and_directions_are_independent():
    counters = PeerTrafficCounters()
    assert counters.snapshot() == {"peers": {}, "unattributed": {"sent_bytes": 0, "received_bytes": 0}}

    counters.record_sent("server", 12)
    counters.record_sent("server", 8)
    counters.record_received("server", 7)
    counters.record_received("site-2", 0)

    expected = {
        "peers": {
            "server": {"sent_bytes": 20, "received_bytes": 7},
            "site-2": {"sent_bytes": 0, "received_bytes": 0},
        },
        "unattributed": {"sent_bytes": 0, "received_bytes": 0},
    }
    assert counters.snapshot() == expected
    assert counters.snapshot() == expected


def test_snapshot_does_not_share_mutable_counter_state():
    counters = PeerTrafficCounters(max_peers=1)
    counters.record_sent("server", 3)
    counters.record_received("site-2", 4)
    snapshot = counters.snapshot()

    snapshot["peers"]["server"]["sent_bytes"] = 900
    snapshot["peers"]["site-3"] = {"sent_bytes": 10, "received_bytes": 0}
    snapshot["unattributed"]["received_bytes"] = 900
    counters.record_sent("server", 2)

    assert counters.snapshot() == {
        "peers": {"server": {"sent_bytes": 5, "received_bytes": 0}},
        "unattributed": {"sent_bytes": 0, "received_bytes": 4},
    }
    assert snapshot["peers"]["server"]["sent_bytes"] == 900


def test_full_peer_detail_preserves_known_peers_and_all_other_bytes():
    counters = PeerTrafficCounters(max_peers=1)
    counters.record_sent("server", 5)
    counters.record_sent("site-2", 6)
    counters.record_received("site-2", 7)
    counters.record_sent("site-3", 8)
    counters.record_received("server", 9)
    counters.record_sent("server", 10)

    assert counters.snapshot() == {
        "peers": {"server": {"sent_bytes": 15, "received_bytes": 9}},
        "unattributed": {"sent_bytes": 14, "received_bytes": 7},
    }


def test_peer_name_length_limit_does_not_discard_bytes():
    counters = PeerTrafficCounters()
    valid_peer = "p" * MAX_TRAFFIC_PEER_NAME_LENGTH
    oversized_peer = valid_peer + "p"
    counters.record_sent(valid_peer, 1)
    counters.record_sent(oversized_peer, 2)
    counters.record_received(oversized_peer, 3)

    assert counters.snapshot() == {
        "peers": {valid_peer: {"sent_bytes": 1, "received_bytes": 0}},
        "unattributed": {"sent_bytes": 2, "received_bytes": 3},
    }


@pytest.mark.parametrize("peer", [None, "", 3, [], {}, b"server"])
def test_missing_or_invalid_peer_keeps_unattributed_totals(peer):
    counters = PeerTrafficCounters()
    counters.record_sent(peer, 2)
    counters.record_received(peer, 3)

    assert counters.snapshot() == {"peers": {}, "unattributed": {"sent_bytes": 2, "received_bytes": 3}}


@pytest.mark.parametrize("payload_bytes", [-1, True, False, 0.0, "2", None, [], {}])
def test_invalid_byte_counts_are_ignored(payload_bytes):
    counters = PeerTrafficCounters()
    counters.record_sent("server", payload_bytes)
    counters.record_received(None, payload_bytes)

    assert counters.snapshot() == {"peers": {}, "unattributed": {"sent_bytes": 0, "received_bytes": 0}}


def test_zero_peer_budget_retains_byte_totals():
    counters = PeerTrafficCounters(max_peers=0)
    counters.record_sent("server", 10)
    counters.record_received("server", 11)

    assert counters.snapshot() == {"peers": {}, "unattributed": {"sent_bytes": 10, "received_bytes": 11}}


def test_concurrent_recording_and_snapshots_keep_exact_totals():
    counters = PeerTrafficCounters(max_peers=1)

    def record_many():
        for _ in range(500):
            counters.record_sent("server", 2)
            counters.record_received("server", 3)
            counters.record_sent(None, 5)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(record_many) for _ in range(8)]
        for _ in range(50):
            snapshot = counters.snapshot()
            assert set(snapshot["peers"]).issubset({"server"})
            assert snapshot["unattributed"]["sent_bytes"] % 5 == 0
            if snapshot["peers"]:
                assert snapshot["peers"]["server"]["sent_bytes"] % 2 == 0
                assert snapshot["peers"]["server"]["received_bytes"] % 3 == 0
        for future in futures:
            future.result()

    assert counters.snapshot() == {
        "peers": {"server": {"sent_bytes": 8000, "received_bytes": 12000}},
        "unattributed": {"sent_bytes": 20000, "received_bytes": 0},
    }
