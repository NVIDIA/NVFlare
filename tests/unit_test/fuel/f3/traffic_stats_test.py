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
from threading import Barrier

import pytest

from nvflare.fuel.f3.traffic_stats import PeerTrafficCounters


def test_totals_are_cumulative_and_directions_are_independent():
    counters = PeerTrafficCounters()
    assert counters.snapshot() == {}

    counters.record_sent("server", 12)
    counters.record_sent("server", 8)
    counters.record_received("server", 7)
    counters.record_received("site-2", 0)

    expected = {
        "server": {"sent": 20, "received": 7},
        "site-2": {"sent": 0, "received": 0},
    }
    assert counters.snapshot() == expected
    assert counters.snapshot() == expected


def test_snapshot_does_not_share_mutable_counter_state():
    counters = PeerTrafficCounters()
    counters.record_sent("server", 3)
    counters.record_received("site-2", 4)
    snapshot = counters.snapshot()

    snapshot["server"]["sent"] = 900
    snapshot["site-3"] = {"sent": 10, "received": 0}
    counters.record_sent("server", 2)

    assert counters.snapshot() == {
        "server": {"sent": 5, "received": 0},
        "site-2": {"sent": 0, "received": 4},
    }
    assert snapshot["server"]["sent"] == 900


def test_all_peers_and_long_names_retain_individual_totals():
    counters = PeerTrafficCounters()
    for index in range(5000):
        counters.record_sent(f"site-{index}", index)
    long_peer = "p" * 1024
    counters.record_sent(long_peer, 2)
    counters.record_received(long_peer, 3)

    snapshot = counters.snapshot()
    assert len(snapshot) == 5001
    assert all(snapshot[f"site-{index}"] == {"sent": index, "received": 0} for index in range(5000))
    assert snapshot[long_peer] == {"sent": 2, "received": 3}


@pytest.mark.parametrize("peer", [None, "", 3, [], {}, b"server"])
def test_missing_or_invalid_peer_is_rejected(peer):
    counters = PeerTrafficCounters()
    for operation in (counters.record_sent, counters.record_received):
        with pytest.raises(ValueError, match="peer must be a nonempty string"):
            operation(peer, 2)
    with pytest.raises(ValueError, match="peer must be a nonempty string"):
        counters.pop(peer)

    assert counters.snapshot() == {}


@pytest.mark.parametrize("payload_bytes", [-1, True, False, 0.0, "2", None, [], {}])
def test_invalid_byte_counts_are_ignored(payload_bytes):
    counters = PeerTrafficCounters()
    counters.record_sent("server", payload_bytes)
    counters.record_received("server", payload_bytes)

    assert counters.snapshot() == {}


def test_pop_removes_only_requested_peer_and_restarts_its_totals():
    counters = PeerTrafficCounters()
    counters.record_sent("server", 10)
    counters.record_received("server", 11)
    counters.record_sent("site-2", 12)

    removed = counters.pop("server")
    assert removed == {"sent": 10, "received": 11}
    assert counters.pop("server") is None
    assert counters.snapshot() == {"site-2": {"sent": 12, "received": 0}}

    counters.record_received("server", 3)
    removed["received"] = 900
    assert counters.snapshot() == {
        "site-2": {"sent": 12, "received": 0},
        "server": {"sent": 0, "received": 3},
    }


def test_concurrent_recording_and_snapshots_keep_exact_totals():
    counters = PeerTrafficCounters()

    def record_many():
        for _ in range(500):
            counters.record_sent("server", 2)
            counters.record_received("server", 3)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(record_many) for _ in range(8)]
        for _ in range(50):
            snapshot = counters.snapshot()
            assert set(snapshot).issubset({"server"})
            if snapshot:
                assert snapshot["server"]["sent"] % 2 == 0
                assert snapshot["server"]["received"] % 3 == 0
        for future in futures:
            future.result()

    assert counters.snapshot() == {"server": {"sent": 8000, "received": 12000}}


def test_concurrent_recording_and_removal_preserve_every_byte():
    counters = PeerTrafficCounters()
    start = Barrier(5)

    def record_many():
        start.wait()
        for _ in range(500):
            counters.record_sent("server", 2)
            counters.record_received("server", 3)

    def remove_many():
        start.wait()
        removed = {"sent": 0, "received": 0}
        for _ in range(500):
            totals = counters.pop("server")
            if totals is not None:
                for direction in removed:
                    removed[direction] += totals[direction]
        return removed

    with ThreadPoolExecutor(max_workers=5) as executor:
        recorders = [executor.submit(record_many) for _ in range(4)]
        remover = executor.submit(remove_many)
        for recorder in recorders:
            recorder.result()
        removed = remover.result()

    remaining = counters.pop("server") or {"sent": 0, "received": 0}
    assert removed["sent"] + remaining["sent"] == 4000
    assert removed["received"] + remaining["received"] == 6000
    assert counters.snapshot() == {}
