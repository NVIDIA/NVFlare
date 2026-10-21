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

"""Cumulative CellNet payload bytes, grouped by adjacent transport peer."""

import threading

MAX_TRAFFIC_PEERS = 4096
MAX_TRAFFIC_PEER_NAME_LENGTH = 512


class PeerTrafficCounters:
    """Keep bounded peer detail without dropping bytes when detail fills up.

    Existing peers remain individually visible. Bytes for additional,
    missing, or oversized peer names accumulate in ``unattributed`` instead.
    Snapshots are independent copies and do not reset collection.
    """

    def __init__(self, max_peers=MAX_TRAFFIC_PEERS):
        if type(max_peers) is not int or max_peers < 0:
            raise ValueError("max_peers must be a nonnegative integer")
        self._max_peers = max_peers
        self._lock = threading.Lock()
        self._peers = {}
        self._unattributed = {"sent_bytes": 0, "received_bytes": 0}

    def record_sent(self, peer, payload_bytes):
        self._record(peer, payload_bytes, "sent_bytes")

    def record_received(self, peer, payload_bytes):
        self._record(peer, payload_bytes, "received_bytes")

    def _record(self, peer, payload_bytes, direction):
        if not isinstance(payload_bytes, int) or isinstance(payload_bytes, bool) or payload_bytes < 0:
            return
        valid_peer = isinstance(peer, str) and 0 < len(peer) <= MAX_TRAFFIC_PEER_NAME_LENGTH
        with self._lock:
            counters = self._peers.get(peer) if valid_peer else None
            if counters is None and valid_peer and len(self._peers) < self._max_peers:
                counters = {"sent_bytes": 0, "received_bytes": 0}
                self._peers[peer] = counters
            if counters is None:
                counters = self._unattributed
            counters[direction] += payload_bytes

    def snapshot(self):
        """Return an atomic copy of all byte totals collected so far."""
        with self._lock:
            return {
                "peers": {peer: counters.copy() for peer, counters in self._peers.items()},
                "unattributed": self._unattributed.copy(),
            }
