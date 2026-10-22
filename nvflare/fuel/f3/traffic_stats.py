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

"""CellNet payload bytes, grouped by adjacent transport peer."""

import threading


class PeerTrafficCounters:
    """Keep byte totals for each peer until its owner explicitly removes them.

    Snapshots are independent copies and do not reset collection. After
    ``pop(peer)``, subsequent traffic starts a new entry for that peer.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._peers = {}

    def record_sent(self, peer, payload_bytes):
        self._record(peer, payload_bytes, "sent")

    def record_received(self, peer, payload_bytes):
        self._record(peer, payload_bytes, "received")

    def _record(self, peer, payload_bytes, direction):
        if not isinstance(peer, str) or not peer:
            raise ValueError("peer must be a nonempty string")
        if not isinstance(payload_bytes, int) or isinstance(payload_bytes, bool) or payload_bytes < 0:
            return
        with self._lock:
            counters = self._peers.get(peer)
            if counters is None:
                counters = {"sent": 0, "received": 0}
                self._peers[peer] = counters
            counters[direction] += payload_bytes

    def snapshot(self):
        """Return an atomic copy of the retained per-peer byte totals."""
        with self._lock:
            return {peer: counters.copy() for peer, counters in self._peers.items()}

    def pop(self, peer):
        """Remove and return one peer's totals, or ``None`` if it is absent.

        Concurrent updates are either included in the returned totals or
        retained in a new entry. Other peers are unchanged.
        """
        if not isinstance(peer, str) or not peer:
            raise ValueError("peer must be a nonempty string")
        with self._lock:
            counters = self._peers.pop(peer, None)
            return counters.copy() if counters is not None else None
