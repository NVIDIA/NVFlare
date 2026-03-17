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

"""Tests for via_downloader fixes (YT2).

YT2: _MIN_DOWNLOAD_TIMEOUT was set to 3600s, but it is an inactivity timeout
(time between consecutive chunk requests), not a total-transfer budget.  Active
downloads refresh last_active_time with every chunk, so the timeout only fires
when the receiver stalls.  3600s was incorrect for this semantic; the fix
changes it to 300s (5 min), which covers GC pauses and CPU-bound deserialization.

Notes:
- _on_tx_done (GC callback via transaction_done_cb) was removed in Fix 4.
  The test_gc_callback_removed test in test_download_complete_gating.py
  verifies its absence.
- DOWNLOAD_COMPLETE_CB wiring tests are in test_download_complete_gating.py.
"""

from nvflare.fuel.utils.fobs.decomposers.via_downloader import _MIN_DOWNLOAD_TIMEOUT


class TestMinDownloadTimeout:
    """YT2 fix: _MIN_DOWNLOAD_TIMEOUT must be 300s (inactivity timeout), not 3600s."""

    def test_value_is_300(self):
        """_MIN_DOWNLOAD_TIMEOUT must be 300 after YT2 fix (was 3600)."""
        assert _MIN_DOWNLOAD_TIMEOUT == 300, (
            f"_MIN_DOWNLOAD_TIMEOUT must be 300 (inactivity timeout, not total-transfer budget). "
            f"Got {_MIN_DOWNLOAD_TIMEOUT}. "
            "See YT2 fix: active downloads refresh last_active_time with every chunk; "
            "300s covers GC pauses between requests."
        )

    def test_value_is_not_old_3600(self):
        """Regression: _MIN_DOWNLOAD_TIMEOUT must NOT be 3600 (old incorrect value)."""
        assert _MIN_DOWNLOAD_TIMEOUT != 3600, (
            "_MIN_DOWNLOAD_TIMEOUT must not be 3600 — that value was wrong: it was "
            "mistaken for a total-transfer timeout but is actually an inactivity timeout."
        )

    def test_value_is_positive(self):
        """_MIN_DOWNLOAD_TIMEOUT must be positive."""
        assert _MIN_DOWNLOAD_TIMEOUT > 0

    def test_on_tx_done_removed(self):
        """_on_tx_done (GC via transaction_done_cb) must no longer exist.

        _on_tx_done was removed because Fix 4 (C2) changed the transaction_done_cb
        wiring to DOWNLOAD_COMPLETE_CB for subprocess exit gating. GC is now
        triggered explicitly by the aggregator in _end_gather() (M3).
        """
        import nvflare.fuel.utils.fobs.decomposers.via_downloader as vd

        assert not hasattr(vd, "_on_tx_done"), (
            "_on_tx_done must not exist; it was removed when GC via transaction_done_cb "
            "was replaced by explicit aggregator GC (M3) and subprocess exit gating (Fix 16)."
        )
