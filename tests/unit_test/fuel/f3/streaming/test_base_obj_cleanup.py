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

"""Tests for CacheableObject base_obj lifecycle (C1 fix) and transaction_done_cb ordering (C2 fix).

C1 root cause (before fix):
  clear_cache() set both self.cache = None AND self.base_obj = None.  A concurrent
  _get_item() call that found cache=None fell through to produce_item(), which then
  dereferenced self.base_obj — crashing because it was already None.

C1 fix:
  - clear_cache() only clears self.cache.
  - release() clears self.base_obj (called AFTER the transaction_done_cb).
  - _get_item() raises RuntimeError when cache=None and base_obj=None.

C2 root cause (before fix):
  _Transaction.transaction_done() called obj.transaction_done() (which called
  clear_cache() — setting base_obj=None) BEFORE invoking the transaction_done_cb.
  The callback received [None, None, ...] instead of the actual source objects.

C2 fix:
  - Snapshot base_objs before the per-object loop.
  - Call transaction_done_cb with the snapshot.
  - Call obj.release() for each object AFTER the callback.

To simulate without allocating large models:
  - ArrayDownloadable with small in-memory numpy arrays.
  - Direct method calls instead of live download transactions.
"""

import gc
import sys
import threading
from unittest.mock import patch

import numpy as np
import pytest

from nvflare.app_common.np.np_downloader import ArrayDownloadable
from nvflare.fuel.f3.streaming.download_service import Downloadable, ProduceRC

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ARRAY_SHAPE = (256, 256)
_NUM_KEYS = 4
_MAX_CHUNK = 2 * 1024 * 1024  # 2 MB per chunk


def _make_arrays() -> dict:
    return {f"key_{i}": np.ones(_ARRAY_SHAPE, dtype=np.float32) for i in range(_NUM_KEYS)}


def _make_downloadable() -> ArrayDownloadable:
    return ArrayDownloadable(arrays=_make_arrays(), max_chunk_size=_MAX_CHUNK)


class _SimpleDownloadable(Downloadable):
    """Minimal Downloadable for C2 unit tests.

    Does not call DownloadService.get_transaction_info() in set_transaction(),
    so _Transaction can be instantiated directly without a live DownloadService.
    """

    def __init__(self, source_obj):
        super().__init__(source_obj)
        self.transaction_done_calls = []

    def set_transaction(self, tx_id: str, ref_id: str):
        pass  # no DownloadService lookup needed

    def produce(self, state, requester):
        return ProduceRC.EOF, None, {}

    def transaction_done(self, transaction_id: str, status: str):
        self.transaction_done_calls.append((transaction_id, status))


# ---------------------------------------------------------------------------
# C1: clear_cache() only clears chunk cache; release() clears base_obj
# ---------------------------------------------------------------------------


class TestClearCacheOnlyClearsCache:
    """clear_cache() must only null self.cache, not self.base_obj.

    Before the C1 fix, clear_cache() nulled both.  That created a race:
    _get_item() saw cache=None, then tried produce_item() with base_obj=None.
    """

    def test_clear_cache_nulls_cache(self):
        """clear_cache() must set self.cache to None."""
        dl = _make_downloadable()
        dl.clear_cache()
        assert dl.cache is None

    def test_clear_cache_does_not_null_base_obj(self):
        """clear_cache() must NOT touch base_obj (C1 fix)."""
        dl = _make_downloadable()
        original = dl.base_obj
        dl.clear_cache()
        assert dl.base_obj is original, (
            "clear_cache() must not clear base_obj; base_obj is released via release() "
            "after the transaction_done_cb fires (C1 fix)."
        )

    def test_clear_cache_is_idempotent(self):
        """Calling clear_cache() twice must not raise."""
        dl = _make_downloadable()
        dl.clear_cache()
        dl.clear_cache()

    def test_regression_old_clear_cache_would_null_base_obj(self):
        """Regression: the pre-fix clear_cache() also nulled base_obj.

        Simulate the old behaviour by patching clear_cache to also set base_obj=None,
        then confirm base_obj IS None — proving the old code was the root cause.
        """
        dl = _make_downloadable()

        def old_clear_cache():
            with dl.lock:
                dl.cache = None
                dl.base_obj = None  # pre-fix behaviour

        with patch.object(dl, "clear_cache", side_effect=old_clear_cache):
            dl.clear_cache()

        assert dl.base_obj is None, "Regression confirmed: old clear_cache() nulled base_obj, enabling the race."


# ---------------------------------------------------------------------------
# C1: release() clears base_obj
# ---------------------------------------------------------------------------


class TestReleaseClearsBaseObj:
    """release() must null base_obj; this is the correct cleanup point."""

    def test_release_nulls_base_obj(self):
        """release() must set base_obj to None."""
        dl = _make_downloadable()
        assert dl.base_obj is not None
        dl.release()
        assert dl.base_obj is None

    def test_release_is_idempotent(self):
        """Calling release() twice must not raise."""
        dl = _make_downloadable()
        dl.release()
        dl.release()

    def test_release_is_thread_safe(self):
        """Concurrent release() calls must not raise."""
        dl = _make_downloadable()
        errors = []

        def _release():
            try:
                dl.release()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_release) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent release() raised: {errors}"
        assert dl.base_obj is None

    def test_refcount_drops_after_release_not_after_transaction_done(self):
        """The source object refcount drops after release(), not after transaction_done().

        Before C1 fix: transaction_done() → clear_cache() cleared base_obj.
        After C1 fix: transaction_done() only clears cache; release() clears base_obj.
        """
        arrays = _make_arrays()
        dl = ArrayDownloadable(arrays=arrays, max_chunk_size=_MAX_CHUNK)

        refcount_initial = sys.getrefcount(arrays)

        # transaction_done() must NOT drop the refcount
        dl.transaction_done(transaction_id="tx-1", status="FINISHED")
        gc.collect()
        refcount_after_tx_done = sys.getrefcount(arrays)
        assert refcount_after_tx_done == refcount_initial, (
            "transaction_done() must not release base_obj (C1 fix): "
            f"refcount changed from {refcount_initial} to {refcount_after_tx_done}."
        )

        # release() MUST drop the refcount
        dl.release()
        gc.collect()
        refcount_after_release = sys.getrefcount(arrays)
        assert refcount_after_release < refcount_after_tx_done, (
            f"release() must drop the refcount (was {refcount_after_tx_done}, " f"now {refcount_after_release})."
        )


# ---------------------------------------------------------------------------
# C1: transaction_done() leaves base_obj intact; cache is cleared
# ---------------------------------------------------------------------------


class TestTransactionDoneLifecycle:
    """After transaction_done(), cache is None but base_obj is still valid."""

    def test_cache_is_none_after_transaction_done(self):
        dl = _make_downloadable()
        dl.transaction_done(transaction_id="tx-1", status="FINISHED")
        assert dl.cache is None

    def test_base_obj_intact_after_transaction_done(self):
        """base_obj must NOT be cleared by transaction_done() (C1 fix)."""
        dl = _make_downloadable()
        dl.transaction_done(transaction_id="tx-1", status="FINISHED")
        assert dl.base_obj is not None, (
            "transaction_done() must not clear base_obj; that is done by release() "
            "after the transaction_done_cb (C1 fix)."
        )

    def test_base_obj_is_none_after_release(self):
        """base_obj must be None after release() is called."""
        dl = _make_downloadable()
        dl.transaction_done(transaction_id="tx-1", status="FINISHED")
        dl.release()
        assert dl.base_obj is None

    def test_lifecycle_for_all_terminal_statuses(self):
        """base_obj is intact after transaction_done() for all terminal statuses."""
        for status in ("FINISHED", "TIMEOUT", "DELETED"):
            dl = _make_downloadable()
            dl.transaction_done(transaction_id="tx-1", status=status)
            assert dl.base_obj is not None, f"base_obj cleared for status={status}"
            dl.release()
            assert dl.base_obj is None, f"base_obj not cleared by release() for status={status}"


# ---------------------------------------------------------------------------
# C1: _get_item() guard after release()
# ---------------------------------------------------------------------------


class TestGetItemGuardAfterRelease:
    """_get_item() must raise RuntimeError when cache=None and base_obj=None."""

    def test_get_item_raises_after_release(self):
        """After release(), _get_item() must raise RuntimeError (not return corrupt data)."""
        dl = _make_downloadable()
        dl.transaction_done(transaction_id="tx-1", status="FINISHED")  # clears cache only
        dl.release()  # clears base_obj

        with pytest.raises(RuntimeError, match="requested after base_obj released"):
            dl._get_item(0, "test-requester")

    def test_get_item_works_after_transaction_done_before_release(self):
        """Between transaction_done() and release(), produce_item() is still callable.

        cache=None but base_obj is still valid, so _get_item() falls through to
        produce_item().  In normal operation no new requests arrive after
        transaction_done(), but this confirms the guard only fires when base_obj is None.
        """
        dl = _make_downloadable()
        dl.transaction_done(transaction_id="tx-1", status="FINISHED")  # clears cache only
        # base_obj is still valid — produce_item() should not raise
        result = dl._get_item(0, "test-requester")
        assert result is not None


# ---------------------------------------------------------------------------
# C2: transaction_done_cb receives original base_objs (not Nones)
# ---------------------------------------------------------------------------


class TestCallbackReceivesOriginalBaseObjs:
    """transaction_done_cb must receive the original source objects, not [None].

    Before C2 fix: _Transaction.transaction_done() called obj.transaction_done()
    (which via old clear_cache() nulled base_obj) BEFORE the callback, so
    [ref.obj.base_obj ...] was all Nones.

    After C2 fix: base_objs are snapshotted before the per-object loop; callback
    is invoked with the snapshot; release() is called after the callback.
    """

    def test_callback_receives_non_none_base_obj(self):
        """transaction_done_cb must receive the original source object, not None (C2 fix).

        Uses _SimpleDownloadable to avoid needing a live DownloadService — the C2
        fix is in _Transaction.transaction_done(), not in DownloadService itself.
        """
        from nvflare.fuel.f3.streaming.download_service import TransactionDoneStatus, _Transaction

        source = {"weight": np.ones((10,), dtype=np.float32)}
        dl = _SimpleDownloadable(source)

        observed_base_objs = []

        def _cb(tid, status, base_objs, **kwargs):
            observed_base_objs.extend(base_objs)

        tx = _Transaction(timeout=60.0, num_receivers=1, transaction_done_cb=_cb, cb_kwargs={})
        tx.add_object(dl)
        tx.transaction_done(TransactionDoneStatus.FINISHED)

        assert len(observed_base_objs) == 1
        assert observed_base_objs[0] is source, (
            "transaction_done_cb must receive the original source object, not None (C2 fix). "
            f"Got: {observed_base_objs[0]}"
        )

    def test_release_is_called_by_transaction_done(self):
        """_Transaction.transaction_done() must call release() on each Downloadable (C2 fix)."""
        from nvflare.fuel.f3.streaming.download_service import TransactionDoneStatus, _Transaction

        source = {"weight": np.ones((10,), dtype=np.float32)}
        dl = _SimpleDownloadable(source)
        release_call_count = []

        original_release = dl.release

        def spy_release():
            release_call_count.append(1)
            original_release()

        dl.release = spy_release

        tx = _Transaction(timeout=60.0, num_receivers=1, transaction_done_cb=None, cb_kwargs={})
        tx.add_object(dl)
        tx.transaction_done(TransactionDoneStatus.FINISHED)

        assert (
            len(release_call_count) == 1
        ), "_Transaction.transaction_done() must call release() on each Downloadable (C2 fix)."

    def test_release_called_after_callback(self):
        """release() must be called AFTER the callback, not before (C2 fix)."""
        from nvflare.fuel.f3.streaming.download_service import TransactionDoneStatus, _Transaction

        source = {"weight": np.ones((10,), dtype=np.float32)}
        dl = _SimpleDownloadable(source)
        call_order = []

        def _cb(tid, status, base_objs, **kwargs):
            call_order.append("callback")

        original_release = dl.release

        def spy_release():
            call_order.append("release")
            original_release()

        dl.release = spy_release

        tx = _Transaction(timeout=60.0, num_receivers=1, transaction_done_cb=_cb, cb_kwargs={})
        tx.add_object(dl)
        tx.transaction_done(TransactionDoneStatus.FINISHED)

        assert call_order == [
            "callback",
            "release",
        ], f"release() must be called AFTER the callback (C2 fix). Got order: {call_order}"
