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

"""
Unit tests for the B1 pass-through architecture in ViaDownloaderDecomposer.

Tests verify:
  1. LazyDownloadRef and _LazyBatchInfo construction.
  2. process_datum() in PASS_THROUGH mode stores _LazyBatchInfo sentinel
     and does NOT call the network download path.
  3. recompose() returns LazyDownloadRef when _LazyBatchInfo is in context.
  4. decompose() with a LazyDownloadRef re-emits the original server datum
     via a post-callback (_finalize_lazy_batch) — only one post-CB registered
     regardless of how many items are in the batch.
  5. No download transactions are created at the CJ (no memory accumulation):
     _CtxKey.OBJECTS must be absent from fobs_ctx after a PASS_THROUGH round.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from nvflare.fuel.utils.fobs import FOBSContextKey
from nvflare.fuel.utils.fobs.datum import Datum, DatumManager, DatumType
from nvflare.fuel.utils.fobs.decomposers.via_downloader import (
    _LAZY_BATCH_CTX_SUFFIX,
    _CtxKey,
    _RefKey,
    EncKey,
    EncType,
    LazyDownloadRef,
    _LazyBatchInfo,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SERVER_FQCN = "server/app_server"
_REF_ID = "deadbeef-0000-1111-2222-333333333333"
_ITEM_ID_0 = "T0"
_ITEM_ID_1 = "T1"


def _make_decomposer():
    """Return a NumpyArrayDecomposer (concrete ViaDownloaderDecomposer subclass)."""
    from nvflare.app_common.decomposers.numpy_decomposers import NumpyArrayDecomposer

    return NumpyArrayDecomposer()


def _make_manager(fobs_ctx: dict = None) -> DatumManager:
    """Return a DatumManager with an optional pre-populated fobs_ctx."""
    ctx = fobs_ctx if fobs_ctx is not None else {}
    return DatumManager(threshold=1024, fobs_ctx=ctx)


def _ref_datum(fqcn: str = _SERVER_FQCN, ref_id: str = _REF_ID) -> Datum:
    """Build the kind of Datum that ViaDownloaderDecomposer emits on the sender side."""
    from nvflare.app_common.decomposers.numpy_decomposers import NumpyArrayDecomposer

    decomposer = NumpyArrayDecomposer()
    ref = {_RefKey.FQCN: fqcn, _RefKey.REF_ID: ref_id}
    return Datum(datum_type=DatumType.TEXT, value=json.dumps(ref), dot=decomposer.get_download_dot())


# ---------------------------------------------------------------------------
# 1. LazyDownloadRef
# ---------------------------------------------------------------------------


class TestLazyDownloadRef:
    def test_construction_stores_all_attributes(self):
        lazy = LazyDownloadRef(fqcn=_SERVER_FQCN, ref_id=_REF_ID, item_id=_ITEM_ID_0)
        assert lazy.fqcn == _SERVER_FQCN
        assert lazy.ref_id == _REF_ID
        assert lazy.item_id == _ITEM_ID_0

    def test_slots_prevent_arbitrary_attributes(self):
        lazy = LazyDownloadRef(fqcn=_SERVER_FQCN, ref_id=_REF_ID, item_id=_ITEM_ID_0)
        with pytest.raises(AttributeError):
            lazy.unexpected_field = "oops"

    def test_different_item_ids_for_different_instances(self):
        a = LazyDownloadRef(fqcn=_SERVER_FQCN, ref_id=_REF_ID, item_id=_ITEM_ID_0)
        b = LazyDownloadRef(fqcn=_SERVER_FQCN, ref_id=_REF_ID, item_id=_ITEM_ID_1)
        assert a.item_id != b.item_id
        # fqcn and ref_id are shared (same batch)
        assert a.fqcn == b.fqcn
        assert a.ref_id == b.ref_id


# ---------------------------------------------------------------------------
# 2. _LazyBatchInfo
# ---------------------------------------------------------------------------


class TestLazyBatchInfo:
    def test_construction_stores_fqcn_and_ref_id(self):
        info = _LazyBatchInfo(fqcn=_SERVER_FQCN, ref_id=_REF_ID)
        assert info.fqcn == _SERVER_FQCN
        assert info.ref_id == _REF_ID

    def test_slots_prevent_arbitrary_attributes(self):
        info = _LazyBatchInfo(fqcn=_SERVER_FQCN, ref_id=_REF_ID)
        with pytest.raises(AttributeError):
            info.unexpected_field = "oops"

    def test_isinstance_check_is_reliable(self):
        """_LazyBatchInfo is unambiguous — a plain tuple would not pass."""
        info = _LazyBatchInfo(fqcn=_SERVER_FQCN, ref_id=_REF_ID)
        assert isinstance(info, _LazyBatchInfo)
        assert not isinstance((_SERVER_FQCN, _REF_ID), _LazyBatchInfo)


# ---------------------------------------------------------------------------
# 3. process_datum() in PASS_THROUGH mode
# ---------------------------------------------------------------------------


class TestProcessDatumPassThrough:
    def test_stores_lazy_batch_info_in_context(self):
        """PASS_THROUGH mode should store _LazyBatchInfo in fobs_ctx, not download."""
        decomposer = _make_decomposer()
        fobs_ctx = {FOBSContextKey.PASS_THROUGH: True}
        mgr = _make_manager(fobs_ctx)
        datum = _ref_datum()

        # Should NOT raise and should NOT attempt any network call
        with patch.object(decomposer, "_download_from_remote_cell") as mock_dl:
            decomposer.process_datum(datum, mgr)
            mock_dl.assert_not_called()

        items = fobs_ctx.get(decomposer.items_key)
        assert isinstance(items, _LazyBatchInfo), (
            f"Expected _LazyBatchInfo in fobs_ctx[{decomposer.items_key!r}], got {type(items)}"
        )
        assert items.fqcn == _SERVER_FQCN
        assert items.ref_id == _REF_ID

    def test_does_not_store_raw_tuple(self):
        """items_key must never hold a plain tuple — that was the fragile old design."""
        decomposer = _make_decomposer()
        fobs_ctx = {FOBSContextKey.PASS_THROUGH: True}
        mgr = _make_manager(fobs_ctx)
        datum = _ref_datum()

        with patch.object(decomposer, "_download_from_remote_cell"):
            decomposer.process_datum(datum, mgr)

        items = fobs_ctx.get(decomposer.items_key)
        assert not isinstance(items, tuple), "items_key must not hold a raw tuple (fragile legacy design)"

    def test_normal_mode_calls_download(self):
        """Without PASS_THROUGH, process_datum() must attempt a real download."""
        decomposer = _make_decomposer()
        # Note: DatumManager replaces a falsy (empty) fobs_ctx with a fresh dict,
        # so we must read items from mgr.fobs_ctx, not the original local variable.
        mgr = _make_manager()  # no PASS_THROUGH flag
        datum = _ref_datum()

        # _download_from_remote_cell requires a Cell; patch it to avoid network
        fake_items = {"T0": MagicMock()}
        with patch.object(decomposer, "_download_from_remote_cell", return_value=fake_items) as mock_dl:
            decomposer.process_datum(datum, mgr)
            mock_dl.assert_called_once()

        assert mgr.fobs_ctx.get(decomposer.items_key) == fake_items


# ---------------------------------------------------------------------------
# 4. recompose() with _LazyBatchInfo sentinel
# ---------------------------------------------------------------------------


class TestRecomposePassThrough:
    def _ctx_with_lazy_batch(self):
        decomposer = _make_decomposer()
        fobs_ctx = {
            decomposer.items_key: _LazyBatchInfo(fqcn=_SERVER_FQCN, ref_id=_REF_ID),
        }
        return decomposer, fobs_ctx

    def test_returns_lazy_download_ref(self):
        decomposer, fobs_ctx = self._ctx_with_lazy_batch()
        mgr = _make_manager(fobs_ctx)

        result = decomposer.recompose({EncKey.TYPE: EncType.REF, EncKey.DATA: _ITEM_ID_0}, mgr)

        assert isinstance(result, LazyDownloadRef), f"Expected LazyDownloadRef, got {type(result)}"

    def test_lazy_ref_carries_correct_fqcn_and_ref_id(self):
        decomposer, fobs_ctx = self._ctx_with_lazy_batch()
        mgr = _make_manager(fobs_ctx)

        result = decomposer.recompose({EncKey.TYPE: EncType.REF, EncKey.DATA: _ITEM_ID_0}, mgr)

        assert result.fqcn == _SERVER_FQCN
        assert result.ref_id == _REF_ID

    def test_lazy_ref_carries_correct_item_id(self):
        decomposer, fobs_ctx = self._ctx_with_lazy_batch()
        mgr = _make_manager(fobs_ctx)

        result = decomposer.recompose({EncKey.TYPE: EncType.REF, EncKey.DATA: _ITEM_ID_1}, mgr)

        assert result.item_id == _ITEM_ID_1

    def test_multiple_items_produce_distinct_lazy_refs(self):
        """Each call to recompose() yields a distinct LazyDownloadRef with its own item_id."""
        decomposer, fobs_ctx = self._ctx_with_lazy_batch()
        mgr = _make_manager(fobs_ctx)

        r0 = decomposer.recompose({EncKey.TYPE: EncType.REF, EncKey.DATA: _ITEM_ID_0}, mgr)
        r1 = decomposer.recompose({EncKey.TYPE: EncType.REF, EncKey.DATA: _ITEM_ID_1}, mgr)

        assert r0.item_id == _ITEM_ID_0
        assert r1.item_id == _ITEM_ID_1
        # Both must point at the same server batch
        assert r0.fqcn == r1.fqcn
        assert r0.ref_id == r1.ref_id


# ---------------------------------------------------------------------------
# 5. decompose() with LazyDownloadRef input
# ---------------------------------------------------------------------------


class TestDecomposeWithLazyDownloadRef:
    def test_returns_ref_enc_with_correct_item_id(self):
        """decompose() on a LazyDownloadRef must return {type: ref, data: item_id}."""
        decomposer = _make_decomposer()
        mgr = _make_manager()
        lazy = LazyDownloadRef(fqcn=_SERVER_FQCN, ref_id=_REF_ID, item_id=_ITEM_ID_0)

        result = decomposer.decompose(lazy, mgr)

        assert result[EncKey.TYPE] == EncType.REF
        assert result[EncKey.DATA] == _ITEM_ID_0

    def test_post_cb_registered_on_first_item(self):
        """First LazyDownloadRef in a batch must register _finalize_lazy_batch post-CB."""
        decomposer = _make_decomposer()
        mgr = _make_manager()
        lazy = LazyDownloadRef(fqcn=_SERVER_FQCN, ref_id=_REF_ID, item_id=_ITEM_ID_0)

        decomposer.decompose(lazy, mgr)

        # There should be exactly one post-CB registered
        assert len(mgr.post_cbs) == 1
        cb, _ = mgr.post_cbs[0]
        assert cb.__name__ == "_finalize_lazy_batch"

    def test_post_cb_registered_only_once_for_whole_batch(self):
        """Multiple LazyDownloadRefs from the same batch share one post-CB."""
        decomposer = _make_decomposer()
        mgr = _make_manager()
        lazy0 = LazyDownloadRef(fqcn=_SERVER_FQCN, ref_id=_REF_ID, item_id=_ITEM_ID_0)
        lazy1 = LazyDownloadRef(fqcn=_SERVER_FQCN, ref_id=_REF_ID, item_id=_ITEM_ID_1)

        decomposer.decompose(lazy0, mgr)
        decomposer.decompose(lazy1, mgr)

        # Still only ONE _finalize_lazy_batch CB, regardless of number of items
        finalize_cbs = [cb for cb, _ in mgr.post_cbs if cb.__name__ == "_finalize_lazy_batch"]
        assert len(finalize_cbs) == 1, (
            f"Expected exactly 1 _finalize_lazy_batch CB, got {len(finalize_cbs)}"
        )

    def test_finalize_lazy_batch_adds_exactly_one_datum(self):
        """_finalize_lazy_batch post-CB must add exactly one datum per batch."""
        decomposer = _make_decomposer()
        mgr = _make_manager()
        lazy0 = LazyDownloadRef(fqcn=_SERVER_FQCN, ref_id=_REF_ID, item_id=_ITEM_ID_0)
        lazy1 = LazyDownloadRef(fqcn=_SERVER_FQCN, ref_id=_REF_ID, item_id=_ITEM_ID_1)

        decomposer.decompose(lazy0, mgr)
        decomposer.decompose(lazy1, mgr)

        mgr.post_process()

        datums = list(mgr.get_datums().values())
        assert len(datums) == 1, f"Expected 1 datum from lazy batch, got {len(datums)}"

    def test_finalize_lazy_batch_datum_has_correct_fqcn_and_ref_id(self):
        """The datum emitted by _finalize_lazy_batch must preserve original server fqcn/ref_id."""
        decomposer = _make_decomposer()
        mgr = _make_manager()
        lazy = LazyDownloadRef(fqcn=_SERVER_FQCN, ref_id=_REF_ID, item_id=_ITEM_ID_0)

        decomposer.decompose(lazy, mgr)
        mgr.post_process()

        datums = list(mgr.get_datums().values())
        assert len(datums) == 1
        ref = json.loads(datums[0].value)
        assert ref[_RefKey.FQCN] == _SERVER_FQCN
        assert ref[_RefKey.REF_ID] == _REF_ID

    def test_finalize_lazy_batch_datum_dot_matches_decomposer(self):
        """The datum DOT must equal the decomposer's download DOT."""
        decomposer = _make_decomposer()
        mgr = _make_manager()
        lazy = LazyDownloadRef(fqcn=_SERVER_FQCN, ref_id=_REF_ID, item_id=_ITEM_ID_0)

        decomposer.decompose(lazy, mgr)
        mgr.post_process()

        datums = list(mgr.get_datums().values())
        assert datums[0].dot == decomposer.get_download_dot()


# ---------------------------------------------------------------------------
# 6. No download transactions created at CJ (no memory accumulation)
# ---------------------------------------------------------------------------


class TestNoMemoryAccumulation:
    def test_no_objects_entry_in_fobs_ctx_after_pass_through(self):
        """PASS_THROUGH processing must not populate _CtxKey.OBJECTS in fobs_ctx.

        _CtxKey.OBJECTS is populated only when a real download transaction is
        created (in _create_datum / _finalize_download_tx). In PASS_THROUGH mode
        the decomposer re-emits the original server datum instead; no new
        transaction is opened and _CtxKey.OBJECTS must remain absent.
        """
        decomposer = _make_decomposer()
        # --- Simulate: server sent a batch datum; CJ receives it in PASS_THROUGH mode ---
        fobs_ctx = {FOBSContextKey.PASS_THROUGH: True}
        rx_mgr = _make_manager(fobs_ctx)

        with patch.object(decomposer, "_download_from_remote_cell"):
            decomposer.process_datum(_ref_datum(), rx_mgr)

        # --- Simulate: CJ re-serializes the LazyDownloadRef for the subprocess ---
        lazy = decomposer.recompose({EncKey.TYPE: EncType.REF, EncKey.DATA: _ITEM_ID_0}, rx_mgr)
        assert isinstance(lazy, LazyDownloadRef)

        tx_mgr = _make_manager()
        decomposer.decompose(lazy, tx_mgr)
        tx_mgr.post_process()

        # _CtxKey.OBJECTS must NOT be present (no download transaction opened at CJ)
        assert _CtxKey.OBJECTS not in tx_mgr.fobs_ctx, (
            "_CtxKey.OBJECTS must be absent in PASS_THROUGH mode — no download transaction "
            "should be created at the CJ."
        )

    def test_no_download_service_transaction_created_at_cj(self):
        """PASS_THROUGH decompose must not register any DownloadService transaction.

        A normal decompose() would call _create_downloader() -> ObjectDownloader ->
        DownloadService.new_transaction().  In PASS_THROUGH mode this path is
        entirely bypassed; the DownloadService transaction table must be unchanged.
        """
        from nvflare.fuel.f3.streaming.download_service import DownloadService

        before = set(DownloadService._tx_table.keys())

        decomposer = _make_decomposer()
        mgr = _make_manager()
        lazy = LazyDownloadRef(fqcn=_SERVER_FQCN, ref_id=_REF_ID, item_id=_ITEM_ID_0)
        decomposer.decompose(lazy, mgr)
        mgr.post_process()

        after = set(DownloadService._tx_table.keys())
        new_transactions = after - before
        assert not new_transactions, (
            f"PASS_THROUGH decompose must not create DownloadService transactions, "
            f"but {len(new_transactions)} new transaction(s) were found."
        )

    def test_repeated_pass_through_cycles_do_not_grow_fobs_ctx(self):
        """Running many PASS_THROUGH decode+encode cycles must not accumulate state.

        Each round uses a fresh DatumManager (as happens per-message in the real
        runtime), so no state should bleed between rounds.
        """
        decomposer = _make_decomposer()

        for i in range(50):
            # Step 1: receive (PASS_THROUGH decode)
            fobs_ctx = {FOBSContextKey.PASS_THROUGH: True}
            rx_mgr = _make_manager(fobs_ctx)
            with patch.object(decomposer, "_download_from_remote_cell"):
                decomposer.process_datum(_ref_datum(ref_id=f"ref-{i}"), rx_mgr)

            lazy = decomposer.recompose({EncKey.TYPE: EncType.REF, EncKey.DATA: f"T{i}"}, rx_mgr)
            assert isinstance(lazy, LazyDownloadRef)

            # Step 2: re-emit (encode for subprocess)
            tx_mgr = _make_manager()
            decomposer.decompose(lazy, tx_mgr)
            tx_mgr.post_process()

            # Each cycle produces exactly one datum and no OBJECTS entry
            assert len(tx_mgr.get_datums()) == 1, f"Round {i}: expected 1 datum"
            assert _CtxKey.OBJECTS not in tx_mgr.fobs_ctx, f"Round {i}: OBJECTS must not be set"
