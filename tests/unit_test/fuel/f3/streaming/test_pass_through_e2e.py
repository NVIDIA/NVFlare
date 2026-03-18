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
End-to-end tests for the B1 pass-through architecture using real Cell objects.

Validates the full data path exercised by ClientAPILauncherExecutor when an
external subprocess agent is in use:

    FL Server ──serialize──▶ CJ (PASS_THROUGH, no download) ──re-emit──▶ Subprocess
                              └── only LazyDownloadRef placeholders         └── downloads directly
                                  land in CJ memory                             from FL Server

Two real cells are created:
  • server  — acts as the FL server; owns the download transactions.
  • subprocess — acts as the subprocess agent; downloads directly from server.

"CJ" is simulated by calling load_from_bytes with PASS_THROUGH=True (no live
cell required for the CJ hop) and then dump_to_bytes on the resulting
LazyDownloadRef objects (also requires no cell, as the lazy branch re-emits
the original server datum verbatim).

Tests:
  1. test_arrays_survive_pass_through_hop
     Full round-trip correctness: arrays serialised on server arrive intact
     at subprocess after passing through CJ in PASS_THROUGH mode.

  2. test_cj_holds_only_lazy_refs_not_tensor_data
     Verifies that CJ deserialization produces LazyDownloadRef objects, NOT
     actual numpy arrays — confirming no tensor data is copied into CJ memory.

  3. test_cj_creates_no_download_transaction
     DownloadService transaction table must not grow during the CJ hop
     (PASS_THROUGH deserialization + LazyDownloadRef re-serialization).

  4. test_forwarded_payload_carries_original_server_ref
     The datum emitted by CJ's re-serialization must contain the same fqcn
     and ref_id as the server's original datum — guaranteeing the subprocess
     downloads from the correct source.

  5. test_multiple_array_roundtrip
     A larger batch of arrays (simulating a realistic model state-dict) all
     survive the pass-through hop with bit-exact values.
"""

import json
import time
import uuid

import numpy as np
import pytest

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.utils.fobs import FOBSContextKey
from nvflare.fuel.utils.fobs.decomposers.via_downloader import LazyDownloadRef, _RefKey
from nvflare.fuel.utils.fobs.lobs import dump_to_bytes, load_from_bytes
from nvflare.fuel.utils.network_utils import get_open_ports

CONNECT_WAIT = 2.0  # seconds to wait for TCP cell connection


def _register_numpy():
    """Register numpy decomposers (idempotent)."""
    from nvflare.app_common.decomposers import numpy_decomposers

    numpy_decomposers.register()


def _small_model():
    """Return a small model state-dict (numpy) for fast testing."""
    rng = np.random.default_rng(42)
    return {
        "fc1.weight": rng.standard_normal((32, 64)).astype(np.float32),
        "fc1.bias": rng.standard_normal(32).astype(np.float32),
        "fc2.weight": rng.standard_normal((10, 32)).astype(np.float32),
        "fc2.bias": rng.standard_normal(10).astype(np.float32),
    }


def _simulate_cj_pass_through(server_bytes: bytes) -> bytes:
    """Simulate the CJ PASS_THROUGH hop.

    Deserialises *server_bytes* with PASS_THROUGH=True (no cell needed) to
    obtain LazyDownloadRef placeholders, then immediately re-serialises them.
    Returns the forwarded bytes that the CJ would pipe to the subprocess.
    """
    cj_result = load_from_bytes(server_bytes, fobs_ctx={FOBSContextKey.PASS_THROUGH: True})
    forwarded_bytes = dump_to_bytes(cj_result)
    return cj_result, forwarded_bytes


@pytest.mark.timeout(60)
class TestPassThroughE2E:
    """End-to-end B1 pass-through tests with real Cells."""

    @pytest.fixture(scope="class", autouse=True)
    def register_decomposers(self):
        _register_numpy()

    @pytest.fixture(scope="class")
    def cells(self):
        port = get_open_ports(1)[0]
        server_fqcn = f"server-{uuid.uuid4().hex[:8]}"
        subproc_fqcn = f"subprocess-{uuid.uuid4().hex[:8]}"
        server = Cell(server_fqcn, f"tcp://localhost:{port}", secure=False, credentials={})
        server.core_cell.start()
        subproc = Cell(subproc_fqcn, f"tcp://localhost:{port}", secure=False, credentials={})
        subproc.core_cell.start()
        time.sleep(CONNECT_WAIT)
        try:
            yield server, subproc
        finally:
            subproc_name = subproc.get_fqcn()
            server_name = server.get_fqcn()
            subproc.core_cell.stop()
            server.core_cell.stop()
            CoreCell.ALL_CELLS.pop(subproc_name, None)
            CoreCell.ALL_CELLS.pop(server_name, None)

    # ------------------------------------------------------------------
    # Test 1 — full round-trip correctness
    # ------------------------------------------------------------------

    def test_arrays_survive_pass_through_hop(self, cells):
        """Arrays serialised on server arrive intact at subprocess after CJ PASS_THROUGH hop."""
        server, subproc = cells
        original = _small_model()

        # Server serializes (creates ObjectDownloader transaction)
        server_bytes = dump_to_bytes(original, fobs_ctx={FOBSContextKey.CELL: server})

        # CJ: PASS_THROUGH deserialization + re-serialization
        _, forwarded_bytes = _simulate_cj_pass_through(server_bytes)

        # Subprocess downloads directly from server
        result = load_from_bytes(forwarded_bytes, fobs_ctx={FOBSContextKey.CELL: subproc})

        assert isinstance(result, dict), f"Expected dict from subprocess, got {type(result)}"
        assert set(result.keys()) == set(
            original.keys()
        ), f"Key mismatch: expected {set(original.keys())}, got {set(result.keys())}"
        for key in original:
            np.testing.assert_array_almost_equal(
                result[key],
                original[key],
                err_msg=f"Array mismatch for key '{key}'",
            )

    # ------------------------------------------------------------------
    # Test 2 — CJ must never hold actual tensor data
    # ------------------------------------------------------------------

    def test_cj_holds_only_lazy_refs_not_tensor_data(self, cells):
        """CJ deserialization in PASS_THROUGH mode must yield LazyDownloadRef, not numpy arrays.

        This is the critical invariant of the B1 architecture: tensor data must
        never materialise at the CJ process.  Only lightweight placeholders
        (LazyDownloadRef) should be present after CJ deserialization.
        """
        server, _ = cells
        original = {"layer": np.ones((128, 128), dtype=np.float32)}  # ~64 KB

        server_bytes = dump_to_bytes(original, fobs_ctx={FOBSContextKey.CELL: server})

        # CJ deserialization — no cell provided, PASS_THROUGH=True
        cj_result = load_from_bytes(server_bytes, fobs_ctx={FOBSContextKey.PASS_THROUGH: True})

        assert isinstance(cj_result, dict)
        for key, val in cj_result.items():
            assert isinstance(val, LazyDownloadRef), (
                f"CJ got {type(val).__name__} for '{key}' instead of LazyDownloadRef. "
                "B1 pass-through requires that no tensor data is materialised at CJ."
            )
            assert not isinstance(
                val, np.ndarray
            ), f"Key '{key}': numpy array must NOT be present at CJ in PASS_THROUGH mode."

    # ------------------------------------------------------------------
    # Test 3 — no download transaction created at CJ
    # ------------------------------------------------------------------

    def test_cj_creates_no_download_transaction(self, cells):
        """PASS_THROUGH hop must not open any new DownloadService transactions.

        Normal serialization (server) creates exactly one transaction per batch.
        The CJ PASS_THROUGH deserialization and LazyDownloadRef re-serialization
        must leave the DownloadService transaction table unchanged.
        """
        from nvflare.fuel.f3.streaming.download_service import DownloadService

        server, _ = cells
        original = {"param": np.eye(16, dtype=np.float32)}

        # Server serializes — one transaction is created
        server_bytes = dump_to_bytes(original, fobs_ctx={FOBSContextKey.CELL: server})
        tx_snapshot = set(DownloadService._tx_table.keys())

        # CJ PASS_THROUGH + re-serialize
        cj_result, forwarded_bytes = _simulate_cj_pass_through(server_bytes)

        tx_after_cj = set(DownloadService._tx_table.keys())
        new_transactions = tx_after_cj - tx_snapshot

        assert not new_transactions, (
            f"CJ created {len(new_transactions)} unexpected DownloadService transaction(s). "
            "PASS_THROUGH mode must not open any download transactions at the CJ."
        )

    # ------------------------------------------------------------------
    # Test 4 — forwarded payload carries original server ref
    # ------------------------------------------------------------------

    def test_forwarded_payload_carries_original_server_ref(self, cells):
        """The datum emitted by CJ re-serialization must preserve the server's fqcn and ref_id.

        The subprocess must be able to download from the FL server directly.
        If the CJ's forwarded datum contained the CJ's own fqcn/ref_id the
        subprocess would attempt to download from CJ — breaking the architecture.
        """
        from io import BytesIO

        from nvflare.fuel.utils.fobs.lobs import HEADER_LEN, MARKER_DATUM_TEXT, _Header

        server, _ = cells
        original = {"w": np.zeros((8, 8), dtype=np.float32)}

        server_bytes = dump_to_bytes(original, fobs_ctx={FOBSContextKey.CELL: server})
        _, forwarded_bytes = _simulate_cj_pass_through(server_bytes)

        # Extract the datum(s) from forwarded_bytes and verify the ref content.
        # lobs format: [main_header + main_body] [datum_header + datum_id + datum_body] ...
        stream = BytesIO(forwarded_bytes)
        # skip main section
        header_buf = stream.read(HEADER_LEN)
        hdr = _Header.from_bytes(header_buf)
        stream.read(hdr.size)  # skip main body

        # Read datum sections
        found_ref_datum = False
        while True:
            hdr_buf = stream.read(HEADER_LEN)
            if not hdr_buf:
                break
            hdr = _Header.from_bytes(hdr_buf)
            datum_id_bytes = stream.read(16)
            body = stream.read(hdr.size - 16)

            if hdr.marker == MARKER_DATUM_TEXT:
                text = body.decode("utf-8")
                ref = json.loads(text)
                if _RefKey.FQCN in ref and _RefKey.REF_ID in ref:
                    expected_server = server.get_fqcn()
                    assert ref[_RefKey.FQCN] == expected_server, (
                        f"Forwarded datum fqcn must be '{expected_server}' (FL server), "
                        f"but got '{ref[_RefKey.FQCN]}'. Subprocess would download from wrong source."
                    )
                    assert ref[_RefKey.REF_ID], "Forwarded datum must have a non-empty ref_id."
                    found_ref_datum = True

        assert found_ref_datum, (
            "No download-ref datum found in forwarded_bytes. "
            "CJ re-serialization must include the server's fqcn/ref_id datum."
        )

    # ------------------------------------------------------------------
    # Test 5 — larger batch (simulates a realistic model)
    # ------------------------------------------------------------------

    def test_multiple_array_roundtrip(self, cells):
        """A batch of 8 arrays all survive the pass-through hop with correct values."""
        server, subproc = cells
        rng = np.random.default_rng(7)
        original = {f"layer{i}": rng.standard_normal((32, 32)).astype(np.float32) for i in range(8)}

        server_bytes = dump_to_bytes(original, fobs_ctx={FOBSContextKey.CELL: server})
        _, forwarded_bytes = _simulate_cj_pass_through(server_bytes)
        result = load_from_bytes(forwarded_bytes, fobs_ctx={FOBSContextKey.CELL: subproc})

        assert set(result.keys()) == set(original.keys())
        for key in original:
            np.testing.assert_array_almost_equal(result[key], original[key], err_msg=f"Mismatch for '{key}'")
