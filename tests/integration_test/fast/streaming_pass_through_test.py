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

Validates the full data path exercised by the external-process Client API when an
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
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from nvflare.apis.fl_constant import ServerCommandNames
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.cellnet.defs import CellChannel, MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply, new_cell_message
from nvflare.fuel.f3.streaming.download_service import OBJ_DOWNLOADER_CHANNEL, DownloadService
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs import FOBSContextKey, dots
from nvflare.fuel.utils.fobs.decomposers.via_downloader import LazyDownloadRef, _RefKey
from nvflare.fuel.utils.fobs.lobs import dump_to_bytes, load_from_bytes
from nvflare.fuel.utils.network_utils import get_open_ports
from tests.timing_utils import WAIT_TIMEOUT, ThreadGroup, join_thread, wait_for


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
        subproc = Cell(subproc_fqcn, f"tcp://localhost:{port}", secure=False, credentials={})
        try:
            server.core_cell.start()
            subproc.core_cell.start()
            wait_for(
                lambda: server.core_cell.running
                and subproc.core_cell.running
                and server.core_cell.is_cell_connected(subproc_fqcn)
                and subproc.core_cell.is_cell_connected(server_fqcn),
                message="pass-through cells did not become ready",
            )
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

    def test_disk_backed_tensor_ref_travels_as_a_plain_tensor(self, cells, tmp_path):
        import torch
        from safetensors.torch import save_file

        from nvflare.app_opt.pt.decomposers import TensorDecomposer
        from nvflare.app_opt.pt.lazy_tensor_dict import safetensors_refs

        server, subproc = cells
        expected = torch.arange(16, dtype=torch.float32)
        save_file({"weight": expected}, tmp_path / "weight.safetensors")
        ref = safetensors_refs(str(tmp_path / "weight.safetensors"))["weight"]

        fobs.register(TensorDecomposer)
        server_bytes = dump_to_bytes({"weight": ref}, fobs_ctx={FOBSContextKey.CELL: server})

        cj_result, forwarded_bytes = _simulate_cj_pass_through(server_bytes)
        assert isinstance(cj_result["weight"], LazyDownloadRef)
        assert cj_result["weight"].dot == dots.TENSOR_DOWNLOAD

        result = load_from_bytes(forwarded_bytes, fobs_ctx={FOBSContextKey.CELL: subproc})
        assert torch.equal(result["weight"], expected)


def _late_update_shutdown_worker():
    """Exercise the fatal server-job adapter in a disposable interpreter."""
    _register_numpy()
    url = f"tcp://127.0.0.1:{get_open_ports(1)[0]}"
    cells = []
    chunk_received = threading.Event()
    release_chunk = threading.Event()
    callback_entered = threading.Event()
    responses = []
    try:
        server = Cell("server", url, secure=False, credentials={}, create_internal_listener=True)
        cells.append(server)
        server.start()
        parent_url = wait_for(server.get_internal_listener_url, message="server listener did not start")
        job = Cell("server.job-1", url, secure=False, credentials={}, parent_url=parent_url)
        cells.append(job)
        job.start()
        source = Cell("site-3", url, secure=False, credentials={})
        cells.append(source)
        source.start()
        wait_for(
            lambda: server.is_cell_connected("site-3") and server.is_cell_connected("server.job-1"),
            message="late-update cells did not connect",
        )

        def submit_update(request):
            callback_entered.set()
            return make_reply(ReturnCode.OK)

        job.register_request_cb(
            channel=CellChannel.SERVER_COMMAND, topic=ServerCommandNames.SUBMIT_UPDATE, cb=submit_update
        )
        job.register_request_cb(
            channel=CellChannel.SERVER_COMMAND,
            topic=ServerCommandNames.GET_TASK,
            cb=lambda request: make_reply(ReturnCode.OK, body="alive"),
        )
        send_request = job.send_request

        def pause_download(*args, **kwargs):
            reply = send_request(*args, **kwargs)
            if kwargs.get("channel") == OBJ_DOWNLOADER_CHANNEL:
                chunk_received.set()
                release_chunk.wait()
            return reply

        def send_update():
            responses.append(
                source.send_request(
                    channel=CellChannel.SERVER_COMMAND,
                    topic=ServerCommandNames.SUBMIT_UPDATE,
                    target=job.get_fqcn(),
                    request=new_cell_message(headers={}, payload={"weights": np.ones(1024, dtype=np.float32)}),
                    timeout=WAIT_TIMEOUT,
                )
            )

        with patch.object(job, "send_request", side_effect=pause_download), ThreadGroup(release_chunk) as workers:
            sender = workers.start(send_update)
            assert chunk_received.wait(WAIT_TIMEOUT), "late result never entered a real object download"
            # The first two clients have already finished the final round. The
            # server now shuts down while this third result is being decoded.
            DownloadService.shutdown()
            release_chunk.set()
            join_thread(sender)

        assert responses[0].get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.SERVICE_UNAVAILABLE
        assert not callback_entered.is_set(), "a cancelled payload reached the application"
        response = source.send_request(
            channel=CellChannel.SERVER_COMMAND,
            topic=ServerCommandNames.GET_TASK,
            target=job.get_fqcn(),
            request=new_cell_message(headers={}),
            timeout=WAIT_TIMEOUT,
        )
        assert response.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
        assert response.payload == "alive"
        print("late-update shutdown cancelled safely; server job remains alive", flush=True)
    finally:
        release_chunk.set()
        DownloadService.shutdown()
        for cell in reversed(cells):
            cell.stop()
            CoreCell.ALL_CELLS.pop(cell.get_fqcn(), None)


@pytest.mark.timeout(70)
def test_late_result_download_during_shutdown_does_not_exit_server_job():
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; runpy.run_path(sys.argv[1])['_late_update_shutdown_worker']()",
            str(Path(__file__).resolve()),
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "late-update shutdown cancelled safely; server job remains alive" in result.stdout
