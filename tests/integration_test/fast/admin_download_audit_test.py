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

"""Real download transport and the administrative terminal-audit callback.

An isolated child owns real Cells and the transfer service. Minimal session
binding is controlled here; real administrator sessions are exercised by the
peer-administrator integration suite. A completed record describes the source
service's report only, without claiming receiver receipt or local extraction.
"""

import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import pytest

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.streaming.download_service import Consumer, download_object
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import MetaKey
from nvflare.fuel.hci.server.binary_transfer import BinaryTransfer
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.sec.admin_audit import audit_context
from nvflare.fuel.sec.audit import AuditService
from nvflare.fuel.utils.network_utils import get_open_ports
from tests.process_test_utils import run_owned_process
from tests.timing_utils import wait_for


class _SessionBinding:
    monitor_interval = 0.1

    def __init__(self, idle_timeout):
        self.idle_timeout = idle_timeout
        self.callbacks = {}

    def bind_download(self, session_id, tx_id, cancel):
        self.callbacks[tx_id] = cancel
        return True

    def mark_download_active(self, session_id, tx_id):
        return tx_id in self.callbacks

    def end_download(self, session_id, tx_id):
        self.callbacks.pop(tx_id, None)

    def cancel_all(self):
        for cancel in list(self.callbacks.values()):
            cancel()


class _Consumer(Consumer):
    def __init__(self, fail=False):
        super().__init__()
        self.data = bytearray()
        self.fail = fail
        self.done = False
        self.error = None

    def consume(self, ref_id, state, data):
        if self.fail:
            raise RuntimeError("private-consumer-error-marker")
        self.data.extend(data)
        return state

    def download_completed(self, ref_id):
        self.done = True

    def download_failed(self, ref_id, reason):
        self.done = True
        self.error = reason


def _transfer_worker(directory):
    directory = Path(directory)
    expected = b"private-download-content-marker" * 16
    audit_path = directory / "audit.jsonl"
    root_url = f"tcp://localhost:{get_open_ports(1)[0]}"
    with ExitStack() as cleanup:
        server = Cell("server", root_url, secure=False, credentials={})
        cleanup.callback(server.core_cell.stop)
        server.core_cell.start()
        client = Cell("admin", root_url, secure=False, credentials={})
        cleanup.callback(client.core_cell.stop)
        client.core_cell.start()
        wait_for(
            lambda: client.core_cell.is_cell_connected(server.get_fqcn())
            and server.core_cell.is_cell_connected(client.get_fqcn()),
            message="download audit Cells did not connect",
        )
        AuditService.initialize(str(audit_path))
        cleanup.callback(AuditService.close)
        outcomes = [
            ("complete", "completed", "all_receivers_succeeded"),
            ("receiver_failure", "failed", "receiver_failed"),
            ("cancel", "aborted", "deleted"),
            ("timeout", "failed", "timeout"),
        ]
        for case, status, reason in outcomes:
            manager = _SessionBinding(0.1 if case == "timeout" else 30.0)
            cleanup.callback(manager.cancel_all)
            source_folder = directory / "download" / case / "job-1"
            source_folder.mkdir(parents=True)
            (source_folder / "private-result.txt").write_bytes(expected)
            conn = Connection(
                props={
                    ConnProps.DOWNLOAD_DIR: str(directory / "download"),
                    ConnProps.ENGINE: SimpleNamespace(get_cell=lambda: server),
                    ConnProps.ADMIN_SERVER: SimpleNamespace(timeout=0.2, sess_mgr=manager),
                    ConnProps.SESSION: SimpleNamespace(sess_id="private-session-secret"),
                }
            )
            with audit_context(
                user="admin@test.org", request_id=f"request-{case}", command="download_job", job_id="job-1"
            ):
                BinaryTransfer().download_folder(conn, case, "job-1")
            meta = conn.buffer.meta
            assert len(meta[MetaKey.FILES]) == 1
            ref_id = meta[MetaKey.FILES][0][1]
            if case in ("complete", "receiver_failure"):
                consumer = _Consumer(fail=case == "receiver_failure")
                download_object(
                    from_fqcn=server.get_fqcn(),
                    ref_id=ref_id,
                    per_request_timeout=3.0,
                    cell=client,
                    consumer=consumer,
                    max_retries=0,
                )
                assert consumer.done
                if case == "complete":
                    assert consumer.error is None
                    assert consumer.data == expected
                else:
                    assert consumer.error is not None
            elif case == "cancel":
                # Cancel the live source transaction before the first receiver pull.
                # The service reports deletion, not a successful receiver delivery.
                manager.cancel_all()

            def recorded():
                records = [json.loads(line) for line in audit_path.read_text().splitlines()]
                return [r for r in records if r.get("reference") == f"request-{case}"]

            records = wait_for(recorded, timeout=15, message=f"missing terminal download audit for {case}")
            assert len(records) == 1, records
            event = records[0]
            assert event["action"] == f"download.transfer.{status}", event
            assert event["user"] == "admin@test.org"
            assert event["job"] == "job-1"
            assert f"reason={reason}" in event["message"]
            assert not source_folder.exists()
            assert not manager.callbacks
        records = [json.loads(line) for line in audit_path.read_text().splitlines()]
        assert len(records) == len(outcomes)
        for event in records:
            assert set(event) <= {"event_id", "reference", "timestamp", "user", "action", "job", "scope", "message"}
            assert all(isinstance(value, str) for value in event.values())
            assert "private-" not in json.dumps(event)
            assert "scope" not in event
            assert "receipt" not in event["message"]
            assert "extract" not in event["message"]


@pytest.mark.timeout(100)
def test_real_administrative_download_terminal_records(tmp_path):
    root = Path(__file__).resolve().parents[3]
    result = run_owned_process(
        [
            sys.executable,
            "-c",
            'import runpy, sys; runpy.run_path(sys.argv[1])["_transfer_worker"](sys.argv[2])',
            str(Path(__file__).resolve()),
            str(tmp_path),
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root) + os.pathsep + os.environ.get("PYTHONPATH", "")},
        timeout=85,
    )
    assert result.returncode == 0, result.stdout + result.stderr
