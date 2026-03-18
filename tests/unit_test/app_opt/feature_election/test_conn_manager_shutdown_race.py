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

"""Regression tests for shutdown-race guards in ConnManager.

Root cause: Between stop() completing (executor shut down) and in-flight
callbacks completing, start_connector() and process_frame_task() could be
called after the ThreadPoolExecutors were torn down.

Fix:
- start_connector(): wraps executor.submit() in try/except RuntimeError so
  a post-shutdown call logs a debug message and returns cleanly.
- process_frame_task(): checks self.stopped at entry and returns immediately.

These tests lock in that behaviour so future refactors cannot regress it.
"""

import threading
from unittest.mock import MagicMock

from nvflare.fuel.f3.drivers.connector_info import ConnectorInfo
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.sfm.conn_manager import ConnManager
from nvflare.fuel.utils.constants import Mode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conn_manager() -> ConnManager:
    endpoint = Endpoint(name="test-endpoint")
    return ConnManager(local_endpoint=endpoint)


def _make_connector(started: bool = False) -> ConnectorInfo:
    driver = MagicMock()
    # short_url() requires scheme/host/port/path keys to format the connector log line
    params = {"scheme": "tcp", "host": "localhost", "port": "8080", "path": ""}
    return ConnectorInfo(
        handle="h1",
        driver=driver,
        params=params,
        mode=Mode.ACTIVE,
        total_conns=0,
        curr_conns=0,
        started=started,
        stopped=threading.Event(),
    )


def _make_sfm_conn():
    sfm_conn = MagicMock()
    sfm_conn.get_name.return_value = "test-conn"
    return sfm_conn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestShutdownRaceGuards:
    """start_connector and process_frame_task must not raise after shutdown."""

    def test_start_connector_after_executor_shutdown_does_not_raise(self):
        """start_connector() must silently skip submission after executor is shut down.

        Simulates the race where stop() shuts down conn_mgr_executor but a
        late add_connector() callback fires after that.
        """
        mgr = _make_conn_manager()
        mgr.conn_mgr_executor.shutdown(wait=False)

        connector = _make_connector(started=False)

        # Must not raise RuntimeError — the guard catches it and logs debug.
        mgr.start_connector(connector)  # no exception expected

    def test_start_connector_after_stop_does_not_raise(self):
        """start_connector() must not raise when called after mgr.stop().

        mgr.stop() shuts down the executor; any subsequent start_connector()
        call (e.g. from a concurrent thread still holding a connector ref)
        must be silently dropped.
        """
        mgr = _make_conn_manager()
        mgr.heartbeat_monitor = MagicMock()  # avoid real heartbeat thread
        mgr.stop()

        connector = _make_connector(started=False)
        mgr.start_connector(connector)  # no exception expected

    def test_process_frame_task_after_stopped_returns_immediately(self):
        """process_frame_task() must return without processing when self.stopped=True.

        Simulates an in-flight frame arriving from the network thread after
        stop() has been called.  The frame must be silently discarded.
        """
        mgr = _make_conn_manager()
        mgr.stopped = True

        sfm_conn = _make_sfm_conn()
        # Use a minimal valid frame; if the guard is missing, Prefix.from_bytes
        # would raise or attempt real work on this dummy payload.
        dummy_frame = b"\x00" * 8

        # Must not raise and must not attempt to decode the frame.
        mgr.process_frame_task(sfm_conn, dummy_frame)

    def test_process_frame_task_stopped_does_not_call_prefix_from_bytes(self):
        """process_frame_task() with self.stopped=True must not decode the frame.

        Verifies the early-return guard fires before any frame parsing, so
        malformed post-shutdown frames cannot cause spurious errors in logs.
        """
        from unittest.mock import patch

        mgr = _make_conn_manager()
        mgr.stopped = True

        sfm_conn = _make_sfm_conn()

        with patch("nvflare.fuel.f3.sfm.conn_manager.Prefix.from_bytes") as mock_parse:
            mgr.process_frame_task(sfm_conn, b"\x00" * 8)
            mock_parse.assert_not_called()
