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

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from nvflare.apis.shareable import Shareable
from nvflare.private.fed.server.server_engine import ServerEngine
from nvflare.private.fed.server.server_runner import ServerRunner


def _make_engine():
    args = SimpleNamespace(set=[])
    engine = ServerEngine(
        server=MagicMock(),
        args=args,
        client_manager=MagicMock(),
        snapshot_persistor=MagicMock(),
    )
    engine.logger = MagicMock()
    return engine


class TestServerEngineGetCell:
    def test_returns_parent_cell_even_when_run_manager_cell_present(self):
        engine = _make_engine()
        parent_cell = MagicMock(name="parent_cell")
        run_cell = MagicMock(name="run_cell")
        engine.cell = parent_cell
        engine.run_manager = SimpleNamespace(cell=run_cell)

        assert engine.get_cell() is parent_cell

    def test_falls_back_to_parent_cell_when_run_cell_missing(self):
        engine = _make_engine()
        parent_cell = MagicMock(name="parent_cell")
        engine.cell = parent_cell
        engine.run_manager = SimpleNamespace(cell=None)

        assert engine.get_cell() is parent_cell

    def test_returns_none_when_no_cells_available(self):
        engine = _make_engine()
        engine.cell = None
        engine.run_manager = None

        assert engine.get_cell() is None


def _make_server_runner_for_submission(status="started"):
    runner = ServerRunner.__new__(ServerRunner)
    runner.wf_lock = threading.RLock()
    runner.status = status
    runner.current_wf = MagicMock()
    runner.log_info = MagicMock()
    runner._report_client_active = MagicMock()
    return runner


class TestLateSubmissionAdmission:
    def test_submission_after_terminal_state_does_not_touch_run_state(self):
        runner = _make_server_runner_for_submission(status="done")
        fl_ctx = MagicMock()

        with patch.object(runner, "_process_submission") as process_submission:
            runner.process_submission(MagicMock(name="client"), "train", "task-1", Shareable(), fl_ctx)

        process_submission.assert_not_called()
        runner._report_client_active.assert_not_called()
        fl_ctx.set_prop.assert_not_called()

    def test_submission_queued_behind_teardown_is_dropped(self):
        runner = _make_server_runner_for_submission()
        fl_ctx = MagicMock()
        finished = threading.Event()

        def submit():
            runner.process_submission(MagicMock(name="client"), "train", "task-1", Shareable(), fl_ctx)
            finished.set()

        with runner.wf_lock:
            thread = threading.Thread(target=submit)
            thread.start()
            runner.status = "done"
            runner.current_wf = None

        assert finished.wait(timeout=1.0)
        thread.join(timeout=1.0)
        runner._report_client_active.assert_not_called()
        fl_ctx.set_prop.assert_not_called()
