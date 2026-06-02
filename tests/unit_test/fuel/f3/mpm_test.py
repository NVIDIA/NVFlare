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
import os
from unittest.mock import MagicMock, patch

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.fuel.f3.mpm import MainProcessMonitor


def _fake_thread(name, daemon):
    t = MagicMock()
    t.name = name
    t.daemon = daemon
    return t


class TestMainProcessMonitorReturnCode:
    """Regression tests for FLARE-2976.

    When an ext_process job's main_func returns implicitly (None) and non-daemon
    threads are still alive at shutdown, MPM used to write the literal string "None"
    to _process_rc.txt. The parent then did int("None") -> ValueError and mislogged a
    successful job as RC=1. The value written must always be an int the parent can parse.
    """

    def _run_and_read_rc(self, run_dir, main_func):
        rc_file = os.path.join(str(run_dir), FLMetaKey.PROCESS_RC_FILE)

        main_thread = _fake_thread("MainThread", daemon=False)
        # a non-daemon, non-MainThread worker (e.g. cellnet conn_mgr / frame_mgr pool)
        # still alive at shutdown is what forces the rc-file write + SIGKILL path
        lingering = _fake_thread("cellnet-worker", daemon=False)

        with patch("nvflare.fuel.f3.mpm.threading.current_thread", return_value=main_thread), patch(
            "nvflare.fuel.f3.mpm.threading.enumerate", return_value=[main_thread, lingering]
        ), patch.object(MainProcessMonitor, "_start_shutdown"), patch(
            "nvflare.fuel.f3.mpm.AioContext.close_global_context"
        ), patch(
            "nvflare.fuel.f3.mpm.os.kill"
        ):
            MainProcessMonitor.run(main_func, run_dir=str(run_dir))

        with open(rc_file) as f:
            return f.read().strip()

    def test_implicit_none_return_writes_parseable_zero(self, tmp_path):
        content = self._run_and_read_rc(tmp_path, lambda: None)
        # must be int-parseable (not "None") and represent success
        assert int(content) == 0

    def test_int_return_code_is_preserved(self, tmp_path):
        # a real exit code (e.g. ProcessExitCode.CONFIG_ERROR == 103) must pass through unchanged
        content = self._run_and_read_rc(tmp_path, lambda: 103)
        assert int(content) == 103

    def test_non_int_return_is_coerced(self, tmp_path):
        # any other non-int return value also stays int-parseable
        content = self._run_and_read_rc(tmp_path, lambda: "done")
        assert int(content) == 0
