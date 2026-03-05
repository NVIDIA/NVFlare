# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import shutil
import tempfile
from io import BufferedReader, BytesIO
from unittest.mock import Mock

import pytest

from unittest.mock import patch

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.app_common.launchers.subprocess_launcher import (
    SubprocessLauncher,
    _route_subprocess_line,
    log_subprocess_output,
)


class TestSubprocessLauncher:
    def test_launch(self):
        tempdir = tempfile.mkdtemp()
        fl_ctx = FLContext()
        launcher = SubprocessLauncher("echo 'test'")
        launcher._app_dir = tempdir

        signal = Signal()
        task_name = "__test_task"
        dxo = DXO(DataKind.WEIGHTS, {})
        status = launcher.launch_task(task_name, dxo.to_shareable(), fl_ctx, signal)
        assert status is True
        shutil.rmtree(tempdir)

    def test_stop(self):
        tempdir = tempfile.mkdtemp()
        fl_ctx = FLContext()
        launcher = SubprocessLauncher("python -c \"for i in range(1000000): print('cool')\"", shutdown_timeout=0.0)
        launcher._app_dir = tempdir

        signal = Signal()
        task_name = "__test_task"
        dxo = DXO(DataKind.WEIGHTS, {})
        status = launcher.launch_task(task_name, dxo.to_shareable(), fl_ctx, signal)
        assert status is True
        launcher.stop_task(task_name, fl_ctx, signal)

        assert launcher._process is None
        shutil.rmtree(tempdir)

    def test_default_launch_once_is_true(self):
        """Test that launch_once defaults to True."""
        launcher = SubprocessLauncher("echo 'test'")
        assert launcher._launch_once is True

    def test_default_shutdown_timeout_is_zero(self):
        """Test that shutdown_timeout defaults to 0.0."""
        launcher = SubprocessLauncher("echo 'test'")
        assert launcher._shutdown_timeout == 0.0

    @pytest.mark.parametrize(
        "launch_once,shutdown_timeout",
        [
            (True, 0.0),  # Default values
            (False, 0.0),  # launch_once=False with default timeout
            (True, 10.0),  # launch_once=True with custom timeout
            (False, 15.0),  # launch_once=False with custom timeout
            (True, 100.0),  # Large timeout value
        ],
    )
    def test_launch_once_and_shutdown_timeout_initialization(self, launch_once, shutdown_timeout):
        """Test various launch_once and shutdown_timeout configurations."""
        launcher = SubprocessLauncher(script="echo 'test'", launch_once=launch_once, shutdown_timeout=shutdown_timeout)
        assert launcher._launch_once == launch_once
        assert launcher._shutdown_timeout == shutdown_timeout

    def test_launch_once_false_behavior(self):
        """Test that launch_once=False is properly configured."""
        launcher = SubprocessLauncher("echo 'test'", launch_once=False)

        # Verify launch_once is set to False
        assert launcher._launch_once is False

        # Verify the process is initially None
        assert launcher._process is None

    def test_launch_once_true_behavior(self):
        """Test that launch_once=True is properly configured."""
        launcher = SubprocessLauncher("echo 'test'", launch_once=True)

        # Verify launch_once is set to True
        assert launcher._launch_once is True

        # Verify the process is initially None
        assert launcher._process is None

    def test_shutdown_timeout_parameter(self):
        """Test that shutdown_timeout is properly stored."""
        launcher = SubprocessLauncher("echo 'test'", shutdown_timeout=30.0)
        assert launcher._shutdown_timeout == 30.0

    def test_clean_up_script_with_launch_once(self):
        """Test that clean_up_script can be configured with launch_once=False."""
        launcher = SubprocessLauncher(
            script="echo 'main'",
            launch_once=False,
            clean_up_script="echo 'cleanup'",
            shutdown_timeout=0.0,
        )

        # Verify parameters are set correctly
        assert launcher._launch_once is False
        assert launcher._clean_up_script == "echo 'cleanup'"
        assert launcher._shutdown_timeout == 0.0

    def test_log_subprocess_output(self):
        class _Proc:
            pass

        p = _Proc()
        p.stdout = BufferedReader(BytesIO(b"line1\nline2\r\npartial"))
        logger = Mock()
        log_subprocess_output(p, logger)

        logged = [call.args[0] for call in logger.info.call_args_list]
        assert logged == ["line1", "line2", "partial"]

    def test_log_subprocess_output_formatted_lines_not_double_logged(self):
        """Formatted NVFlare log lines must NOT be re-logged via logger.info().

        When the subprocess has both consoleHandler and file handlers (same config
        as parent CJ), formatted lines are already written to log.txt by the file
        handler. The parent captures stdout and must only print() them to the
        terminal — calling logger.info() would create a second copy in log.txt.
        """

        class _Proc:
            pass

        ts_line = b"2026-03-05 10:00:00 - nvflare.foo - INFO - [] - hello\n"
        plain_line = b"training loss: 0.5\n"
        p = _Proc()
        p.stdout = BufferedReader(BytesIO(ts_line + plain_line))
        logger = Mock()
        with patch("builtins.print") as mock_print:
            log_subprocess_output(p, logger)

        # Formatted line → print() only (no double logging in log.txt)
        mock_print.assert_called_once_with("2026-03-05 10:00:00 - nvflare.foo - INFO - [] - hello")
        # Plain line → logger.info() (reaches log.txt)
        logger.info.assert_called_once_with("training loss: 0.5")


class TestRouteSubprocessLine:
    """Unit tests for _route_subprocess_line() routing logic."""

    def test_plain_print_line_goes_to_logger_info(self):
        """Raw print() from user training script must reach logger.info() → log.txt."""
        logger = Mock()
        with patch("builtins.print") as mock_print:
            _route_subprocess_line("training loss: 0.5", logger)

        logger.info.assert_called_once_with("training loss: 0.5")
        mock_print.assert_not_called()

    def test_formatted_log_line_goes_to_print_not_logger(self):
        """Formatted NVFlare log line must NOT be re-logged — goes to print() only.

        The file handler in the subprocess writes it to log.txt already.
        Calling logger.info() here would create a duplicate entry.
        """
        logger = Mock()
        line = "2026-03-05 10:00:00 - nvflare.foo - INFO - [] - some message"
        with patch("builtins.print") as mock_print:
            _route_subprocess_line(line, logger)

        mock_print.assert_called_once_with(line)
        logger.info.assert_not_called()

    def test_ansi_colored_formatted_line_goes_to_print_not_logger(self):
        """ANSI-colored formatted line from consoleHandler must go to print() only.

        ANSI codes must be stripped before the timestamp pattern is checked so
        the color prefix does not defeat the guard and cause double logging.
        """
        logger = Mock()
        # Simulate ColorFormatter output: ANSI green prefix before timestamp
        ansi_line = "\x1b[32m2026-03-05 10:00:00\x1b[0m - nvflare.foo - INFO - [] - colored"
        with patch("builtins.print") as mock_print:
            _route_subprocess_line(ansi_line, logger)

        mock_print.assert_called_once_with(ansi_line)
        logger.info.assert_not_called()

    def test_empty_line_goes_to_logger_info(self):
        """An empty/whitespace-only line has no timestamp → logger.info()."""
        logger = Mock()
        with patch("builtins.print") as mock_print:
            _route_subprocess_line("", logger)

        logger.info.assert_called_once_with("")
        mock_print.assert_not_called()

    def test_partial_timestamp_line_goes_to_logger_info(self):
        """A line starting with something that looks like a partial date but lacks
        the full 'YYYY-MM-DD HH:MM:SS' pattern must NOT be treated as a log line."""
        logger = Mock()
        line = "2026-03-05 epoch complete"  # date but no HH:MM:SS
        with patch("builtins.print") as mock_print:
            _route_subprocess_line(line, logger)

        logger.info.assert_called_once_with(line)
        mock_print.assert_not_called()
