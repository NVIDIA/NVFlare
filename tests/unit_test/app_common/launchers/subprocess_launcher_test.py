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
import signal as signal_module
import subprocess
import tempfile
from io import BufferedReader, BytesIO
from unittest.mock import Mock, patch

import pytest

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.app_common.launchers.subprocess_launcher import SubprocessLauncher
from nvflare.fuel.utils.secret_utils import secret_file_ref
from nvflare.utils.process_utils import _route_subprocess_line, log_subprocess_output


class TestSubprocessLauncher:
    @staticmethod
    def _make_fl_ctx(tmp_dir):
        class _Workspace:
            def get_app_custom_dir(self, job_id):
                return tmp_dir

        fl_ctx = FLContext()
        fl_ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, _Workspace(), private=True, sticky=False)
        fl_ctx.set_prop(FLContextKey.CURRENT_JOB_ID, "test_job", private=True, sticky=False)
        return fl_ctx

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

    def test_start_external_process_uses_posix_process_group(self, monkeypatch, tmp_path):
        popen_calls = []

        class _Proc:
            pid = 1234
            stdout = BufferedReader(BytesIO(b""))

            def __init__(self, *args, **kwargs):
                popen_calls.append({"args": args, "kwargs": kwargs})

        monkeypatch.setattr("nvflare.app_common.launchers.subprocess_launcher.os.name", "posix")
        monkeypatch.setattr("nvflare.app_common.launchers.subprocess_launcher.subprocess.Popen", _Proc)
        launcher = SubprocessLauncher("python train.py", launch_once=False)
        launcher._app_dir = str(tmp_path)

        launcher.launch_task(
            "__test_task", DXO(DataKind.WEIGHTS, {}).to_shareable(), self._make_fl_ctx(str(tmp_path)), Signal()
        )
        launcher._log_thread.join()

        assert popen_calls[0]["kwargs"]["start_new_session"] is True
        assert launcher._process is not None

    def test_start_external_process_preserves_argv_sequence(self, monkeypatch, tmp_path):
        popen_calls = []

        class _Proc:
            pid = 1234
            stdout = BufferedReader(BytesIO(b""))

            def __init__(self, *args, **kwargs):
                popen_calls.append({"args": args, "kwargs": kwargs})

        monkeypatch.setattr("nvflare.app_common.launchers.subprocess_launcher.subprocess.Popen", _Proc)
        command = ["python3", "custom/train model.py", "--label", "two words", "--empty", ""]
        launcher = SubprocessLauncher(command, launch_once=False)
        launcher._app_dir = str(tmp_path)

        launcher.launch_task(
            "__test_task", DXO(DataKind.WEIGHTS, {}).to_shareable(), self._make_fl_ctx(str(tmp_path)), Signal()
        )
        launcher._log_thread.join()

        assert popen_calls[0]["args"][0] == command

    def test_start_external_process_resolves_secret_refs(self, monkeypatch, tmp_path):
        popen_calls = []

        class _Proc:
            pid = 1234
            stdout = BufferedReader(BytesIO(b""))

            def __init__(self, *args, **kwargs):
                popen_calls.append({"args": args, "kwargs": kwargs})

        monkeypatch.setattr("nvflare.app_common.launchers.subprocess_launcher.subprocess.Popen", _Proc)
        monkeypatch.setenv("TEST_SECRET_REF_VAR", "resolved secret")
        launcher = SubprocessLauncher("python train.py --api-key ${secret:TEST_SECRET_REF_VAR}", launch_once=False)
        launcher._app_dir = str(tmp_path)

        launcher.launch_task(
            "__test_task", DXO(DataKind.WEIGHTS, {}).to_shareable(), self._make_fl_ctx(str(tmp_path)), Signal()
        )
        launcher._log_thread.join()

        # the value from the site env is injected as a single argument, even with whitespace
        assert popen_calls[0]["args"][0] == ["python", "train.py", "--api-key", "resolved secret"]
        # the launcher's configured script (what job configs carry) still holds only the placeholder
        assert "${secret:TEST_SECRET_REF_VAR}" in launcher._script
        assert "resolved secret" not in launcher._script

    def test_start_external_process_resolves_file_ref_as_one_argument(self, monkeypatch, tmp_path):
        popen_calls = []

        class _Proc:
            pid = 1234
            stdout = BufferedReader(BytesIO(b""))

            def __init__(self, *args, **kwargs):
                popen_calls.append({"args": args, "kwargs": kwargs})

        secret_path = tmp_path / "api'key\\value"
        secret_path.write_text("resolved secret --not-option")
        placeholder = secret_file_ref(str(secret_path))
        monkeypatch.setattr("nvflare.app_common.launchers.subprocess_launcher.subprocess.Popen", _Proc)
        launcher = SubprocessLauncher(f"python train.py --api-key {placeholder}", launch_once=False)
        launcher._app_dir = str(tmp_path)

        launcher.launch_task(
            "__test_task", DXO(DataKind.WEIGHTS, {}).to_shareable(), self._make_fl_ctx(str(tmp_path)), Signal()
        )
        launcher._log_thread.join()

        assert popen_calls[0]["args"][0] == ["python", "train.py", "--api-key", "resolved secret --not-option"]
        assert placeholder in launcher._script
        assert "resolved secret" not in launcher._script

    def test_start_external_process_missing_secret_ref_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr("nvflare.app_common.launchers.subprocess_launcher.subprocess.Popen", Mock())
        monkeypatch.delenv("TEST_UNSET_SECRET_VAR", raising=False)
        launcher = SubprocessLauncher("python train.py --api-key ${secret:TEST_UNSET_SECRET_VAR}", launch_once=False)
        launcher._app_dir = str(tmp_path)

        with pytest.raises(ValueError, match="TEST_UNSET_SECRET_VAR"):
            launcher.launch_task(
                "__test_task", DXO(DataKind.WEIGHTS, {}).to_shareable(), self._make_fl_ctx(str(tmp_path)), Signal()
            )

    @pytest.mark.parametrize(
        "script",
        [
            "sh -c 'echo ${secret:TEST_SECRET_REF_VAR}'",
            "/bin/bash -lc 'echo ${secret:TEST_SECRET_REF_VAR}'",
            "bash --rcfile /tmp/bashrc -c 'echo ${secret:TEST_SECRET_REF_VAR}'",
            "env FOO=bar sh -c 'echo ${secret:TEST_SECRET_REF_VAR}'",
            "env -- FOO=bar sh -c 'echo ${secret:TEST_SECRET_REF_VAR}'",
            "env -- OUTER=1 env -- INNER=2 sh -c 'echo ${secret:TEST_SECRET_REF_VAR}'",
            "/usr/bin/env -i FOO=bar /bin/bash -c 'echo ${secret:TEST_SECRET_REF_VAR}'",
            "env --unknown-option value sh -c 'echo ${secret:TEST_SECRET_REF_VAR}'",
            "env -S \"bash -c 'echo ${secret:TEST_SECRET_REF_VAR}'\"",
            "pwsh -Command 'Invoke-Expression ${secret:TEST_SECRET_REF_VAR}'",
            "pwsh -Bogus value -Command 'Invoke-Expression ${secret:TEST_SECRET_REF_VAR}'",
            "pwsh -EncodedCommand ${secret:TEST_SECRET_REF_VAR}",
            "pwsh -enc ${secret:TEST_SECRET_REF_VAR}",
        ],
    )
    def test_start_external_process_rejects_secret_ref_in_nested_command(self, monkeypatch, tmp_path, script):
        popen = Mock()
        monkeypatch.setattr("nvflare.app_common.launchers.subprocess_launcher.subprocess.Popen", popen)
        monkeypatch.setenv("TEST_SECRET_REF_VAR", "'; injected-command #")
        launcher = SubprocessLauncher(script, launch_once=False)
        launcher._app_dir = str(tmp_path)

        with pytest.raises(ValueError, match="nested interpreter command strings"):
            launcher.launch_task(
                "__test_task", DXO(DataKind.WEIGHTS, {}).to_shareable(), self._make_fl_ctx(str(tmp_path)), Signal()
            )

        popen.assert_not_called()

    @pytest.mark.parametrize(
        "script,expected_command",
        [
            (
                "bash train.sh ${secret:TEST_SECRET_REF_VAR}",
                ["bash", "train.sh", "resolved secret"],
            ),
            (
                "pwsh -File train.ps1 ${secret:TEST_SECRET_REF_VAR}",
                ["pwsh", "-File", "train.ps1", "resolved secret"],
            ),
            (
                "env -- API_TOKEN=${secret:TEST_SECRET_REF_VAR} sh -c 'echo \"$API_TOKEN\"'",
                ["env", "--", "API_TOKEN=resolved secret", "sh", "-c", 'echo "$API_TOKEN"'],
            ),
            (
                "echo python -c ${secret:TEST_SECRET_REF_VAR}",
                ["echo", "python", "-c", "resolved secret"],
            ),
            (
                "python train.py --label node -e ${secret:TEST_SECRET_REF_VAR}",
                ["python", "train.py", "--label", "node", "-e", "resolved secret"],
            ),
            (
                "pwsh train.ps1 -Command ${secret:TEST_SECRET_REF_VAR}",
                ["pwsh", "train.ps1", "-Command", "resolved secret"],
            ),
        ],
    )
    def test_start_external_process_allows_secret_ref_in_interpreter_argv(
        self, monkeypatch, tmp_path, script, expected_command
    ):
        popen_calls = []

        class _Proc:
            pid = 1234
            stdout = BufferedReader(BytesIO(b""))

            def __init__(self, *args, **kwargs):
                popen_calls.append({"args": args, "kwargs": kwargs})

        monkeypatch.setattr("nvflare.app_common.launchers.subprocess_launcher.subprocess.Popen", _Proc)
        monkeypatch.setenv("TEST_SECRET_REF_VAR", "resolved secret")
        launcher = SubprocessLauncher(script, launch_once=False)
        launcher._app_dir = str(tmp_path)

        launcher.launch_task(
            "__test_task", DXO(DataKind.WEIGHTS, {}).to_shareable(), self._make_fl_ctx(str(tmp_path)), Signal()
        )
        launcher._log_thread.join()

        assert popen_calls[0]["args"][0] == expected_command

    def test_clean_up_rejects_secret_ref_in_nested_command(self, monkeypatch):
        popen = Mock()
        launcher = SubprocessLauncher(
            "python train.py",
            launch_once=False,
            clean_up_script="env FOO=bar bash -c 'echo ${secret:TEST_SECRET_REF_VAR}'",
        )
        launcher._process = Mock()
        launcher._log_thread = Mock()
        launcher._terminate_process = Mock()
        monkeypatch.setattr("nvflare.app_common.launchers.subprocess_launcher.subprocess.Popen", popen)

        with pytest.raises(ValueError, match="nested interpreter command strings"):
            launcher._stop_external_process()

        popen.assert_not_called()

    def test_stop_external_process_terminates_posix_process_group(self, monkeypatch):
        killpg_calls = []

        class _Proc:
            pid = 1234

            def wait(self, timeout=None):
                raise subprocess.TimeoutExpired("python train.py", timeout)

        launcher = SubprocessLauncher("python train.py", launch_once=False)
        launcher._process = _Proc()
        launcher._log_thread = Mock()
        monkeypatch.setattr("nvflare.app_common.launchers.subprocess_launcher.os.name", "posix")
        monkeypatch.setattr(
            "nvflare.app_common.launchers.subprocess_launcher.os.killpg",
            lambda pid, sig: killpg_calls.append((pid, sig)),
        )

        launcher._stop_external_process()

        assert killpg_calls == [(1234, signal_module.SIGTERM)]
        launcher._log_thread.join.assert_called_once()
        assert launcher._process is None

    def test_log_subprocess_output(self):
        class _Proc:
            pass

        p = _Proc()
        p.stdout = BufferedReader(BytesIO(b"line1\nline2\r\npartial"))
        logger = Mock()
        log_subprocess_output(p, logger)

        logged = [call.args[0] for call in logger.info.call_args_list]
        assert logged == ["line1", "line2", "partial"]

    def test_log_subprocess_output_replaces_malformed_utf8_and_continues_draining(self):
        class _Proc:
            pass

        p = _Proc()
        p.stdout = BufferedReader(BytesIO(b"malformed \xff line\nlater line\nmalformed tail \xfe"))
        logger = Mock()

        log_subprocess_output(p, logger)

        logged = [call.args[0] for call in logger.info.call_args_list]
        assert logged == ["malformed \ufffd line", "later line", "malformed tail \ufffd"]

    def test_log_subprocess_output_continues_draining_after_logger_failure(self):
        class _Proc:
            pass

        p = _Proc()
        p.stdout = BufferedReader(BytesIO(b"first\nsecond\n"))
        logger = Mock()
        logger.info.side_effect = [RuntimeError("logger failed"), None]

        log_subprocess_output(p, logger)

        assert [call.args[0] for call in logger.info.call_args_list] == ["first", "second"]

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
