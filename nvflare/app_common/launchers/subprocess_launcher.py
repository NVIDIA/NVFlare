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

import os
import re
import signal
import subprocess
from threading import Lock, Thread
from typing import Optional

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.launcher import Launcher, LauncherRunStatus
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.fuel.utils.secret_utils import has_secret_refs, resolve_secret_refs, split_command_preserving_secret_refs
from nvflare.utils.job_launcher_utils import add_custom_dir_to_path


def get_line(buffer: bytearray):
    """Read a line from the binary buffer. It treats all combinations of \n and \r as line breaks.

    Args:
        buffer: A binary buffer

    Returns:
        (line, remaining): Return the first line as str and the remaining buffer.
        line is None if no newline found

    """
    size = len(buffer)
    r = buffer.find(b"\r")
    if r < 0:
        r = size + 1
    n = buffer.find(b"\n")
    if n < 0:
        n = size + 1
    index = min(r, n)

    if index >= size:
        return None, buffer

    # if \r and \n are adjacent, treat them as one
    if abs(r - n) == 1:
        index = index + 1

    line = buffer[:index].decode().rstrip()
    if index >= size - 1:
        remaining = bytearray()
    else:
        remaining = buffer[index + 1 :]
    return line, remaining


# Matches the start of a formatted NVFlare log line after stripping ANSI color
# codes: "YYYY-MM-DD HH:MM:SS" produced by BaseFormatter / ColorFormatter.
# Lines from the subprocess consoleHandler match this; raw print() lines do not.
_ANSI_ESC_RE = re.compile(r"\x1b\[[0-9;]*m")
_LOG_LINE_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
_SHELL_COMMAND_INTERPRETERS = frozenset({"sh", "bash"})
_POWERSHELL_COMMAND_INTERPRETERS = frozenset({"powershell", "pwsh"})
_ENV_COMMAND_WRAPPERS = frozenset({"env"})
_SHELL_OPTIONS_WITH_VALUE = frozenset({"-o", "-O", "--init-file", "--rcfile"})


def _command_basename(command: str) -> str:
    """Return a command basename for either POSIX or Windows-style paths."""
    return command.replace("\\", "/").rsplit("/", maxsplit=1)[-1]


def _raise_nested_command_secret_ref(interpreter: str, option: str) -> None:
    detail = f"{interpreter} {option}"
    raise ValueError(f"secret references are not supported in nested interpreter command strings ({detail})")


def _unwrap_env_commands(command_seq: list[str]) -> list[str]:
    """Unwrap simple leading env commands without guessing option operands."""
    while command_seq and _command_basename(command_seq[0]).casefold().removesuffix(".exe") in _ENV_COMMAND_WRAPPERS:
        interpreter = _command_basename(command_seq[0])
        command_seq = command_seq[1:]
        parse_options = True
        while command_seq:
            option = command_seq[0]
            if parse_options and option == "--":
                command_seq = command_seq[1:]
                parse_options = False
                continue
            if "=" in option and not option.startswith(("-", "=")):
                command_seq = command_seq[1:]
                continue
            if not parse_options:
                break
            if option in {"-i", "--ignore-environment"}:
                command_seq = command_seq[1:]
                continue
            if option.startswith("-"):
                if any(has_secret_refs(arg) for arg in command_seq):
                    _raise_nested_command_secret_ref(interpreter, option)
                return []
            break
    return command_seq


def _reject_shell_command_refs(command_seq: list[str], interpreter: str) -> None:
    index = 1
    while index < len(command_seq):
        option = command_seq[index]
        if option == "--" or not option.startswith("-"):
            return
        # POSIX shells allow short options to be combined, for example ``bash -lc``.
        if not option.startswith("--") and "c" in option[1:]:
            command_index = index + 1
            if has_secret_refs(option) or (
                command_index < len(command_seq) and has_secret_refs(command_seq[command_index])
            ):
                _raise_nested_command_secret_ref(interpreter, option)
            return
        index += 2 if option in _SHELL_OPTIONS_WITH_VALUE else 1


def _reject_powershell_code_refs(command_seq: list[str], interpreter: str) -> None:
    for index, option in enumerate(command_seq[1:], start=1):
        normalized_option = option.casefold()
        if option == "--" or not option.startswith("-") or normalized_option in {"-file", "-f"}:
            return
        if normalized_option in {"-command", "-c"}:
            if any(has_secret_refs(arg) for arg in command_seq[index + 1 :]):
                _raise_nested_command_secret_ref(interpreter, option)
            return
        if normalized_option in {"-encodedcommand", "-e", "-ec", "-enc"}:
            if index + 1 < len(command_seq) and has_secret_refs(command_seq[index + 1]):
                _raise_nested_command_secret_ref(interpreter, option)
            return
        if any(has_secret_refs(arg) for arg in command_seq[index:]):
            _raise_nested_command_secret_ref(interpreter, option)
        return


def _reject_secret_refs_in_nested_command(command_seq: list[str]) -> None:
    """Reject refs in code strings for direct or explicitly env-wrapped shell interpreters."""
    command_seq = _unwrap_env_commands(command_seq)
    if not command_seq:
        return

    interpreter = _command_basename(command_seq[0])
    normalized_interpreter = interpreter.casefold().removesuffix(".exe")
    if normalized_interpreter in _SHELL_COMMAND_INTERPRETERS:
        _reject_shell_command_refs(command_seq, interpreter)
    elif normalized_interpreter in _POWERSHELL_COMMAND_INTERPRETERS:
        _reject_powershell_code_refs(command_seq, interpreter)


def _prepare_command(command: str) -> list[str]:
    command_seq = split_command_preserving_secret_refs(command, posix=True)
    _reject_secret_refs_in_nested_command(command_seq)
    return [resolve_secret_refs(token) for token in command_seq]


def _route_subprocess_line(line: str, logger) -> None:
    """Route one stdout line from the subprocess to the right destination.

    Formatted log lines (from the subprocess consoleHandler) are already written
    to the shared log files by the subprocess file handler, so we just print them
    to the terminal for interactive visibility.  Raw print() lines from user
    training scripts have no timestamp, so we wrap them with logger.info() to
    ensure they reach both the terminal and log.txt.
    """
    plain = _ANSI_ESC_RE.sub("", line)
    if _LOG_LINE_RE.match(plain):
        print(line)
    else:
        logger.info(line)


def log_subprocess_output(process, logger):

    buffer = bytearray()
    while True:
        chunk = process.stdout.read1(4096)
        if not chunk:
            break
        buffer = buffer + chunk

        while True:
            line, buffer = get_line(buffer)
            if line is None:
                break

            if line:
                _route_subprocess_line(line, logger)

    if buffer:
        _route_subprocess_line(buffer.decode(), logger)


class SubprocessLauncher(Launcher):
    def __init__(
        self,
        script: str,
        launch_once: Optional[bool] = True,
        clean_up_script: Optional[str] = None,
        shutdown_timeout: Optional[float] = 0.0,
    ):
        """Initializes the SubprocessLauncher.

        Args:
            script (str): Script to be launched using subprocess.
            launch_once (bool): Whether the external process will be launched only once at the beginning or on each task.
            clean_up_script (Optional[str]): Optional clean up script to be run after the main script execution.
            shutdown_timeout (float): If provided, will wait for this number of seconds before shutdown.
        """
        super().__init__()

        self._app_dir = None
        self._process = None
        self._script = script
        self._launch_once = launch_once
        self._clean_up_script = clean_up_script
        self._shutdown_timeout = shutdown_timeout
        self._lock = Lock()
        self.logger = get_obj_logger(self)

    def initialize(self, fl_ctx: FLContext):
        self._app_dir = self.get_app_dir(fl_ctx)
        if self._launch_once:
            self._start_external_process(fl_ctx)

    def finalize(self, fl_ctx: FLContext) -> None:
        if self._launch_once and self._process:
            self._stop_external_process()

    def needs_deferred_stop(self) -> bool:
        return not self._launch_once

    def launch_task(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> bool:
        if not self._launch_once:
            self._start_external_process(fl_ctx)
        return True

    def stop_task(self, task_name: str, fl_ctx: FLContext, abort_signal: Signal) -> None:
        if not self._launch_once:
            self._stop_external_process()

    def _start_external_process(self, fl_ctx: FLContext):
        with self._lock:
            if self._process is None:
                self.logger.info("_start_external_process: launching new subprocess")
                command = self._script
                env = os.environ.copy()
                env["CLIENT_API_TYPE"] = "EX_PROCESS_API"

                workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
                job_id = fl_ctx.get_prop(FLContextKey.CURRENT_JOB_ID)
                app_custom_folder = workspace.get_app_custom_dir(job_id)
                add_custom_dir_to_path(app_custom_folder, env)

                # Resolve ${secret:ENV_VAR} references from this site's environment after shlex
                # splitting, so injected values never re-tokenize. References in supported nested
                # shell command strings are rejected because those strings are parsed again.
                # Resolved values exist only in the subprocess argv and must never be logged.
                command_seq = _prepare_command(command)
                self._process = subprocess.Popen(
                    command_seq,
                    shell=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=self._app_dir,
                    env=env,
                    start_new_session=os.name == "posix",
                )
                self._log_thread = Thread(target=log_subprocess_output, args=(self._process, self.logger))
                self._log_thread.start()

    def _terminate_process(self):
        if os.name == "posix":
            try:
                os.killpg(self._process.pid, signal.SIGTERM)
                return
            except ProcessLookupError:
                return
            except Exception as e:
                self.logger.debug(f"failed to terminate subprocess process group: {e}")
        self._process.terminate()

    def _stop_external_process(self):
        with self._lock:
            if self._process:
                try:
                    self._process.wait(self._shutdown_timeout)
                except subprocess.TimeoutExpired:
                    pass
                self.logger.info(f"_stop_external_process: terminating pid={self._process.pid}")
                self._terminate_process()
                self._log_thread.join()
                if self._clean_up_script:
                    command_seq = _prepare_command(self._clean_up_script)
                    process = subprocess.Popen(command_seq, cwd=self._app_dir, shell=False)
                    process.wait()
                self._process = None

    def check_run_status(self, task_name: str, fl_ctx: FLContext) -> str:
        with self._lock:
            if self._process is None:
                return LauncherRunStatus.NOT_RUNNING
            return_code = self._process.poll()
            if return_code is None:
                return LauncherRunStatus.RUNNING
            if return_code == 0:
                return LauncherRunStatus.COMPLETE_SUCCESS
            return LauncherRunStatus.COMPLETE_FAILED
