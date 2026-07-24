# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import logging
import os
import re
import signal
import subprocess
from typing import List, Optional, Sequence, Union

from nvflare.fuel.utils.secret_utils import has_secret_refs, resolve_secret_refs, split_command_preserving_secret_refs

log = logging.getLogger(__name__)

_POSIX_SPAWN_SUPPORTED = hasattr(os, "posix_spawn") and os.name == "posix"

# Matches the start of a formatted NVFlare log line after stripping ANSI color
# codes: "YYYY-MM-DD HH:MM:SS" produced by BaseFormatter / ColorFormatter.
# Lines from a subprocess consoleHandler match this; raw print() lines do not.
_ANSI_ESC_RE = re.compile(r"\x1b\[[0-9;]*m")
_LOG_LINE_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
_SHELL_COMMAND_INTERPRETERS = frozenset({"ash", "bash", "dash", "fish", "ksh", "mksh", "sh", "zsh"})
_ENV_COMMAND_WRAPPERS = frozenset({"env"})
_COMMAND_MULTIPLEXERS = frozenset({"busybox"})
_SHELL_OPTIONS_WITH_VALUE = frozenset({"-o", "-O", "--init-file", "--rcfile"})
_PYTHON_INTERPRETER_RE = re.compile(r"^(?:python|pypy)(?:\d+(?:\.\d+)*)?$")


def _command_basename(command: str) -> str:
    """Return a command basename."""
    return command.rsplit("/", maxsplit=1)[-1]


def _raise_nested_command_secret_ref(interpreter: str, option: str) -> None:
    detail = f"{interpreter} {option}"
    raise ValueError(f"secret references are not supported in nested interpreter command strings ({detail})")


def _unwrap_env_commands(command_seq: list[str]) -> list[str]:
    """Unwrap simple leading env commands without guessing option operands."""
    while command_seq and _command_basename(command_seq[0]).casefold() in _ENV_COMMAND_WRAPPERS:
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
    normalized_interpreter = interpreter.casefold()
    index = 1
    while index < len(command_seq):
        option = command_seq[index]
        if option == "--" or not option.startswith("-"):
            return
        # POSIX shells allow short options to be combined, for example ``bash -lc``.
        is_command_option = not option.startswith("--") and (
            "c" in option[1:] or (normalized_interpreter == "fish" and "C" in option[1:])
        )
        if is_command_option:
            command_index = index + 1
            if has_secret_refs(option) or (
                command_index < len(command_seq) and has_secret_refs(command_seq[command_index])
            ):
                _raise_nested_command_secret_ref(interpreter, option)
            return
        index += 2 if option in _SHELL_OPTIONS_WITH_VALUE else 1


def _reject_python_code_refs(command_seq: list[str], interpreter: str) -> None:
    for index, option in enumerate(command_seq[1:], start=1):
        if option == "--" or not option.startswith("-"):
            return
        if option == "-c" or option.startswith("-c"):
            if has_secret_refs(option) or (index + 1 < len(command_seq) and has_secret_refs(command_seq[index + 1])):
                _raise_nested_command_secret_ref(interpreter, option)
            return


def _reject_secret_refs_in_nested_command(command_seq: list[str]) -> None:
    """Reject refs in code strings for direct or explicitly env-wrapped shell interpreters."""
    command_seq = _unwrap_env_commands(command_seq)
    if not command_seq:
        return

    interpreter = _command_basename(command_seq[0])
    normalized_interpreter = interpreter.casefold()
    if normalized_interpreter in _COMMAND_MULTIPLEXERS and len(command_seq) > 1:
        command_seq = command_seq[1:]
        interpreter = _command_basename(command_seq[0])
        normalized_interpreter = interpreter.casefold()
    if normalized_interpreter in _SHELL_COMMAND_INTERPRETERS:
        _reject_shell_command_refs(command_seq, interpreter)
    elif _PYTHON_INTERPRETER_RE.fullmatch(normalized_interpreter):
        _reject_python_code_refs(command_seq, interpreter)


def prepare_subprocess_command(command: Union[str, Sequence[str]]) -> list[str]:
    """Build argv for a shell-free subprocess command and resolve secret references safely.

    The command is split before references are resolved, so a secret containing spaces or
    command-line metacharacters remains one argv element. References inside recognized nested
    interpreter command strings are rejected because those strings are parsed a second time.

    Args:
        command: Command string or pre-tokenized argv from job configuration.

    Returns:
        A resolved argv list suitable for ``subprocess.Popen(..., shell=False)``.
    """
    if isinstance(command, str):
        command_seq = split_command_preserving_secret_refs(command, posix=True)
    else:
        command_seq = list(command)
        if not command_seq or not all(isinstance(arg, str) for arg in command_seq):
            raise ValueError("command argv must be a non-empty sequence of strings")
    _reject_secret_refs_in_nested_command(command_seq)
    return [resolve_secret_refs(token) for token in command_seq]


def _get_line(buffer: bytearray):
    """Read one line from a binary buffer, accepting every CR/LF combination."""
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

    # If CR and LF are adjacent, treat them as one line break.
    if abs(r - n) == 1:
        index += 1

    line = buffer[:index].decode(errors="replace").rstrip()
    remaining = bytearray() if index >= size - 1 else buffer[index + 1 :]
    return line, remaining


def _route_subprocess_line(line: str, logger) -> None:
    """Route one subprocess stdout line to interactive output or the NVFlare logger."""
    plain = _ANSI_ESC_RE.sub("", line)
    if _LOG_LINE_RE.match(plain):
        print(line)
    else:
        logger.info(line)


def _safe_route_subprocess_line(line: str, logger) -> None:
    try:
        _route_subprocess_line(line, logger)
    except Exception:
        # Output routing must not stop the pipe drain and deadlock the child.
        pass


def log_subprocess_output(process, logger) -> None:
    """Drain a subprocess's merged stdout/stderr stream and route complete lines."""
    buffer = bytearray()
    while True:
        chunk = process.stdout.read1(4096)
        if not chunk:
            break
        buffer += chunk

        while True:
            line, buffer = _get_line(buffer)
            if line is None:
                break
            if line:
                _safe_route_subprocess_line(line, logger)

    if buffer:
        _safe_route_subprocess_line(buffer.decode(errors="replace"), logger)


class ProcessAdapter:
    def __init__(self, process: Optional[subprocess.Popen] = None, pid: Optional[int] = None):
        """Adapter to manage a process, whether created via subprocess.Popen or os.posix_spawn.

        Args:
            process: The subprocess.Popen object (if created via subprocess)
            pid: The process ID (if created via posix_spawn, or fallback for process.pid)
        """
        self.process = process
        self.pid = pid if pid is not None else (process.pid if process else None)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._return_code: Optional[int] = None

        if self.pid is None:
            raise ValueError("ProcessAdapter requires either a process object or a pid.")

    def terminate(self) -> None:
        """Terminate the process group.

        Sends SIGKILL to the entire process group. No need to call process.terminate()
        separately since SIGKILL already terminates all processes in the group.
        """
        self._kill_process_group()

    def poll(self) -> Optional[int]:
        """Check if the process has terminated.

        Returns:
            None if process is still running, otherwise the exit code.
        """
        if self.process:
            return self.process.poll()

        return self._poll_pid()

    def wait(self) -> None:
        """Wait for the process to terminate."""
        if self.process:
            self.process.wait()
            return

        if self.pid is None:
            return

        if self._return_code is None:
            try:
                _, status = os.waitpid(self.pid, 0)
                self._return_code = self._decode_status(status)
            except ChildProcessError:
                pass

    def _poll_pid(self) -> Optional[int]:
        if self.pid is None:
            return None

        if self._return_code is not None:
            return self._return_code

        try:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
        except ChildProcessError:
            # Process already reaped or doesn't exist, treat as terminated
            if self._return_code is None:
                self._return_code = -1
            return self._return_code

        if pid == 0:
            return None

        self._return_code = self._decode_status(status)
        return self._return_code

    def _decode_status(self, status: int) -> int:
        if hasattr(os, "waitstatus_to_exitcode"):
            return os.waitstatus_to_exitcode(status)

        if os.WIFEXITED(status):
            return os.WEXITSTATUS(status)
        if os.WIFSIGNALED(status):
            return -os.WTERMSIG(status)
        # Fallback/Error case
        return -1

    def _kill_process_group(self):
        if self.pid is None:
            return

        if not hasattr(os, "killpg") or not hasattr(os, "getpgid"):
            return

        try:
            pgid = os.getpgid(self.pid)
        except ProcessLookupError:
            # Process already gone; nothing left to terminate.
            return
        except PermissionError as exc:
            self.logger.warning("Unable to read pgid for %s (%s)", self.pid, exc)
            pgid = self.pid

        try:
            os.killpg(pgid, signal.SIGKILL)
            self.logger.debug("kill signal sent")
        except ProcessLookupError:
            # Group already terminated, treat as success.
            return
        except Exception as exc:
            self.logger.warning("Failed to kill process group %s (%s)", pgid, exc)


def spawn_process(cmd_args: List[str], env: dict) -> ProcessAdapter:
    """Launch a process using posix_spawn if available, falling back to subprocess.Popen.

    This method attempts to use os.posix_spawn with setsid=True to avoid fork() related issues
    (such as gRPC deadlocks). If posix_spawn is unavailable or fails, it falls back to
    subprocess.Popen with preexec_fn=os.setsid.

    Args:
        cmd_args: The command arguments as a list of strings.
        env: The environment variables dictionary.

    Returns:
        ProcessAdapter: An adapter wrapping the launched process.
    """
    if _POSIX_SPAWN_SUPPORTED and cmd_args:
        try:
            # Note: 'setsid' is a potential extension or patch in some python environments.
            # We wrap it in try-except to gracefully fallback if not supported.
            path = cmd_args[0]
            pid = os.posix_spawn(path, cmd_args, env, setsid=True)
            log.info("Launch the job in process ID: %s (posix_spawn)", pid)
            return ProcessAdapter(pid=pid)
        except TypeError as exc:
            # Happens when this interpreter lacks posix_spawn(..., setsid=...) support and silently falls back to fork.
            log.warning("posix_spawn missing setsid support (%s); falling back to subprocess.", exc)
        except Exception as exc:
            # Covers launch failures unrelated to setsid (e.g. binary missing, permission issues).
            log.warning("posix_spawn failed (%s); falling back to subprocess.", exc)

    preexec_fn = os.setsid if hasattr(os, "setsid") else None
    process = subprocess.Popen(cmd_args, shell=False, preexec_fn=preexec_fn, env=env)
    log.info("Launch the job in process ID: %s (subprocess)", process.pid)

    return ProcessAdapter(process=process)
