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

"""Owned Flower process groups and bounded CLI calls.

Flower starts SuperExec and app workers. Put each SuperLink/SuperNode in its own
session so shutdown and failed startup also clean up descendants.
"""

import json
import logging
import os
import selectors
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid

import psutil

from nvflare.app_common.tie.process_mgr import ProcessManager
from nvflare.app_opt.flower.process_guard import OWNER_ENV as _OWNER_ENV
from nvflare.app_opt.flower.process_guard import capture_owned


class FlowerProcessManager(ProcessManager):
    def __init__(self, cmd_desc, stop_method="term"):
        super().__init__(cmd_desc, stop_method)
        self._owner = uuid.uuid4().hex
        self._root = None
        self._descendants = set()
        self._descendant_lock = threading.Lock()
        self._monitor_stop = threading.Event()
        self._log_stop = threading.Event()
        self._monitor = None
        self._log_writer = None
        self._monitor_error = None
        self._log_error = None
        self._scan_errors = []
        self._guard = None
        self._spawn_fence = None

    def start(self, fl_ctx):
        if os.name != "posix":
            raise RuntimeError("The Flower HTTP integration requires a POSIX host")
        ws = fl_ctx.get_engine().get_workspace()
        run_dir = ws.get_run_dir(fl_ctx.get_job_id())
        os.makedirs(run_dir, exist_ok=True)
        if self.cmd_desc.log_file_name:
            self.log_file = open(os.path.join(run_dir, self.cmd_desc.log_file_name), "a")
        self.msg_prefix = f"[{self.cmd_desc.stdout_msg_prefix}@{fl_ctx.get_identity_name()}]"
        env = os.environ.copy()
        env.update(self.cmd_desc.env or {})
        env[_OWNER_ENV] = self._owner
        try:
            self._start_guard()
            self.process = subprocess.Popen(
                # The first exec installs the ownership marker. Keep the
                # launch fence open until then, including across close_fds.
                [
                    sys.executable,
                    "-c",
                    "import os,sys; os.close(int(sys.argv[1])); os.execvpe(sys.argv[2],sys.argv[2:],os.environ)",
                    str(self._spawn_fence),
                    *shlex.split(self.cmd_desc.cmd),
                ],
                cwd=self.cmd_desc.cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                bufsize=0,
                pass_fds=(self._spawn_fence,),
            )
            try:
                self._root = psutil.Process(self.process.pid)
                self._guard.stdin.write(json.dumps((self._root.pid, self._root.create_time())).encode() + b"\n")
                self._guard.stdin.flush()
            except psutil.NoSuchProcess:
                pass
            self._close_spawn_fence()
            os.set_blocking(self.process.stdout.fileno(), False)
            self._monitor = threading.Thread(target=self._monitor_descendants)
            self._monitor.start()
            self._log_writer = threading.Thread(target=self._write_log)
            self._log_writer.start()
        except BaseException as error:
            try:
                self.stop(timeout=0.1)
            except Exception as cleanup_error:
                raise error from cleanup_error
            raise

    def _start_guard(self):
        # Establish independent ownership before any Flower process can start.
        # The target inherits neither the control pipe nor the guard's session.
        read_fence, self._spawn_fence = os.pipe()
        try:
            self._guard = subprocess.Popen(
                [
                    sys.executable,
                    os.path.join(os.path.dirname(__file__), "process_guard.py"),
                    self._owner,
                    str(read_fence),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log_file or subprocess.DEVNULL,
                start_new_session=True,
                bufsize=0,
                pass_fds=(read_fence,),
            )
        finally:
            os.close(read_fence)
        with selectors.DefaultSelector() as selector:
            selector.register(self._guard.stdout, selectors.EVENT_READ)
            if not selector.select(30) or self._guard.stdout.readline() != b"ready\n":
                raise RuntimeError("Flower process guard did not become ready")

    def _stop_guard(self):
        self._close_spawn_fence()
        if self._guard:
            guard = self._guard
            try:
                guard.stdin.close()
                try:
                    code = guard.wait(timeout=10)
                except subprocess.TimeoutExpired as error:
                    guard.terminate()
                    try:
                        guard.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        guard.kill()
                        guard.wait(timeout=1)
                    raise RuntimeError("Flower process guard timed out and was terminated") from error
                if code:
                    raise RuntimeError(f"Flower process guard failed: {code}")
            finally:
                guard.stdout.close()
                if guard.poll() is not None:
                    self._guard = None

    def _close_spawn_fence(self):
        if self._spawn_fence is not None:
            os.close(self._spawn_fence)
            self._spawn_fence = None

    def _owned_descendants(self):
        with self._descendant_lock:
            return tuple(self._descendants)

    def _capture_descendants(self, include_orphans=False):
        with self._descendant_lock:
            roots = tuple(self._descendants) + ((self._root,) if self._root else ())
            errors = []
            found = capture_owned(roots, self._owner if include_orphans else None, errors)
            found.discard(self._root)
            self._descendants = found
            self._scan_errors = errors

    def _monitor_descendants(self):
        try:
            while not self._monitor_stop.is_set():
                try:
                    self._capture_descendants()
                except (psutil.Error, OSError):
                    logging.getLogger(__name__).exception(
                        "Flower monitor scan failed; retaining ownership and retrying"
                    )
                self._monitor_stop.wait(0.5)
        except Exception as error:
            self._monitor_error = error

    def _write_log(self):
        # readline() can wait forever when a surviving detached worker holds
        # the pipe open, including when it only wrote a partial line.
        pending = b""
        try:
            fd = self.process.stdout.fileno()
            with selectors.DefaultSelector() as selector:
                selector.register(fd, selectors.EVENT_READ)
                while not self._log_stop.is_set():
                    if not selector.select(timeout=0.1):
                        continue
                    chunk = os.read(fd, 65536)
                    if not chunk:
                        break
                    lines = (pending + chunk).split(b"\n")
                    pending = lines.pop()
                    for line in lines:
                        self._emit_log(line + b"\n")
            if pending:
                self._emit_log(pending)
        except Exception as error:
            self._log_error = error

    def _emit_log(self, data):
        line = data.decode("utf-8", errors="replace")
        with self.file_lock:
            if self.log_file:
                self.log_file.write(line)
                self.log_file.flush()
        if self.cmd_desc.log_stdout:
            if self.msg_prefix and not line.startswith("\r"):
                line = f"{self.msg_prefix} {line}"
            sys.stdout.write(line)
            sys.stdout.flush()

    def _signal_group(self, sig):
        # After reaping the leader, signal retained process identities instead
        # of risking a reused process-group ID.
        if self.process.poll() is not None:
            return
        try:
            os.killpg(self.process.pid, sig)
        except ProcessLookupError:
            pass

    def stop(self, timeout=5.0):
        errors = []
        timeout = max(timeout, 0.1)

        def attempt(action):
            try:
                action()
            except (psutil.NoSuchProcess, ProcessLookupError):
                pass
            except Exception as error:
                errors.append(error)

        try:
            # Let the stop path own discovery so a monitor scan cannot overwrite
            # an incomplete orphan scan before its result is reported.
            self._monitor_stop.set()
            if self._monitor and self._monitor.ident is not None:
                attempt(lambda: self._monitor.join(timeout=5))
            if self.process:
                attempt(lambda: self._capture_descendants(include_orphans=True))
                # Give Flower a chance to reap its own workers first.
                leader_alive = self.process.poll() is None
                attempt(lambda: self._signal_group(signal.SIGTERM))
                try:
                    self.process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    pass
                except Exception as error:
                    errors.append(error)
                attempt(lambda: self._capture_descendants(include_orphans=True))
                attempt(lambda: self._signal_group(signal.SIGKILL))
                attempt(lambda: self.process.wait(timeout=timeout))
                if leader_alive:
                    attempt(lambda: self._wait_for_descendants(timeout))
                # A worker may create another detached process just before it
                # is killed. Rescan after each kill pass, including after the
                # deadline, so unknown survivors cannot count as success.
                deadline = time.monotonic() + max(timeout, 1.0)
                killed_once = False
                while True:
                    attempt(lambda: self._capture_descendants(include_orphans=True))
                    alive = []
                    for child in self._owned_descendants():
                        try:
                            if _is_alive(child):
                                alive.append(child)
                        except Exception as error:
                            errors.append(error)
                    if not alive and not self._scan_errors:
                        break
                    if killed_once and time.monotonic() >= deadline:
                        if alive:
                            errors.append(RuntimeError(f"Flower workers survived cleanup: {[p.pid for p in alive]}"))
                        errors.extend(self._scan_errors)
                        break
                    # Retained psutil identities protect against PID reuse.
                    # One failed kill must not prevent attempts on other workers.
                    for child in alive:
                        attempt(child.kill)
                    killed_once = True
                    time.sleep(0.05)
        finally:
            self._monitor_stop.set()
            attempt(self._stop_guard)
            # Allow normal EOF to drain remaining output, then explicitly
            # release the reader even if killing a worker failed.
            if self._log_writer and self._log_writer.ident is not None:
                attempt(lambda: self._log_writer.join(timeout=1))
            self._log_stop.set()
            for worker in (self._monitor, self._log_writer):
                if worker and worker.ident is not None:
                    attempt(lambda worker=worker: worker.join(timeout=5))
                    if worker.is_alive():
                        errors.append(RuntimeError("Flower process management thread did not stop"))
            if self.process and self.process.stdout:
                attempt(self.process.stdout.close)
            if self.log_file:
                with self.file_lock:
                    attempt(self.log_file.close)
                    self.log_file = None
        errors.extend(error for error in (self._monitor_error, self._log_error) if error is not None)
        self._monitor_error = self._log_error = None
        if errors:
            raise RuntimeError(f"Flower process cleanup failed: {errors}") from errors[0]
        return self.process.returncode if self.process else 0

    def _wait_for_descendants(self, timeout):
        deadline = time.monotonic() + timeout
        while any(_is_alive(p) for p in self._owned_descendants()) and time.monotonic() < deadline:
            time.sleep(0.05)


def start_process(cmd_desc, fl_ctx, stop_method="term"):
    manager = FlowerProcessManager(cmd_desc, stop_method)
    manager.start(fl_ctx)
    return manager


def run_command(cmd_desc):
    env = os.environ.copy()
    env.update(cmd_desc.env or {})
    result = subprocess.run(
        shlex.split(cmd_desc.cmd),
        cwd=cmd_desc.cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
        check=True,
    )
    return result.stdout


def _is_alive(process):
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
