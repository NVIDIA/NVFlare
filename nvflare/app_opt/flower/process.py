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

_OWNER_ENV = "NVFLARE_FLOWER_PROCESS_OWNER"


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
            self.process = subprocess.Popen(
                shlex.split(self.cmd_desc.cmd),
                cwd=self.cmd_desc.cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                bufsize=0,
            )
            try:
                self._root = psutil.Process(self.process.pid)
            except psutil.NoSuchProcess:
                pass
            os.set_blocking(self.process.stdout.fileno(), False)
            self._monitor = threading.Thread(target=self._monitor_descendants)
            self._monitor.start()
            self._log_writer = threading.Thread(target=self._write_log)
            self._log_writer.start()
        except BaseException:
            self.stop(timeout=0.1)
            raise

    def _owned_descendants(self):
        with self._descendant_lock:
            return tuple(self._descendants)

    def _capture_descendants(self, include_orphans=False):
        found, gone = set(), set()
        roots = self._owned_descendants() + ((self._root,) if self._root else ())
        for process in roots:
            try:
                if not _is_alive(process):
                    gone.add(process)
                    continue
                found.update(process.children(recursive=True))
            except psutil.NoSuchProcess:
                gone.add(process)
        if include_orphans:
            # The inherited marker also finds workers whose parent exited
            # between snapshots. Never identify workers by executable name.
            for process in psutil.process_iter():
                try:
                    if process.pid != self.process.pid and process.environ().get(_OWNER_ENV) == self._owner:
                        found.add(process)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        with self._descendant_lock:
            self._descendants.difference_update(gone)
            self._descendants.update(found)

    def _monitor_descendants(self):
        try:
            while not self._monitor_stop.is_set():
                self._capture_descendants()
                self._monitor_stop.wait(0.1)
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
            if self.process:
                attempt(lambda: self._capture_descendants(include_orphans=True))
                # Give Flower a chance to reap its own workers first.
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
                    if not alive:
                        break
                    if killed_once and time.monotonic() >= deadline:
                        errors.append(RuntimeError(f"Flower workers survived cleanup: {[p.pid for p in alive]}"))
                        break
                    # Retained psutil identities protect against PID reuse.
                    # One failed kill must not prevent attempts on other workers.
                    for child in alive:
                        attempt(child.kill)
                    killed_once = True
                    time.sleep(0.05)
        finally:
            self._monitor_stop.set()
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
