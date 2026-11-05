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

"""Bounded ownership of test subprocesses, including detached descendants."""

import errno
import os
import subprocess
import tempfile
import time
import uuid

import psutil

_OWNER_ENV = "NVFLARE_TEST_PROCESS_OWNER"


def process_running(process):
    try:
        return process.is_running() and process.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
    except psutil.NoSuchProcess:
        return False


def wait_for_processes(processes, timeout):
    """Keep process identities after a Linux pidfd wait failure; never infer exit from it."""
    processes = set(processes)
    deadline = time.monotonic() + timeout
    try:
        return psutil.wait_procs(processes, timeout=timeout)
    except OSError as error:
        if error.errno != errno.EINVAL:
            raise
    while True:
        alive = [process for process in processes if process_running(process)]
        remaining = max(0.0, deadline - time.monotonic())
        if not alive or not remaining:
            return list(processes.difference(alive)), alive
        time.sleep(min(0.05, remaining))


class OwnedProcesses:
    """Register before launch and retain identities through exit, detachment and failure.

    Children inherit a unique environment marker, allowing cleanup to find them
    even after their parent exits. Snapshot descendants while roots are live as
    well, since trainers may replace their environment. Only owned identities
    are signalled. Popen owners reap the direct children.
    """

    def __init__(self, grace=5.0, process_factory=None):
        self.grace = grace
        self.process_factory = process_factory or subprocess.Popen
        self.token = uuid.uuid4().hex
        self.processes = []
        self.identities = set()
        self.callbacks = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.close()
        except Exception as cleanup_error:
            if exc_value is not None:
                exc_value.cleanup_error = str(cleanup_error)
                raise exc_value from cleanup_error
            raise

    def start(self, args, *, env=None, **kwargs):
        environment = dict(os.environ if env is None else env)
        environment[_OWNER_ENV] = self.token
        process = self.process_factory(args, env=environment, start_new_session=os.name == "posix", **kwargs)
        self.processes.append(process)
        return process

    def capture(self, include_orphans=False):
        for process in self.processes:
            if process.poll() is None:
                try:
                    self.identities.add(psutil.Process(process.pid))
                except psutil.NoSuchProcess:
                    pass
        if include_orphans:
            for process in psutil.process_iter():
                try:
                    if process.environ().get(_OWNER_ENV) == self.token:
                        self.identities.add(process)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        for process in list(self.identities):
            try:
                self.identities.update(process.children(recursive=True))
            except psutil.NoSuchProcess:
                pass

    def close(self):
        errors = []

        def attempt(action):
            try:
                action()
            except Exception as error:
                errors.append(error)

        callbacks, self.callbacks = self.callbacks, []
        for callback in callbacks:
            attempt(callback)
        for action in ("terminate", "kill"):
            attempt(lambda: self.capture(include_orphans=True))
            for process in list(self.identities):
                try:
                    if process_running(process):
                        getattr(process, action)()
                except psutil.NoSuchProcess:
                    pass
                except Exception as error:
                    errors.append(error)
            # Also stop roots if identity capture itself failed.
            for process in self.processes:
                try:
                    if process.poll() is None:
                        getattr(process, action)()
                except ProcessLookupError:
                    pass
                except Exception as error:
                    errors.append(error)
            deadline = time.monotonic() + self.grace
            for process in self.processes:
                try:
                    process.wait(timeout=max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired as error:
                    if action == "kill":
                        errors.append(error)
                except Exception as error:
                    errors.append(error)
            # Popen must reap its own children to retain their actual exit codes.
            roots = {process.pid for process in self.processes}
            descendants = [process for process in self.identities if process.pid not in roots]
            attempt(lambda: wait_for_processes(descendants, max(0.0, deadline - time.monotonic())))
        attempt(lambda: self.capture(include_orphans=True))
        survivors = []
        for process in self.identities:
            try:
                if process_running(process):
                    survivors.append(process.pid)
            except Exception as error:
                survivors.append(process.pid)
                errors.append(error)
        deadline = time.monotonic() + self.grace
        for process in self.processes:
            attempt(lambda process=process: process.wait(timeout=max(0.0, deadline - time.monotonic())))
        if survivors or errors:
            raise RuntimeError(f"Process cleanup failed; survivors={survivors}; errors={[str(e) for e in errors]}")


def run_owned_process(args, *, cwd=None, env=None, timeout=300.0, cleanup_grace=5.0):
    """Capture output without inherited pipe hangs and clean every owned child on every exit."""
    with (
        tempfile.TemporaryFile(mode="w+t", encoding="utf-8", errors="replace") as stdout,
        tempfile.TemporaryFile(mode="w+t", encoding="utf-8", errors="replace") as stderr,
    ):
        process = None
        try:
            with OwnedProcesses(grace=cleanup_grace) as owner:
                process = owner.start(args, cwd=cwd, env=env, stdout=stdout, stderr=stderr)
                deadline = time.monotonic() + timeout
                while process.poll() is None:
                    owner.capture()
                    remaining = max(0.0, deadline - time.monotonic())
                    if not remaining:
                        raise subprocess.TimeoutExpired(args, timeout)
                    try:
                        process.wait(timeout=min(0.1, remaining))
                    except subprocess.TimeoutExpired:
                        pass
        except BaseException as error:
            stdout.seek(0)
            stderr.seek(0)
            error.output, error.stderr = stdout.read(), stderr.read()
            error.returncode = process.returncode if process is not None else None
            if isinstance(error, RuntimeError):
                error.args = (
                    f"{error}\ncommand={args!r}; returncode={error.returncode}"
                    f"\nstdout (tail):\n{error.output[-20000:]}\nstderr (tail):\n{error.stderr[-20000:]}",
                )
            raise
        stdout.seek(0)
        stderr.seek(0)
        return subprocess.CompletedProcess(args, process.returncode, stdout.read(), stderr.read())
