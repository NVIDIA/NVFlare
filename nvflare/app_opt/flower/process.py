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
import shlex
import signal
import subprocess
import threading
import time

import psutil

from nvflare.app_common.tie.process_mgr import ProcessManager


class FlowerProcessManager(ProcessManager):
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
        try:
            self.process = subprocess.Popen(
                shlex.split(self.cmd_desc.cmd),
                cwd=self.cmd_desc.cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception:
            if self.log_file:
                self.log_file.close()
                self.log_file = None
            raise
        self._log_writer = threading.Thread(target=self._write_log)
        self._log_writer.start()

    def _signal_group(self, sig):
        try:
            os.killpg(self.process.pid, sig)
        except ProcessLookupError:
            pass

    def stop(self, timeout=5.0):
        if not self.process:
            return 0
        # SuperLink detaches SuperExec into another session. Snapshot those
        # descendants before terminating the parent, retaining psutil's PID reuse
        # protection. Normal Flower parent monitors also stop orphaned workers.
        try:
            descendants = psutil.Process(self.process.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            descendants = []
        # Flower handles SIGTERM and reaps its own workers on graceful shutdown.
        self._signal_group(signal.SIGTERM)
        try:
            self.process.wait(timeout=max(timeout, 0.1))
        except subprocess.TimeoutExpired:
            self._signal_group(signal.SIGKILL)
            self.process.wait()
        finally:
            # A parent may have crashed before cleaning up its remaining workers.
            self._signal_group(signal.SIGKILL)
            deadline = time.monotonic() + max(timeout, 0.1)
            while any(_is_alive(p) for p in descendants) and time.monotonic() < deadline:
                time.sleep(0.05)
            for child in reversed(descendants):
                if _is_alive(child):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
            self._log_writer.join(timeout=5)
            if self._log_writer.is_alive():
                raise RuntimeError("Flower log pipe remained open after process-tree shutdown")
            self.process.stdout.close()
            with self.file_lock:
                if self.log_file:
                    self.log_file.close()
                    self.log_file = None
        return self.process.returncode


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
