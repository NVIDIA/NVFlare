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

"""Process-only entry point; arm parent-loss cleanup before worker imports."""

import argparse
import logging
import os
import signal
import sys
import threading
import time

import psutil


def _is_live(process):
    try:
        return process.is_running() and process.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        # Uncertainty is not proof of owner loss or group quiescence.
        return True


def _dead_descendant_snapshot(worker_pid: int):
    """Return dead member identities, or None for live/uncertain membership."""
    dead_members = set()
    try:
        # Like launcher settlement, use raw PIDs and two stable snapshots so a
        # disappearing process cannot hide a child forked during enumeration.
        for pid in psutil.pids():
            if pid == 0 or pid in (worker_pid, os.getpid()):
                # macOS lists kernel PID 0, but getpgid(0) means this caller's
                # group. Also exclude the known-dead worker and this guardian.
                continue
            try:
                try:
                    if os.getpgid(pid) != worker_pid:
                        continue
                except ProcessLookupError:
                    # Darwin may drop a zombie's PGID before its PID disappears.
                    pass
                process = psutil.Process(pid)
                if process.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                    return None
                dead_members.add((pid, process.create_time()))
            except (ProcessLookupError, psutil.NoSuchProcess, PermissionError, psutil.AccessDenied):
                return None
    except (PermissionError, psutil.AccessDenied):
        return None
    return dead_members


def _watch_parent(parent: psutil.Process, worker: psutil.Process):
    # This tiny guardian stays in the owned group, including after the worker
    # exits. A thread would disappear at os._exit and could leave descendants
    # orphaned if CJ dies before its launcher finishes settlement.
    # The inherited Process objects pin identities before the fork. Worker exit
    # only starts a quiescence check; the live CJ still owns graceful shutdown.
    worker_pid = worker.pid
    group_empty = False
    try:
        while _is_live(parent):
            if not _is_live(worker):
                first = _dead_descendant_snapshot(worker_pid)
                if first is not None and first == _dead_descendant_snapshot(worker_pid):
                    group_empty = True
                    return
            time.sleep(0.05)
    finally:
        if group_empty:
            # Do not keep an otherwise quiescent group alive until the launcher
            # escalates. Only the launcher may reap the worker and settle it.
            os._exit(0)
        try:
            # Stop the identity-checked leader first. On some POSIX systems a
            # killpg from within the group can kill this guardian before the
            # signal reaches the leader. Group cleanup still covers descendants.
            try:
                worker.kill()
            except psutil.NoSuchProcess:
                pass
        finally:
            try:
                # Membership reserves this PGID even if the leader has exited;
                # never signal a different group or reap the launcher-owned leader.
                if os.getpgrp() == worker_pid:
                    os.killpg(worker_pid, signal.SIGKILL)
            finally:
                os._exit(1)


def _start_parent_guard(parent_pid: int):
    parent = psutil.Process(parent_pid)
    worker = psutil.Process(os.getpid())
    if worker.ppid() != parent_pid:
        raise RuntimeError("task worker lost its owning Client Job before starting the guardian")
    # Fork before framework/application imports or threads. The guardian ignores
    # SIGTERM so it covers the launcher's graceful-cancellation window too.
    old_handler = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        guardian_pid = os.fork()
        if guardian_pid == 0:
            _watch_parent(parent, worker)
    finally:
        signal.signal(signal.SIGTERM, old_handler)


def _flush_logs():
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()


def main():
    parser = argparse.ArgumentParser(description="Execute one staged NVFlare task assignment")
    parser.add_argument("--bootstrap", required=True, help="absolute path to the worker bootstrap JSON")
    parser.add_argument("--parent_pid", type=int, default=None, help="owning Client Job PID (Process launcher)")
    args = parser.parse_args()
    sys.argv[:] = ["nvflare-task-worker"]
    parent_pid = args.parent_pid if args.parent_pid is not None else os.getppid()
    if parent_pid <= 1 or os.getpgrp() != os.getpid():
        raise RuntimeError("a Process task worker requires a live parent and its own process group")
    if os.getppid() != parent_pid:
        os.killpg(os.getpid(), signal.SIGKILL)
    from nvflare.apis.job_launcher_spec import pop_credential_env

    # Also protect the guardian if a custom launcher forwarded a broader env.
    pop_credential_env()
    _start_parent_guard(parent_pid)
    exit_code = 1
    try:
        from nvflare.private.fed.task_worker.worker import run_worker

        run_worker(args.bootstrap)
        exit_code = 0
    except BaseException as e:
        from nvflare.security.logging import secure_format_exception

        logging.getLogger(__name__).error("Task worker failed: %s", secure_format_exception(e))
    finally:
        # run_worker has already finalized components and committed completion.
        # Do not enter Python's unbounded threading/multiprocessing atexit joins.
        # The launcher settles any remaining members of the owned process group.
        try:
            flush = threading.Thread(target=_flush_logs, daemon=True, name="task-log-flush")
            flush.start()
            flush.join(timeout=2.0)
        finally:
            os._exit(exit_code)


if __name__ == "__main__":
    main()
