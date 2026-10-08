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


def _watch_parent(parent_pid: int, worker_pid: int):
    # This tiny guardian stays in the owned group, including after the worker
    # exits. A thread would disappear at os._exit and could leave descendants
    # orphaned if CJ dies before its launcher finishes settlement.
    # Relationship and identity checks observe loss without trusting reusable
    # PIDs. An exited, unreaped parent can retain its relationship, so zombies
    # are not live owners.
    worker = None
    try:
        worker = psutil.Process(worker_pid)
        parent = psutil.Process(parent_pid)
        while (
            os.getppid() == worker_pid
            and worker.is_running()
            and worker.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
            and worker.ppid() == parent_pid
            and parent.is_running()
            and parent.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
        ):
            time.sleep(0.05)
    finally:
        try:
            # Stop the identity-checked leader first. On some POSIX systems a
            # killpg from within the group can kill this guardian before the
            # signal reaches the leader. Group cleanup still covers descendants.
            if worker is not None:
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
    worker_pid = os.getpid()
    # Fork before framework/application imports or threads. The guardian ignores
    # SIGTERM so it covers the launcher's graceful-cancellation window too.
    old_handler = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        guardian_pid = os.fork()
        if guardian_pid == 0:
            _watch_parent(parent_pid, worker_pid)
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
