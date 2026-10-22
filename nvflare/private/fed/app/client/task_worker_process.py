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

"""Shared task-worker entry point; the selected launcher owns settlement."""

import argparse
import logging
import os
import signal
import sys
import threading


def _flush_logs():
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()


def main():
    parser = argparse.ArgumentParser(description="Execute one staged NVFlare task assignment")
    parser.add_argument("--bootstrap", required=True, help="absolute path to the worker bootstrap JSON")
    parser.add_argument(
        "--execution_scope",
        choices=("process", "native"),
        default="process",
        help="launcher-owned execution scope; native workers have no local CJ parent",
    )
    parser.add_argument("--parent_pid", type=int, default=None, help="owning Client Job PID (Process launcher)")
    args = parser.parse_args()
    if args.execution_scope == "native" and args.parent_pid is not None:
        parser.error("native execution cannot use a local parent PID")
    if args.execution_scope == "process" and args.parent_pid is None:
        parser.error("process execution requires --parent_pid")
    sys.argv[:] = ["nvflare-task-worker"]
    if args.execution_scope == "process":
        parent_pid = args.parent_pid
        # Process mode requires a Process CJ. Native workers do not rely on
        # local PID or process-group ownership.
        if parent_pid <= 1 or os.getpgrp() != os.getpid():
            raise RuntimeError("a Process task worker requires a live parent and its own process group")
        if os.getppid() != parent_pid:
            os.killpg(os.getpid(), signal.SIGKILL)
    from nvflare.apis.job_launcher_spec import pop_credential_env

    pop_credential_env()
    exit_code = 1
    try:
        from nvflare.private.fed.client.client_api_task_runner import CLIENT_TASK_CONTEXT_KEYS, bind_client_task_context
        from nvflare.private.fed.task_worker.worker import run_worker

        run_worker(
            args.bootstrap, context_binding=bind_client_task_context, protected_context_keys=CLIENT_TASK_CONTEXT_KEYS
        )
        exit_code = 0
    except BaseException as e:
        from nvflare.security.logging import secure_format_exception

        logging.getLogger(__name__).error("Task worker failed: %s", secure_format_exception(e))
    finally:
        # run_worker has already finalized components and committed completion.
        # Do not enter Python's unbounded threading/multiprocessing atexit joins.
        # The launcher settles the complete owned process group or native scope.
        try:
            flush = threading.Thread(target=_flush_logs, daemon=True, name="task-log-flush")
            flush.start()
            flush.join(timeout=2.0)
        finally:
            os._exit(exit_code)


if __name__ == "__main__":
    main()
