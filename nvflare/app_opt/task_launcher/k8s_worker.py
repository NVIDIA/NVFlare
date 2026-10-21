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

"""Container worker entry point; pod ownership replaces Process parent-PID guards."""

import argparse
import logging
import os
import sys
import threading


def main():
    parser = argparse.ArgumentParser(description="Execute one staged task in an owned Kubernetes pod")
    parser.add_argument("--bootstrap", required=True)
    args = parser.parse_args()
    sys.argv[:] = ["nvflare-task-worker"]
    from nvflare.apis.job_launcher_spec import pop_credential_env

    pop_credential_env()
    from nvflare.private.fed.client.client_api_task_runner import CLIENT_TASK_CONTEXT_KEYS, bind_client_task_context
    from nvflare.private.fed.task_worker.worker import run_worker

    try:
        run_worker(
            args.bootstrap, context_binding=bind_client_task_context, protected_context_keys=CLIENT_TASK_CONTEXT_KEYS
        )
        return 0
    except BaseException as error:
        from nvflare.security.logging import secure_format_exception

        logging.getLogger(__name__).error("Task worker failed: %s", secure_format_exception(error))
        return 1


def _flush_logs():
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    exit_code = main()
    try:
        flush = threading.Thread(target=_flush_logs, daemon=True, name="task-log-flush")
        flush.start()
        flush.join(timeout=2)
    finally:
        # The pod runtime terminates every remaining process when its main
        # container exits. Do not enter unbounded Python atexit/thread joins.
        os._exit(exit_code)
