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

"""One Docker PID-1 worker; the CP service owns whole-container settlement."""

import argparse
import logging
import os
import sys
import threading

from nvflare.apis.job_launcher_spec import pop_credential_env
from nvflare.private.fed.app.client.task_worker_process import _flush_logs


def main():
    parser = argparse.ArgumentParser(description="Execute one staged task in an owned container")
    parser.add_argument("--bootstrap", required=True)
    args = parser.parse_args()
    if os.getpid() != 1:
        raise RuntimeError("Docker task worker must be the container's PID 1")
    pop_credential_env()
    sys.argv[:] = ["nvflare-task-worker"]
    exit_code = 1
    try:
        from nvflare.private.fed.client.client_api_task_runner import CLIENT_TASK_CONTEXT_KEYS, bind_client_task_context
        from nvflare.private.fed.task_worker.worker import run_worker

        run_worker(
            args.bootstrap, context_binding=bind_client_task_context, protected_context_keys=CLIENT_TASK_CONTEXT_KEYS
        )
        exit_code = 0
    except BaseException as error:
        from nvflare.security.logging import secure_format_exception

        logging.getLogger(__name__).error("Task worker failed: %s", secure_format_exception(error))
    finally:
        try:
            flush = threading.Thread(target=_flush_logs, daemon=True, name="task-log-flush")
            flush.start()
            flush.join(timeout=2.0)
        finally:
            os._exit(exit_code)


if __name__ == "__main__":
    main()
