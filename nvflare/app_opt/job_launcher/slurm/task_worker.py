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
"""One Slurm allocation's authorized worker, without a local CJ PID contract."""

import argparse
import os
import sys

from nvflare.private.fed.app.client.task_worker_process import main as worker_main


def main():
    parser = argparse.ArgumentParser(description="Execute one Slurm task assignment")
    parser.add_argument("--bootstrap", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise RuntimeError("Slurm task worker requires an allocation identity")
    # The common entry point binds Client API context, removes inherited job
    # credentials and bounds final log flushing/exit. Slurm owns settlement of
    # the allocation; its worker does not share a host-local CJ parent PID.
    sys.argv[:] = ["nvflare-slurm-task-worker", "--bootstrap", args.bootstrap, "--execution_scope", "native"]
    worker_main()


if __name__ == "__main__":
    main()
