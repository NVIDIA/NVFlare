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

"""Targets for independently owned Flower process lifecycle tests."""

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.tie.process_mgr import CommandDescriptor
from nvflare.app_opt.flower.process import FlowerProcessManager
from tests.timing_utils import wait_for


def guarded_job(run_dir):
    root = Path(run_dir)
    worker_ready = root / "worker.pid"
    worker_code = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"Path({str(worker_ready)!r}).write_text(str(os.getpid())); time.sleep(300)"
    )
    code = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',{worker_code!r}],start_new_session=True); time.sleep(300)"
    )
    context = MagicMock(spec=FLContext)
    context.get_engine.return_value.get_workspace.return_value.get_run_dir.return_value = str(root)
    manager = FlowerProcessManager(CommandDescriptor(shlex.join([sys.executable, "-c", code]), log_stdout=False))
    manager.start(context)
    wait_for(lambda: worker_ready.exists() and worker_ready.read_text().isdigit(), timeout=30)
    (root / "ready.json").write_text(
        json.dumps(
            {"flower": manager.process.pid, "guard": manager._guard.pid, "worker": int(worker_ready.read_text())}
        )
    )
    while not (root / "exit").exists():
        time.sleep(0.05)
    # NVFlare's mpm wrapper also exits without Python thread teardown.
    os._exit(0)


def guarded_job_during_launch(run_dir):
    root = Path(run_dir)
    context = MagicMock(spec=FLContext)
    context.get_engine.return_value.get_workspace.return_value.get_run_dir.return_value = str(root)
    manager = FlowerProcessManager(
        CommandDescriptor(shlex.join([sys.executable, "-c", "import time; time.sleep(300)"]), log_stdout=False)
    )
    popen = subprocess.Popen

    def pause_before_exec():
        # Model the real close_fds -> execve gap, before the owner environment
        # is installed and while the job's Popen still waits for exec completion.
        os.close(manager._guard.stdin.fileno())
        (root / "launch.json").write_text(json.dumps({"target": os.getpid(), "guard": manager._guard.pid}))
        os.kill(os.getpid(), signal.SIGSTOP)

    def launch(args, **kwargs):
        if manager._guard is not None:
            kwargs["preexec_fn"] = pause_before_exec
        return popen(args, **kwargs)

    # This single-threaded test interpreter alone uses preexec_fn. The product
    # uses a fresh interpreter and never runs Python callbacks after fork.
    subprocess.Popen = launch
    manager.start(context)
