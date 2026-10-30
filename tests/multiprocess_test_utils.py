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

"""Small real executors and a CPU-only rank launcher for multiprocess contracts."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.shareable import Shareable
from nvflare.app_common.executors.multi_process_executor import MultiProcessExecutor


class RankExecutor(Executor):
    def __init__(self, root, mode="success", distributed=False):
        super().__init__()
        self.root = Path(root)
        self.mode = mode
        self.distributed = distributed

    def handle_event(self, event_type, fl_ctx):
        rank = fl_ctx.get_prop(FLContextKey.RANK_NUMBER)
        if event_type == EventType.START_RUN:
            (self.root / f"rank-pid-{rank}").write_text(
                json.dumps({"pid": os.getpid(), "create_time": psutil.Process().create_time()})
            )
            if self.mode == "startup_failure":
                raise RuntimeError("injected startup failure")
            engine = fl_ctx.get_engine()
            assert engine.get_cell() is not None
            engine.build_component(
                {"path": "nvflare.app_common.widgets.component_path_authorizer.ComponentPathAuthorizer"}
            )
            if self.distributed:
                import datetime

                import torch.distributed as dist

                dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=15))
            (self.root / f"started-{rank}").touch()
        elif event_type == EventType.END_RUN:
            if self.distributed:
                import torch.distributed as dist

                dist.destroy_process_group()
            (self.root / f"ended-{rank}").touch()

    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        rank = fl_ctx.get_prop(FLContextKey.RANK_NUMBER)
        (self.root / f"executing-{rank}").touch()
        if self.mode == "failure" and rank == 1:
            raise RuntimeError("injected rank-one failure")
        if self.mode == "exit" and rank == 1:
            os._exit(7)
        if self.mode == "block":
            while True:
                time.sleep(1)
        total = None
        if self.distributed:
            import torch
            import torch.distributed as dist

            value = torch.tensor(float(rank + 1))
            dist.all_reduce(value)
            total = value.item()
        return Shareable({"rank": rank, "round": shareable["round"], "total": total})


class LocalLauncherExecutor(MultiProcessExecutor):
    def get_multi_process_command(self):
        return f"{sys.executable} -m tests.multiprocess_test_utils {self.num_of_processes}"


def launch_ranks():
    count = int(sys.argv[1])
    children = []
    try:
        for rank in range(count):
            env = os.environ.copy()
            env["LOCAL_RANK"] = str(rank)
            children.append(subprocess.Popen([sys.executable, *sys.argv[2:]], env=env))
        while any(child.poll() is None for child in children):
            if any(child.poll() not in (None, 0) for child in children):
                return 1
            time.sleep(0.05)
        return max(child.returncode for child in children)
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
        for child in children:
            child.wait(timeout=5)


def run_contract(root, mode, simulator, backend="local"):
    import argparse
    import json
    from types import SimpleNamespace

    from nvflare.apis.fl_context import FLContextManager
    from nvflare.apis.signal import Signal
    from nvflare.apis.workspace import Workspace
    from nvflare.fuel.f3.cellnet.cell import Cell
    from nvflare.fuel.utils.network_utils import get_open_ports
    from nvflare.private.fed.app.fl_conf import create_privacy_manager
    from nvflare.private.fed.utils.fed_utils import fobs_initialize
    from nvflare.private.privacy_manager import PrivacyService

    root = Path(root)
    for folder in ["startup", "local", "job"]:
        (root / folder).mkdir(exist_ok=True)
    (root / "job" / "meta.json").write_text("{}")
    (root / "local" / "resources.json").write_text(json.dumps({"class_allow_list": ["tests.", "nvflare."]}))
    (root / "local" / "log_config.json").write_text(json.dumps({"version": 1, "disable_existing_loggers": False}))
    os.environ["NVFLARE_MPE_TEST_PID_FILE"] = str(root / "launcher.pid")
    workspace = Workspace(str(root), "site-1")
    fobs_initialize()
    PrivacyService.initialize(create_privacy_manager(workspace, names_only=True))
    url = f"tcp://127.0.0.1:{get_open_ports(1)[0]}"
    cell = Cell(fqcn="site-1.job", root_url=url, secure=False, credentials={}, create_internal_listener=True)
    engine = SimpleNamespace(client=SimpleNamespace(cell=cell))
    context_manager = FLContextManager(engine=engine, identity_name="site-1", job_id="job")
    engine.new_context = context_manager.new_context
    engine.fire_event = lambda *args, **kwargs: None
    ctx = engine.new_context()
    ctx.set_prop(FLContextKey.ARGS, argparse.Namespace(workspace=str(root)))
    ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace)
    ctx.set_prop(FLContextKey.SIMULATE_MODE, simulator == "true")

    class FileAbort(Signal):
        @property
        def triggered(self):
            return (root / "abort").exists()

    abort = FileAbort()
    launcher_class = LocalLauncherExecutor
    if backend == "torchrun":
        from nvflare.app_opt.pt.multi_process_executor import PTMultiProcessExecutor

        launcher_class = PTMultiProcessExecutor
        os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
    executor = launcher_class(
        executor_id="trainer",
        num_of_processes=2,
        task_timeout=15,
        components=[
            {
                "id": "trainer",
                "path": "tests.multiprocess_test_utils.RankExecutor",
                "args": {"root": str(root), "mode": mode, "distributed": backend == "torchrun"},
            }
        ],
    )
    cell.start()
    try:
        if mode == "startup_failure":
            try:
                executor.initialize(ctx)
            except RuntimeError:
                (root / "startup-rejected").touch()
            else:
                raise AssertionError("startup failure was swallowed")
        else:
            executor.initialize(ctx)
            (root / "launcher.pid").write_text(
                json.dumps({"pid": executor.exe_process.pid, "create_time": executor._launcher_process.create_time()})
            )
            for round_id in range(2 if mode == "success" else 1):
                result = executor.execute("train", Shareable({"round": round_id}), ctx, abort)
                (root / "result.json").write_text(json.dumps(dict(result)))
                if mode == "success":
                    assert result["rank"] == 0 and result["round"] == round_id
                    if backend == "torchrun":
                        assert result["total"] == 3.0
            executor.handle_event(EventType.END_RUN, ctx)
        assert executor.exe_process.poll() is not None
    finally:
        executor.finalize(ctx)
        cell.stop()


if __name__ == "__main__":
    if sys.argv[1] == "contract":
        run_contract(*sys.argv[2:])
    else:
        pid_file = os.environ.get("NVFLARE_MPE_TEST_PID_FILE")
        if pid_file:
            Path(pid_file).write_text(json.dumps({"pid": os.getpid(), "create_time": psutil.Process().create_time()}))
        raise SystemExit(launch_ranks())
