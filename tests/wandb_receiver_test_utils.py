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

"""Isolated real-SDK worker for W&B backend policy transport tests."""

import json
import sys
from pathlib import Path
from queue import SimpleQueue
from types import SimpleNamespace

from nvflare.apis.fl_constant import ProcessType, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.workspace import Workspace
from nvflare.app_opt.tracking.wandb import wandb_receiver as subject


def main(config_file):
    config = json.loads(Path(config_file).read_text())
    root = Path(config["workspace"])
    (root / "startup").mkdir(parents=True)
    (root / "local").mkdir()
    (root / "local" / "resources.json").write_text(json.dumps({"wandb_base_urls": config.get("approved", [])}))
    context = FLContext()
    context.set_prop(ReservedKey.RUN_NUM, "test-job")
    context.set_prop(ReservedKey.IDENTITY_NAME, "site-1")
    context.set_prop(ReservedKey.PROCESS_TYPE, ProcessType.CLIENT_JOB)
    context.set_prop(
        ReservedKey.ENGINE,
        SimpleNamespace(get_workspace=lambda: Workspace(str(root)), fire_event=lambda *_, **__: None),
    )
    started = []

    class DeferredProcess:
        def __init__(self, **kwargs):
            pass

        def start(self):
            started.append(True)

    # Exercise production initialization and queue processing separately in this
    # interpreter. W&B's own SDK runtime and its HTTP requests remain real.
    subject.Process = DeferredProcess
    subject.Queue = SimpleQueue
    args = {
        "name": "inert",
        "project": "test",
        "group": "test",
        "job_type": "train",
        "settings": {"init_timeout": 10, "console": "off", "disable_git": True},
    }
    if config.get("selected"):
        args["settings"]["base_url"] = config["selected"]
    receiver = subject.WandBReceiver(args, mode=config["mode"])
    result = {}
    try:
        receiver.initialize(context)
    except ValueError:
        result["status"] = "blocked"
    else:
        queue = receiver.get_task_queue("site-1")
        queue.put(subject.WandBTask("site-1", "log", {"loss": 0.25}, 0))
        queue.put(subject.WandBTask("site-1", "stop", {}, 0))
        try:
            receiver._process_queue_tasks(queue)
            result["status"] = "finished"
        except Exception as error:
            result["status"] = "sdk_error"
            result["error_type"] = type(error).__name__
    result["workers_started"] = len(started)
    result["offline_files"] = len(list(root.rglob("*.wandb")))
    Path(config["result"]).write_text(json.dumps(result))


if __name__ == "__main__":
    main(sys.argv[1])
