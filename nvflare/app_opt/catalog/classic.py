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

"""The catalog read job on the Controller/Executor API."""

from nvflare.apis.controller_spec import ClientTask, Task
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_opt.catalog.results import download_catalog, write_results
from nvflare.app_opt.catalog.site import read_site_catalog
from nvflare.job_config.api import FedJob
from nvflare.recipe.spec import ExecEnv, Recipe

TASK_NAME = "catalog_read"
TASK_TIMEOUT_S = 300
REPLY_KEY = "catalog"


class CatalogExecutor(Executor):
    def __init__(self, config_root: str | None = None):
        super().__init__()
        self.config_root = config_root

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        reply = make_reply(ReturnCode.OK)
        reply[REPLY_KEY] = read_site_catalog(fl_ctx, self.config_root)
        return reply


class CatalogController(Controller):
    def start_controller(self, fl_ctx: FLContext):
        pass

    def stop_controller(self, fl_ctx: FLContext):
        pass

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        replies = {client.name: None for client in self._engine.get_clients()}

        def record(client_task: ClientTask, fl_ctx: FLContext):
            if client_task.result.get_return_code() == ReturnCode.OK:
                replies[client_task.client.name] = client_task.result.get(REPLY_KEY)

        task = Task(name=TASK_NAME, data=Shareable(), timeout=TASK_TIMEOUT_S, result_received_cb=record)
        self.broadcast_and_wait(task, fl_ctx, min_responses=len(replies), abort_signal=abort_signal)
        write_results(fl_ctx, replies)


class CatalogRecipe(Recipe):
    def __init__(self, name: str = "catalog", config_root: str | None = None):
        job = FedJob(name=name, min_clients=1)
        job.to_server(CatalogController())
        job.to_clients(CatalogExecutor(config_root=config_root), tasks=[TASK_NAME])
        super().__init__(job)


def get_catalog(env: ExecEnv, output_dir: str | None = None, **recipe_args) -> str:
    """Read the catalog of the Study of ``env`` and return the folder holding its contracts and sites.json."""
    return download_catalog(CatalogRecipe(**recipe_args).execute(env), output_dir)
