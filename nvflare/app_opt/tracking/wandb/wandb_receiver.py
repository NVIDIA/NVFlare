# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import os
import time
from multiprocessing import Process, Queue
from typing import NamedTuple, Optional

import wandb

from nvflare.apis.analytix import AnalyticsData, AnalyticsDataType
from nvflare.apis.dxo import from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.tracking.tracker_types import LogWriterName
from nvflare.app_common.widgets.streaming import AnalyticsReceiver


class WandBTask(NamedTuple):
    task_owner: str
    task_type: str
    task_data: dict
    step: int


class WandBReceiver(AnalyticsReceiver):
    def __init__(self, kwargs: dict, mode: str = "offline", events=None, process_timeout=10):
        if events is None:
            events = ["fed.analytix_log_stats"]
        super().__init__(events=events)
        self.fl_ctx = None
        self.mode = mode
        self.kwargs = kwargs
        self.queues = {}
        self.processes = {}
        self.process_timeout = process_timeout

        # os.environ["WANDB_API_KEY"] = YOUR_KEY_HERE
        os.environ["WANDB_MODE"] = self.mode

    def job(self, queue):
        cnt = 0
        run = None
        try:
            while True:
                wandb_task: WandBTask = queue.get()
                cnt += 1
                if wandb_task.task_type == "stop":
                    self.log_info(self.fl_ctx, f"received request to stop at {wandb_task.task_owner} for run {run}")
                    break
                elif wandb_task.task_type == "init":
                    self.log_info(self.fl_ctx, f"received request to init at {wandb_task.task_owner}")
                    run = wandb.init(**wandb_task.task_data)
                elif wandb_task.task_type == "log":
                    if cnt % 500 == 0:
                        self.log_info(self.fl_ctx, f"process task : {wandb_task}, cnt = {cnt}")

                    if wandb_task.step:
                        wandb.log(wandb_task.task_data, wandb_task.step)
                    else:
                        wandb.log(wandb_task.task_data)
        finally:
            if run:
                run.finish()

    def initialize(self, fl_ctx: FLContext):
        self.fl_ctx = fl_ctx
        sites = fl_ctx.get_engine().get_clients()
        run_group_id = str(int(time.time()))

        run_name = self.kwargs["name"]
        job_id_tag = self.get_job_id_tag(run_group_id)
        wand_config = self.kwargs.get("config", {})

        if self.mode == "online":
            try:
                wandb.login(timeout=1, verify=True)
            except Exception as e:
                self.log_error(self.fl_ctx, f"Unsuccessful login: {e}. Using wandb offline mode.")
                self.mode = "offline"

        for site in sites:
            self.log_info(self.fl_ctx, f"initialize WandB run for site {site.name}")
            self.kwargs["name"] = f"{site.name}-{job_id_tag[:6]}-{run_name}"
            self.kwargs["group"] = f"{run_name}-{job_id_tag}"
            self.kwargs["mode"] = self.mode
            wand_config["job_id"] = job_id_tag
            wand_config["client"] = site.name
            wand_config["run_name"] = run_name

            self.check_kwargs(self.kwargs)

            q = Queue()
            wandb_task = WandBTask(task_owner=site.name, task_type="init", task_data=self.kwargs, step=0)
            # q.put_nowait(wandb_task)
            q.put(wandb_task)

            self.queues[site.name] = q
            p = Process(target=self.job, args=(q,))
            self.processes[site.name] = p
            p.start()
            time.sleep(0.2)

    def get_job_id_tag(self, group_id: str) -> str:
        job_id = self.fl_ctx.get_job_id()
        if job_id == "simulate_job":
            # For simulator, the job ID is the same so we use a string of the time for the job_id_tag
            job_id = group_id
        return job_id

    def save(self, fl_ctx: FLContext, shareable: Shareable, record_origin: str):
        dxo = from_shareable(shareable)
        data = AnalyticsData.from_dxo(dxo, receiver=LogWriterName.WANDB)
        if not data:
            return

        q: Optional[Queue] = self.get_job_queue(record_origin)
        if q:
            if data.data_type == AnalyticsDataType.PARAMETER or data.data_type == AnalyticsDataType.METRIC:
                log_data = {data.tag: data.value}
                q.put(WandBTask(task_owner=record_origin, task_type="log", task_data=log_data, step=data.step))
            elif data.data_type == AnalyticsDataType.PARAMETERS or data.data_type == AnalyticsDataType.METRICS:
                q.put(WandBTask(task_owner=record_origin, task_type="log", task_data=data.value, step=data.step))

    def finalize(self, fl_ctx: FLContext):
        """Called at EventType.END_RUN.

        Args:
            fl_ctx (FLContext): the FLContext
        """
        for site in self.processes:
            self.log_info(self.fl_ctx, f"inform {site} to stop")
            q: Optional[Queue] = self.get_job_queue(site)
            q.put(WandBTask(task_owner=site, task_type="stop", task_data={}, step=0))

        for site in self.processes:
            p = self.processes[site]
            p.join(self.process_timeout)
            p.terminate()

    def get_job_queue(self, record_origin):
        return self.queues.get(record_origin, None)

    def check_kwargs(self, kwargs):
        if "project" not in kwargs:
            raise ValueError("must provide `project' value")

        if "group" not in kwargs:
            raise ValueError("must provide `group' value")

        if "job_type" not in kwargs:
            raise ValueError("must provide `job_type' value")
