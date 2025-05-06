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
from typing import List, NamedTuple, Optional

import wandb

from nvflare.apis.analytix import AnalyticsData, AnalyticsDataType, LogWriterName
from nvflare.apis.dxo import from_shareable
from nvflare.apis.fl_constant import ProcessType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.widgets.streaming import AnalyticsReceiver


class WandBTask(NamedTuple):
    task_owner: str
    task_type: str
    task_data: dict
    step: int


def _check_wandb_args(wandb_args):
    if "project" not in wandb_args:
        raise ValueError("must provide 'project' value")

    if "group" not in wandb_args:
        raise ValueError("must provide 'group' value")

    if "job_type" not in wandb_args:
        raise ValueError("must provide 'job_type' value")


def _get_job_id_tag(fl_ctx: FLContext) -> str:
    """Gets a unique job id tag."""
    job_id = fl_ctx.get_job_id()
    if job_id == "simulate_job":
        # Since all jobs run in the simulator have the same job_id of "simulate_job"
        # Use timestamp as unique identifier for simulation runs
        job_id = str(int(time.time()))
    return job_id


class WandBReceiver(AnalyticsReceiver):
    def __init__(
        self, wandb_args: dict, mode: str = "offline", events: Optional[List[str]] = None, process_timeout: float = 10.0
    ):
        super().__init__(events=events)
        self.fl_ctx = None
        self.mode = mode
        self.wandb_args = wandb_args
        self.queues = {}
        self.processes = {}
        self.metrics_buffer = {}
        self.process_timeout = process_timeout

        # os.environ["WANDB_API_KEY"] = YOUR_KEY_HERE
        os.environ["WANDB_MODE"] = self.mode

    def _process_queue_tasks(self, queue):
        cnt = 0
        run = None
        current_step = 0
        try:
            while True:
                wandb_task: WandBTask = queue.get()
                cnt += 1
                if wandb_task.task_type == "stop":
                    self.log_info(self.fl_ctx, f"received request to stop at {wandb_task.task_owner} for run {run}")
                    # Log the last step's metrics before stopping
                    if current_step in self.metrics_buffer:
                        wandb.log(self.metrics_buffer[current_step], current_step)
                    break
                elif wandb_task.task_type == "init":
                    self.log_info(self.fl_ctx, f"received request to init at {wandb_task.task_owner}")
                    run = wandb.init(**wandb_task.task_data)
                elif wandb_task.task_type == "log":
                    if cnt % 500 == 0:
                        self.log_info(self.fl_ctx, f"process task : {wandb_task}, cnt = {cnt}")

                    if wandb_task.step is not None:
                        if wandb_task.step < current_step:
                            self.log_warning(
                                self.fl_ctx, f"Received out-of-order step: {wandb_task.step} (current: {current_step})"
                            )
                            continue

                        # If we see a new step, log the previous step's metrics
                        if wandb_task.step > current_step and current_step in self.metrics_buffer:
                            wandb.log(self.metrics_buffer[current_step], current_step)
                            del self.metrics_buffer[current_step]

                        # Store metrics in buffer for current step
                        if wandb_task.step not in self.metrics_buffer:
                            self.metrics_buffer[wandb_task.step] = {}
                        self.metrics_buffer[wandb_task.step].update(wandb_task.task_data)
                        current_step = wandb_task.step
                    else:
                        # Use current step for metrics without a step
                        if current_step not in self.metrics_buffer:
                            self.metrics_buffer[current_step] = {}
                        self.metrics_buffer[current_step].update(wandb_task.task_data)
        finally:
            if run:
                run.finish()

    def initialize(self, fl_ctx: FLContext):
        # Determine participating sites
        if fl_ctx.get_process_type() == ProcessType.SERVER_JOB:
            clients = fl_ctx.get_engine().get_clients()
            if not clients:
                raise RuntimeError("No clients found in server context")
            site_names = [c.name for c in clients]
        else:
            # Client context - track only this client
            site_name = fl_ctx.get_identity_name()
            if not site_name:
                raise RuntimeError("Unable to determine client identity")
            site_names = [site_name]

        self.log_info(fl_ctx, f"Initializing WandB tracking for sites: {site_names}")

        self.fl_ctx = fl_ctx

        run_name = self.wandb_args["name"]
        job_id_tag = _get_job_id_tag(fl_ctx)
        wand_config = self.wandb_args.get("config", {})

        if self.mode == "online":
            try:
                wandb.login(timeout=1, verify=True)
            except Exception as e:
                self.log_warning(fl_ctx, f"Unsuccessful login: {e}. Using wandb offline mode.")
                self.mode = "offline"

        for site_name in site_names:
            self.log_info(fl_ctx, f"initialize WandB run for site {site_name}")
            self.wandb_args["name"] = f"{site_name}-{job_id_tag}-{run_name}"
            self.wandb_args["group"] = f"{run_name}-{job_id_tag}"
            self.wandb_args["mode"] = self.mode
            wand_config["job_id"] = job_id_tag
            wand_config["client"] = site_name
            wand_config["run_name"] = run_name

            _check_wandb_args(self.wandb_args)

            q = Queue()
            wandb_task = WandBTask(task_owner=site_name, task_type="init", task_data=self.wandb_args, step=0)
            q.put(wandb_task)

            self.queues[site_name] = q
            p = Process(target=self._process_queue_tasks, args=(q,))
            self.processes[site_name] = p
            p.start()
            time.sleep(0.2)

    def save(self, fl_ctx: FLContext, shareable: Shareable, record_origin: str):
        dxo = from_shareable(shareable)
        data = AnalyticsData.from_dxo(dxo, receiver=LogWriterName.WANDB)
        if not data:
            return

        q: Optional[Queue] = self.get_task_queue(record_origin)
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
            self.log_info(fl_ctx, f"inform {site} to stop")
            q: Optional[Queue] = self.get_task_queue(site)
            q.put(WandBTask(task_owner=site, task_type="stop", task_data={}, step=0))

        for site in self.processes:
            p = self.processes[site]
            p.join(self.process_timeout)
            p.terminate()

    def get_task_queue(self, record_origin):
        return self.queues.get(record_origin, None)
