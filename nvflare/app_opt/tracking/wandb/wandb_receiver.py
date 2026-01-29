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

import copy
import os
import queue
import threading
import time
from typing import List, NamedTuple, Optional

import wandb

from nvflare.apis.analytix import AnalyticsData, AnalyticsDataType, LogWriterName
from nvflare.apis.dxo import from_shareable
from nvflare.apis.fl_constant import ProcessType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.widgets.streaming import AnalyticsReceiver

# Module-level storage for Process and Queue objects to avoid pickling issues in SimEnv
# Keyed by (run_id, site_name) tuple
_WANDB_QUEUES = {}
_WANDB_PROCESSES = {}


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
        self.process_timeout = process_timeout
        self.site_names = []  # Will be populated in initialize()
        self.run_id = None  # Will be set in initialize() for keying module-level dicts

        # os.environ["WANDB_API_KEY"] = YOUR_KEY_HERE
        os.environ["WANDB_MODE"] = self.mode

    def _process_queue_tasks(self, queue):
        cnt = 0
        run = None
        current_step = 0
        metrics_buffer = {}  # Thread-local buffer, not shared across threads
        try:
            while True:
                wandb_task: WandBTask = queue.get()
                cnt += 1
                if wandb_task.task_type == "stop":
                    self.log_info(self.fl_ctx, f"received request to stop at {wandb_task.task_owner} for run {run}")
                    # Log the last step's metrics before stopping
                    if current_step in metrics_buffer:
                        wandb.log(metrics_buffer[current_step], current_step)
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
                        if wandb_task.step > current_step and current_step in metrics_buffer:
                            wandb.log(metrics_buffer[current_step], current_step)
                            del metrics_buffer[current_step]

                        # Store metrics in buffer for current step
                        if wandb_task.step not in metrics_buffer:
                            metrics_buffer[wandb_task.step] = {}
                        metrics_buffer[wandb_task.step].update(wandb_task.task_data)
                        current_step = wandb_task.step
                    else:
                        # Use current step for metrics without a step
                        if current_step not in metrics_buffer:
                            metrics_buffer[current_step] = {}
                        metrics_buffer[current_step].update(wandb_task.task_data)
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
        self.site_names = site_names
        # Use job_id as unique run_id for keying module-level process/queue storage
        self.run_id = _get_job_id_tag(fl_ctx)

        # Check login for online mode (but don't create processes yet)
        if self.mode == "online":
            try:
                wandb.login(timeout=1, verify=True)
            except Exception as e:
                self.log_warning(fl_ctx, f"Unsuccessful login: {e}. Using wandb offline mode.")
                self.mode = "offline"

        # Note: wandb_args validation happens in _init_site_process() after site-specific config is set

    def _init_site_process(self, site_name: str):
        """Lazily initialize process and queue for a specific site.

        This is called on first use (in save()) to avoid pickling issues in SimEnv.
        Process and Queue objects are stored in module-level dicts to keep them
        outside the receiver instance, making the receiver picklable.
        """
        site_key = (self.run_id, site_name)
        if site_key in _WANDB_PROCESSES:
            return  # Already initialized

        if not self.fl_ctx:
            raise RuntimeError("WandBReceiver not initialized - call initialize() first")

        self.log_info(self.fl_ctx, f"Creating WandB process for site {site_name}")

        # Create site-specific config with deep copy to avoid shared mutable state
        # (wandb_args may contain nested dicts like "config" and lists like "tags")
        site_wandb_args = copy.deepcopy(self.wandb_args)

        run_name = self.wandb_args["name"]  # Get from original (immutable string)
        job_id_tag = self.run_id

        # Modify the deep-copied config (safe - no shared state)
        site_wandb_args["name"] = f"{site_name}-{job_id_tag}-{run_name}"
        site_wandb_args["group"] = f"{run_name}-{job_id_tag}"
        site_wandb_args["mode"] = self.mode

        # Update config section (safe - operating on deep copy)
        if "config" not in site_wandb_args:
            site_wandb_args["config"] = {}
        site_wandb_args["config"]["job_id"] = job_id_tag
        site_wandb_args["config"]["client"] = site_name
        site_wandb_args["config"]["run_name"] = run_name

        # Validate site-specific config
        _check_wandb_args(site_wandb_args)

        # Create queue and thread, store in module-level dicts
        # Use threading instead of multiprocessing to avoid pickling issues in SimEnv
        q = queue.Queue()
        wandb_task = WandBTask(task_owner=site_name, task_type="init", task_data=site_wandb_args, step=0)
        q.put(wandb_task)

        _WANDB_QUEUES[site_key] = q
        t = threading.Thread(target=self._process_queue_tasks, args=(q,), daemon=True)
        _WANDB_PROCESSES[site_key] = t
        t.start()
        time.sleep(0.2)

    def save(self, fl_ctx: FLContext, shareable: Shareable, record_origin: str):
        dxo = from_shareable(shareable)
        data = AnalyticsData.from_dxo(dxo, receiver=LogWriterName.WANDB)
        if not data:
            return

        # Lazily initialize the site's process on first use
        site_key = (self.run_id, record_origin)
        if site_key not in _WANDB_PROCESSES:
            self._init_site_process(record_origin)

        q: Optional[queue.Queue] = _WANDB_QUEUES.get(site_key)
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
        # Iterate over all threads for this run_id
        site_keys_to_finalize = [key for key in _WANDB_PROCESSES.keys() if key[0] == self.run_id]

        for site_key in site_keys_to_finalize:
            site_name = site_key[1]
            self.log_info(fl_ctx, f"inform {site_name} to stop")
            q = _WANDB_QUEUES.get(site_key)
            if q:
                q.put(WandBTask(task_owner=site_name, task_type="stop", task_data={}, step=0))

        for site_key in site_keys_to_finalize:
            t = _WANDB_PROCESSES.get(site_key)
            if t and t.is_alive():
                t.join(self.process_timeout)
            # Clean up from module-level dicts
            _WANDB_PROCESSES.pop(site_key, None)
            _WANDB_QUEUES.pop(site_key, None)
