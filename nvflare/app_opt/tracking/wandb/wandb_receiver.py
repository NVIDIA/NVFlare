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

import json
import os
import time
from configparser import ConfigParser
from copy import deepcopy
from multiprocessing import Process, Queue
from pathlib import Path
from typing import List, NamedTuple, Optional
from urllib.parse import urlsplit

import requests
import wandb

from nvflare.apis.analytix import AnalyticsData, AnalyticsDataType, LogWriterName
from nvflare.apis.dxo import from_shareable
from nvflare.apis.fl_constant import ProcessType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.widgets.streaming import AnalyticsReceiver
from nvflare.private.fed.resource_stats.cpu_consumed import mark_child_cpu_coverage_uncertain


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


def _normalize_base_url(url):
    if not isinstance(url, str) or not url or any(c.isspace() for c in url):
        raise ValueError("W&B base URL must be a nonempty HTTP(S) URL")
    parsed = urlsplit(url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("W&B base URL must be HTTP(S), without credentials, query or fragment")
    # Compare the destination the installed SDK will actually use, including
    # its supported cloud URL aliases and trailing-slash normalization.
    return wandb.Settings(base_url=url).base_url


def _operator_base_url():
    configured = os.environ.get("WANDB_BASE_URL")
    if configured:
        return _normalize_base_url(configured)

    # Match the SDK's global settings location. Job-local wandb/settings is
    # deliberately excluded from the authority to choose a backend.
    config_dir = Path(os.environ.get("WANDB_CONFIG_DIR") or "~/.config/wandb").expanduser()
    settings = ConfigParser(interpolation=None)
    try:
        with (config_dir / "settings").open() as settings_file:
            settings.read_file(settings_file)
    except FileNotFoundError:
        pass
    return _normalize_base_url(settings.get("default", "base_url", fallback=wandb.Settings().base_url))


def _get_base_url(settings):
    return settings["base_url"] if isinstance(settings, dict) else settings.base_url


def _verify_backend(base_url):
    credentials = requests.utils.get_netrc_auth(base_url)
    key = os.environ.get("WANDB_API_KEY") or (credentials[1] if credentials else None)
    if not key:
        raise ValueError("No W&B API key available for the selected backend")
    response = requests.post(
        f"{base_url}/graphql",
        json={"query": "query { viewer { id } }"},
        auth=("api", key),
        timeout=(1.0, 1.0),
        allow_redirects=False,
    )
    if response.status_code != 200 or not (response.json().get("data") or {}).get("viewer"):
        raise ValueError("The selected W&B backend did not verify the API key")


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

    def _get_wandb_args(self, fl_ctx):
        """Pin credential-bearing traffic to a site-approved backend before SDK login."""
        args = deepcopy(self.wandb_args)
        settings = args.get("settings")
        if settings is None:
            settings = {}
        if not isinstance(settings, (dict, wandb.Settings)):
            raise ValueError("wandb_args.settings must be a dict or wandb.Settings")

        def get_setting(key):
            if isinstance(settings, dict):
                return settings.get(key)
            if key == "base_url" and key not in getattr(settings, "model_fields_set", {key}):
                return None
            return getattr(settings, key, None)

        # Proxy and TLS overrides can bypass the destination restriction.
        if any(
            get_setting(key) for key in ("http_proxy", "https_proxy", "x_proxies", "_proxies", "insecure_disable_ssl")
        ):
            raise ValueError("job W&B settings cannot override proxies or disable TLS verification")

        default_url = _operator_base_url()
        # Only the receiving site's canonical resources file can authorize an
        # additional backend. ConfigService also searches job-controlled paths.
        resources_file = fl_ctx.get_workspace().get_resources_file_path()
        resources = {}
        if resources_file and os.path.isfile(resources_file):
            with open(resources_file) as f:
                resources = json.load(f)
        if not isinstance(resources, dict):
            raise ValueError("site resources must be a JSON object")
        allowed_urls = resources.get("wandb_base_urls", [])
        if not isinstance(allowed_urls, list):
            raise ValueError("site wandb_base_urls must be a list of HTTP(S) URLs")
        approved = {default_url, *(_normalize_base_url(url) for url in allowed_urls)}
        base_url = _normalize_base_url(get_setting("base_url") or default_url)
        if base_url not in approved:
            raise ValueError("W&B base_url requires approval in site wandb_base_urls")
        if isinstance(settings, dict):
            settings["base_url"] = base_url
            settings["insecure_disable_ssl"] = False
        else:
            settings.base_url = base_url
            settings.insecure_disable_ssl = False
        args["settings"] = settings
        return args

    def _process_queue_tasks(self, queue):
        cnt = 0
        run = None
        current_step = 0
        original_base_url = os.environ.get("WANDB_BASE_URL")
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
                    settings = wandb_task.task_data.get("settings")
                    if settings is not None:
                        # Older SDKs consult this environment variable during
                        # login even when the per-run endpoint is explicit.
                        os.environ["WANDB_BASE_URL"] = _get_base_url(settings)
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
            try:
                if run:
                    run.finish()
            finally:
                if original_base_url is None:
                    os.environ.pop("WANDB_BASE_URL", None)
                else:
                    os.environ["WANDB_BASE_URL"] = original_base_url

    def initialize(self, fl_ctx: FLContext):
        wandb_args = self._get_wandb_args(fl_ctx)
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

        run_name = wandb_args["name"]
        job_id_tag = _get_job_id_tag(fl_ctx)

        if self.mode == "online":
            try:
                _verify_backend(_get_base_url(wandb_args["settings"]))
            except Exception:
                self.log_warning(fl_ctx, "Could not verify the selected W&B backend. Using offline mode.")
                self.mode = "offline"

        for site_name in site_names:
            self.log_info(fl_ctx, f"initialize WandB run for site {site_name}")
            # Queue serialization runs asynchronously: each worker gets its own
            # validated snapshot rather than mutable job constructor arguments.
            task_args = deepcopy(wandb_args)
            wand_config = task_args.get("config", {})
            task_args["name"] = f"{site_name}-{job_id_tag}-{run_name}"
            task_args["group"] = f"{run_name}-{job_id_tag}"
            task_args["mode"] = self.mode
            wand_config["job_id"] = job_id_tag
            wand_config["client"] = site_name
            wand_config["run_name"] = run_name

            _check_wandb_args(task_args)

            q = Queue()
            wandb_task = WandBTask(task_owner=site_name, task_type="init", task_data=task_args, step=0)
            q.put(wandb_task)

            self.queues[site_name] = q
            p = Process(target=self._process_queue_tasks, args=(q,))
            self.processes[site_name] = p
            p.start()
            mark_child_cpu_coverage_uncertain()
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
