# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import time
from typing import List

from nvflare.apis.dxo import DXO, from_dict
from nvflare.apis.signal import Signal

from .config import process_train_config
from .defs import Context, ContextKey, DataSource, EventType, Executor, Filter


class FlareRunner:

    def __init__(
        self,
        job_name: str,
        data_source: DataSource,
        device_info: dict,
        user_info: dict,
        job_timeout: float,
        in_filters: List[Filter] = None,
        out_filters: List[Filter] = None,
        resolver_registry: dict = None,
    ):
        """Constructor of FlareRunner

        Args:
            job_name: name of the job. Used for matching Flare job on host.
            data_source: data source for the training
            device_info: device info
            user_info: info of the device user
            job_timeout: timeout for getting a job from Flare host
            in_filters: app provided filters for input model
            out_filters: app provided filters for output model
            resolver_registry: app provided resolvers

        Note: app provided filters apply to all jobs and are invoked before configured job filters!
        """
        self.job_name = job_name
        self.resolver_registry = {}
        self.data_source = data_source
        self.device_info = device_info
        self.user_info = user_info
        self.job_timeout = job_timeout
        self.app_in_filters = in_filters
        self.app_out_filters = out_filters
        self.abort_signal = Signal()
        self.job_id = None
        self.cookie = None

        # add built-in creators
        self.add_builtin_resolvers()

        # add app-provided resolvers, which can override builtin resolvers!
        if resolver_registry:
            if not isinstance(resolver_registry, dict):
                raise ValueError(f"resolver_registry must be dict but got {type(resolver_registry)}")
            self.resolver_registry.update(resolver_registry)

    def add_builtin_resolvers(self):
        """Add resolvers for Flare's builtin components

        Returns:

        """
        pass

    def run(self):
        while True:
            sess_done = self._do_one_job()
            if sess_done:
                return

    def stop(self):
        if self.abort_signal:
            self.abort_signal.trigger(True)

    def _get_job(self, ctx: Context, abort_signal: Signal) -> dict:
        """Repeatedly try to get job from host

        Returns: a job or None if the host says DONE or timed out.

        """
        pass

    def _get_task(self, ctx: Context, abort_signal: Signal) -> (dict, bool):
        """Repeatedly try to get a task from the host

        Returns: a task or None if the host says DONE

        """
        pass

    def _report_result(self, result: dict, ctx: Context, abort_signal: Signal) -> bool:
        pass

    def _do_filtering(self, data: DXO, filters, ctx) -> DXO:
        if filters:
            for f in filters:
                assert isinstance(f, Filter)
                data = f.filter(data, ctx, self.abort_signal)
                if self.abort_signal.triggered:
                    break
        return data

    def _do_one_job(self) -> bool:
        """Work with the host to do one job

        Returns: whether whole session is done.

        """
        ctx = Context()
        ctx[ContextKey.RUNNER] = self
        ctx[ContextKey.DATA_SOURCE] = self.data_source

        # try to get job
        job = self._get_job(ctx, self.abort_signal)
        if not job:
            # No job for me.
            return True

        self.job_name = job.get("job_name")
        self.job_id = job.get("job_id")
        job_data = job.get("job_data")

        # the job_data in the job contains trainer config!
        train_config = process_train_config(job_data, self.resolver_registry)
        ctx[ContextKey.COMPONENTS] = train_config.objects
        ctx[ContextKey.EVENT_HANDLERS] = train_config.event_handlers

        in_filters = []
        if self.app_in_filters:
            in_filters.extend(self.app_in_filters)

        if train_config.in_filters:
            in_filters.extend(train_config.in_filters)

        out_filters = []
        if self.app_out_filters:
            out_filters.extend(self.app_out_filters)

        if train_config.out_filters:
            out_filters.extend(train_config.out_filters)

        while True:
            task, sess_done = self._get_task(ctx, self.abort_signal)
            if self.abort_signal.triggered:
                return True

            if not task:
                # no more work for this job
                return sess_done

            # create a new context for each task!
            task_ctx = copy.copy(ctx)

            # task is a dict
            assert isinstance(task, dict)
            self.cookie = task.get("cookie")
            task_name = task.get("task_name")

            # task data is DXO format
            task_data = task.get("task_data")
            task_dxo = from_dict(task_data)

            # find the right executor
            executor = train_config.find_executor(task_name)
            if not executor:
                raise RuntimeError(f"cannot find executor for task {task_name}")

            if not isinstance(executor, Executor):
                raise RuntimeError(f"bad executor for task {task_name}: expect Executor but got {type(executor)}")

            task_ctx[ContextKey.TASK_ID] = task.get("task_id")
            task_ctx[ContextKey.TASK_NAME] = task_name
            task_ctx[ContextKey.TASK_DATA] = task_data
            task_ctx[ContextKey.EXECUTOR] = executor

            # filter the input
            task_dxo = self._do_filtering(task_dxo, in_filters, task_ctx)
            if not isinstance(task_dxo, DXO):
                raise RuntimeError(f"task data after filtering is not valid DXO: {type(task_dxo)}")

            if self.abort_signal.triggered:
                return True

            task_ctx.fire_event(EventType.BEFORE_TRAIN, time.time(), self.abort_signal)
            output = executor.execute(task_dxo, task_ctx, self.abort_signal)

            # output must follow DXO format
            if not isinstance(output, DXO):
                raise RuntimeError(f"output from {type(executor)} is not a valid DXO: {type(output)}")

            task_ctx.fire_event(EventType.AFTER_TRAIN, (time.time(), output), self.abort_signal)

            if self.abort_signal.triggered:
                return True

            # filter the output
            output = self._do_filtering(output, out_filters, task_ctx)
            if not isinstance(output, DXO):
                raise RuntimeError(f"output after filtering for task {task_name} is not a valid DXO: {type(output)}")

            if self.abort_signal.triggered:
                return True

            sess_done = self._report_result(output.to_dict(), task_ctx, self.abort_signal)
            if sess_done:
                return sess_done

            if self.abort_signal.triggered:
                return True
