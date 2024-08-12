# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from abc import ABC
from typing import Any, List, Union

from nvflare.apis.executor import Executor
from nvflare.apis.filter import Filter, FilterType
from nvflare.apis.impl.controller import Controller
from nvflare.app_common.widgets.convert_to_fed_event import ConvertToFedEvent
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator
from nvflare.fuel.utils.import_utils import optional_import
from nvflare.job_config.fed_job import FedJob

torch, torch_ok = optional_import(module="torch")
tb, tb_ok = optional_import(module="tensorboard")
if torch_ok and tb_ok:
    from nvflare.app_opt.tracking.tb.tb_receiver import TBAnalyticsReceiver


class JobObj(ABC):
    def __init__(self, obj: Any, resources: Union[str, List[str]] = None):
        """JobObj is used in the FedJob API as wrapper for a base object along with any related external resources.
        Args:
            obj: the base object.
            resources: the filenames or directories to be included in custom directory.
        """
        self.obj = obj
        self.resources = resources

    def add_to_job(self, job: FedJob, target: str):
        """Callback for FedJob to add objects or resources to job."""
        job.add_object(self.obj, target)
        job.add_external_resources(self.resources, target)


class ControllerJobObj(JobObj):
    def __init__(
        self,
        controller: Controller,
        resources: Union[str, List[str]] = None,
        key_metric: str = "accuracy",
    ):
        """Controller JobObj Wrapper.
        Args:
            controller: the Controller object.
            resources: the filenames or directories to be included in custom directory.
        """
        super().__init__(
            obj=controller,
            resources=resources,
        )
        self.key_metric = key_metric

    def add_to_job(self, job: FedJob, target: str):
        super().add_to_job(job, target)
        if len(job._deploy_map) == 1:
            job.add_object(obj=ValidationJsonGenerator(), target=target, id="json_generator")
            job.add_object(obj=IntimeModelSelector(key_metric=self.key_metric), target=target, id="model_selector")
            job.add_object(obj=TBAnalyticsReceiver(events=["fed.analytix_log_stats"]), target=target, id="receiver")


class ExecutorJobObj(JobObj):
    def __init__(
        self,
        executor: Executor,
        resources: Union[str, List[str]] = None,
        tasks: List[str] = None,
        gpu: Union[int, List[int]] = None,
    ):
        """Executor JobObj Wrapper.
        Args:
            executor: the Executor object.
            resources: the filenames or directories to be included in custom directory.
            tasks: List of tasks the executor should handle. Defaults to `None`. If `None`, all tasks will be handled using `[*]`.
            gpu: GPU index or list of GPU indices used for simulating the run on that target.
        """
        super().__init__(
            obj=executor,
            resources=resources,
        )
        self.tasks = tasks
        self.gpu = gpu

    def add_to_job(self, job: FedJob, target: str):
        job.add_object(obj=self.obj, target=target, tasks=self.tasks, gpu=self.gpu)
        job.add_external_resources(self.resources, target=target)
        if len(job._deploy_map) == 1:
            job.add_object(
                obj=ConvertToFedEvent(events_to_convert=["analytix_log_stats"], fed_event_prefix="fed."),
                target=target,
                id="event_to_fed",
            )


class FilterJobObj(JobObj):
    def __init__(self, filter: Filter, tasks: List[str] = None, filter_type: FilterType = None):
        """Filter JobObj Wrapper.
        Args:
            filter: the Filter object.
            resources: the filenames or directories to be included in custom directory.
            tasks: List of tasks the filter should handle. Defaults to `None`. If `None`, all tasks will be handled using `[*]`.
            filter_type: The type of filter used. Either `FilterType.TASK_RESULT` or `FilterType.TASK_DATA`.
        """
        super().__init__(
            obj=filter,
        )
        self.tasks = tasks
        self.filter_type = filter_type

    def add_to_job(self, job: FedJob, target: str):
        job.add_object(obj=self.obj, target=target, tasks=self.tasks, filter_type=self.filter_type)
        job.add_external_resources(self.resources, target=target)
