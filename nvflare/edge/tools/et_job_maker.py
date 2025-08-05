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
from typing import Dict

from nvflare.edge.models.model import DeviceModel
from nvflare.edge.tools.edge_job_maker import EdgeJobMaker
from nvflare.edge.tools.et_job import ETJob


class ETJobMaker(EdgeJobMaker):

    def __init__(
        self,
        job_name: str,
        device_model: DeviceModel,
        input_shape,
        output_shape,
        data_path: str,
        training_config: Dict = None,
        max_num_active_model_versions: int = 3,
        max_model_version: int = 20,
        update_timeout: int = 5.0,
        num_updates_for_model: int = 100,
        max_model_history: int = 10,
        global_lr: float = 0.0001,
        staleness_weight: bool = False,
        device_selection_size: int = 100,
        min_hole_to_fill: int = 1,
        device_reuse: bool = True,
        const_selection: bool = False,
        custom_source_root: str = None,
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.data_path = data_path
        self.training_config = training_config
        EdgeJobMaker.__init__(
            self,
            job_name=job_name,
            device_model=device_model,
            max_num_active_model_versions=max_num_active_model_versions,
            max_model_version=max_model_version,
            update_timeout=update_timeout,
            num_updates_for_model=num_updates_for_model,
            max_model_history=max_model_history,
            global_lr=global_lr,
            staleness_weight=staleness_weight,
            device_selection_size=device_selection_size,
            min_hole_to_fill=min_hole_to_fill,
            device_reuse=device_reuse,
            const_selection=const_selection,
            custom_source_root=custom_source_root,
        )

    def create_job(self):
        return ETJob(
            name=self.job_name,
            edge_method=self.method_name,
            device_model=self.device_model,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            data_path=self.data_path,
            training_config=self.training_config,
        )

    def configure_simulation(
        self,
        task_processor,
        job_timeout: float = 60.0,
        num_devices: int = 1000,
        num_workers: int = 10,
    ):
        # just in case user calls this - ignore the task processor
        self.configure_et_simulation(job_timeout, num_devices, num_workers)

    def configure_et_simulation(
        self,
        job_timeout: float = 60.0,
        num_devices: int = 1000,
        num_workers: int = 10,
    ):
        assert isinstance(self.job, ETJob)
        self.job.configure_et_simulation(job_timeout, num_devices, num_workers)
