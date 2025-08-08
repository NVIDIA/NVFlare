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
import os
from typing import Dict, Optional

import torch.nn as nn

from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.edge.aggregators.model_update_dxo_factory import ModelUpdateDXOAggrFactory
from nvflare.edge.assessors.buff_device_manager import BuffDeviceManager
from nvflare.edge.assessors.buff_model_manager import BuffModelManager
from nvflare.edge.assessors.model_update import ModelUpdateAssessor
from nvflare.edge.simulation.device_task_processor import DeviceTaskProcessor
from nvflare.edge.tools.edge_job import EdgeJob
from nvflare.edge.widgets.evaluator import GlobalEvaluator
from nvflare.recipe.spec import Recipe


class ModelManagerConfig:

    def __init__(
        self,
        max_num_active_model_versions: int = 3,
        max_model_version: int = 20,
        update_timeout: int = 5.0,
        num_updates_for_model: int = 100,
        max_model_history: int = 10,
        staleness_weight: bool = False,
        global_lr: float = 0.0001,
    ):
        self.max_num_active_model_versions = max_num_active_model_versions
        self.max_model_version = max_model_version
        self.update_timeout = update_timeout
        self.num_updates_for_model = num_updates_for_model
        self.max_model_history = max_model_history
        self.staleness_weight = staleness_weight
        self.global_lr = global_lr


class DeviceManagerConfig:

    def __init__(
        self,
        device_selection_size: int = 100,
        min_hole_to_fill: int = 1,
        device_reuse: bool = True,
        const_selection: bool = False,
    ):
        self.device_selection_size = device_selection_size
        self.min_hole_to_fill = min_hole_to_fill
        self.device_reuse = device_reuse
        self.const_selection = const_selection


class SimulationConfig:
    def __init__(
        self,
        task_processor: Optional[DeviceTaskProcessor],
        job_timeout: float = 60.0,
        num_devices: int = 1000,
        num_workers: int = 10,
    ):
        self.task_processor = task_processor
        self.job_timeout = job_timeout
        self.num_devices = num_devices
        self.num_workers = num_workers


class EvaluatorConfig:

    def __init__(
        self,
        eval_frequency: int = 1,
        torchvision_dataset: Optional[Dict] = None,
        custom_dataset: Optional[Dict] = None,
    ):
        self.eval_frequency = eval_frequency
        self.torchvision_dataset = torchvision_dataset
        self.custom_dataset = custom_dataset


class EdgeRecipe(Recipe):

    def __init__(
        self,
        job_name: str,
        model: nn.Module,
        model_manager_config: ModelManagerConfig,
        device_manager_config: DeviceManagerConfig,
        evaluator_config: EvaluatorConfig = None,
        simulation_config: SimulationConfig = None,
        custom_source_root: str = None,
    ):
        if not isinstance(model, nn.Module):
            raise ValueError(f"model must be a nn.Module but got {type(model)}")

        if custom_source_root and not os.path.isdir(custom_source_root):
            raise ValueError(f"{custom_source_root} is not a valid directory")

        self.job_name = job_name
        self.method_name = "edge"
        self.model = model
        self.model_manager_config = model_manager_config
        self.device_manager_config = device_manager_config
        self.evaluator_config = evaluator_config
        self.simulation_config = simulation_config
        self.custom_source_root = custom_source_root
        job = self.create_job()
        self._configure_job(job)
        Recipe.__init__(self, job)

    def create_job(self) -> EdgeJob:
        return EdgeJob(name=self.job_name, edge_method=self.method_name)

    def _configure_job(self, job: EdgeJob):
        if self.evaluator_config:
            evaluator = GlobalEvaluator(
                model_path=self.model,
                torchvision_dataset=self.evaluator_config.torchvision_dataset,
                eval_frequency=self.evaluator_config.eval_frequency,
                custom_dataset=self.evaluator_config.custom_dataset,
            )
            job.to_server(evaluator, id="evaluator")

        if self.simulation_config:
            c = self.simulation_config
            job.configure_simulation(c.task_processor, c.job_timeout, c.num_devices, c.num_workers)

        factory = ModelUpdateDXOAggrFactory()
        job.configure_client(
            aggregator_factory=factory,
            max_model_versions=self.model_manager_config.max_num_active_model_versions,
            update_timeout=self.model_manager_config.update_timeout,
        )

        # add persistor, model_manager, and device_manager
        persistor_id = job.to_server(PTFileModelPersistor(model=self.model), id="persistor")

        model_manager = BuffModelManager(
            num_updates_for_model=self.model_manager_config.num_updates_for_model,
            max_model_history=self.model_manager_config.max_model_history,
            global_lr=self.model_manager_config.global_lr,
            staleness_weight=self.model_manager_config.staleness_weight,
        )
        model_manager_id = job.to_server(model_manager, id="model_manager")

        device_manager = BuffDeviceManager(
            device_selection_size=self.device_manager_config.device_selection_size,
            min_hole_to_fill=self.device_manager_config.min_hole_to_fill,
            device_reuse=self.device_manager_config.device_reuse,
            const_selection=self.device_manager_config.const_selection,
        )
        device_manager_id = job.to_server(device_manager, id="device_manager")

        # add model_update_assessor
        assessor = ModelUpdateAssessor(
            persistor_id=persistor_id,
            model_manager_id=model_manager_id,
            device_manager_id=device_manager_id,
            max_model_version=self.model_manager_config.max_model_version,
        )
        job.configure_server(
            assessor=assessor,
        )

        if self.custom_source_root:
            job.to_server(self.custom_source_root)
            job.to_clients(self.custom_source_root)
