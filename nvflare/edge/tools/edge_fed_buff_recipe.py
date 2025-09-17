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
from nvflare.recipe.spec import ExecEnv, Recipe

DEVICE_SIMULATION_ENV_KEY = "device_simulation"


class ModelManagerConfig:
    """Configuration class for the model manager in federated learning.

    This class configures how the model manager handles model updates, versioning,
    and aggregation strategies for federated learning workflow.

    Attributes:
        max_num_active_model_versions: Maximum number of active model versions
            that can be processed for the current model version. Default: 3
        max_model_version: Maximum model version number before stopping training.
            We start with version 1 initial model, so the minimum for this arg is 2 to have at least one local update phase.
            Default: 20
        update_timeout: Timeout in seconds for waiting for model updates.
            Default: 5.0
        num_updates_for_model: Number of received updates required before generating
            a new global model. Default: 100
        max_model_history: Maximum number of model versions to keep in history
            for staleness calculations and update aggregation. Default: 10
        staleness_weight: Whether to apply staleness weighting to model updates.
            Default: False
        global_lr: Global learning rate for model aggregation. Default: 0.01
    """

    def __init__(
        self,
        max_num_active_model_versions: int = 3,
        max_model_version: int = 20,
        update_timeout: int = 5,
        num_updates_for_model: int = 100,
        max_model_history: int = 10,
        staleness_weight: bool = False,
        global_lr: float = 0.01,
    ):
        self.max_num_active_model_versions = max_num_active_model_versions
        self.max_model_version = max_model_version
        # check if max_model_version is greater than 2
        if max_model_version < 2:
            raise ValueError("max_model_version needs to be at least 2 to have at least one local update phase")

        self.update_timeout = update_timeout
        self.num_updates_for_model = num_updates_for_model
        self.max_model_history = max_model_history
        self.staleness_weight = staleness_weight
        self.global_lr = global_lr


class DeviceManagerConfig:
    """Configuration class for the device manager in federated learning.

    This class configures how the device manager selects and manages devices
    for participation in federated learning workflow.

    Attributes:
        device_selection_size: Number of devices to select for each training round.
            Default: 100
        min_hole_to_fill: Minimum number of model updates to wait for before
            sampling the next batch of devices and dispatching the current global model.
            - If set to 1, the server immediately dispatch the current global model to a sampled device.
            - Higher values cause the server to wait for more updates before dispatching.
            - If set to device_selection_size, we will have synchronous training since all devices' responses need to be collected before dispatching the next global model.
            This parameter works with num_updates_for_model from model manager to achieve trade-off between global model versioning and local execution.
            Default: 1 (immediately dispatch the current global model)
        device_reuse: Whether to allow devices to participate in multiple rounds.
            if False, devices will be selected only once, which could be realistic for real-world scenarios where the
            device pool is huge while participation is random.
            Default: True (always reuse / include the existing devices for further learning)
    """

    def __init__(
        self,
        device_selection_size: int = 100,
        min_hole_to_fill: int = 1,
        device_reuse: bool = True,
    ):
        self.device_selection_size = device_selection_size
        self.min_hole_to_fill = min_hole_to_fill
        # check if min_hole_to_fill is smaller than device_selection_size
        if min_hole_to_fill > device_selection_size:
            raise ValueError("min_hole_to_fill needs to be smaller than or equal to device_selection_size")
        self.device_reuse = device_reuse


class SimulationConfig:
    """Configuration class for simulation settings in federated learning.

    This class configures the simulated devices for testing federated learning
    pipelines.

    Attributes:
        task_processor: Task processor for handling device training simulation.
        job_timeout: Timeout in seconds for the entire job execution. Default: 60.0
        num_devices: Total number of simulated devices for each leaf node. Default: 1000
        num_workers: Number of worker processes for parallel device simulation on each leaf node.
            Default: 10
    """

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
    """Configuration class for the global evaluator.

    This class configures how the global model is evaluated during training,
    including dataset selection and evaluation frequency.

    Attributes:
        eval_frequency: Frequency of global model evaluation (every N new model versions).
            Default: 1
        torchvision_dataset: Configuration for torchvision datasets. Should be a
            dict with 'name' and 'path' keys. Default: None
        custom_dataset: Configuration for custom datasets. Default: None
    """

    def __init__(
        self,
        eval_frequency: int = 1,
        torchvision_dataset: Optional[Dict] = None,
        custom_dataset: Optional[Dict] = None,
    ):
        self.eval_frequency = eval_frequency
        self.torchvision_dataset = torchvision_dataset
        self.custom_dataset = custom_dataset


class EdgeFedBuffRecipe(Recipe):
    """Recipe class for cross-edge federated learning using NVFlare's hierarchical edge system.

    This class provides a high-level interface for configuring cross-edge
    federated learning jobs. It configures the necessary components
    including model managers, device managers, evaluators, and device simulation settings.

    The recipe supports both real device connections and simulated device training,
    making it suitable for both production deployment and prototyping/testing.

    Example usage:
        ```python
        # Basic configuration
        model_manager_config = ModelManagerConfig(
            global_lr=0.1,
            num_updates_for_model=20,
            max_model_version=300,
            max_model_history=100
        )

        device_manager_config = DeviceManagerConfig(
            device_selection_size=200,
            min_hole_to_fill=10,
            device_reuse=False
        )

        recipe = EdgeFedBuffRecipe(
            job_name="my_edge_job",
            model=MyModel(),
            model_manager_config=model_manager_config,
            device_manager_config=device_manager_config
        )
        ```

    Attributes:
        job_name: Name of the federated learning job
        model: PyTorch neural network model to be trained
        model_manager_config: Configuration for the model manager
        device_manager_config: Configuration for the device manager
        evaluator_config: Configuration for the global evaluator (optional)
        simulation_config: Configuration for simulated devices settings (optional)
        custom_source_root: Path to custom source code (optional)
    """

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
        # check if model_manager_config.num_updates_for_model is smaller than device_manager_config.device_selection_size
        if model_manager_config.num_updates_for_model > device_manager_config.device_selection_size:
            raise ValueError(
                f"model_manager_config.num_updates_for_model {model_manager_config.num_updates_for_model} "
                f"needs to be smaller than or equal to device_manager_config.device_selection_size {device_manager_config.device_selection_size}, "
                "otherwise the server will never have enough updates to update the global model"
            )

        job = self.create_job()
        self._configure_job(job)
        Recipe.__init__(self, job)

    @staticmethod
    def _configure_simulation(job, c: SimulationConfig):
        job.configure_simulation(c.task_processor, c.job_timeout, c.num_devices, c.num_workers)

    def process_env(self, env: ExecEnv):
        simulation_config = env.get_extra_prop(DEVICE_SIMULATION_ENV_KEY)
        if not simulation_config:
            return

        if not isinstance(simulation_config, SimulationConfig):
            raise ValueError(
                f"invalid {DEVICE_SIMULATION_ENV_KEY} in env: expect SimulationConfig but got {type(simulation_config)}"
            )

        assert isinstance(self.job, EdgeJob)
        self._configure_simulation(self.job, simulation_config)

    def create_job(self) -> EdgeJob:
        """Create a new EdgeJob instance for cross-edge federated learning.

        Returns:
            EdgeJob: A configured edge job instance
        """
        return EdgeJob(name=self.job_name, edge_method=self.method_name)

    def _configure_job(self, job: EdgeJob):
        """Configure the edge job with all necessary components.

        This method sets up the job with evaluators, simulation settings,
        client configurations, and server-side components including model
        managers, device managers, and assessors.

        Args:
            job: The EdgeJob instance to configure
        """
        if self.evaluator_config:
            evaluator = GlobalEvaluator(
                model_path=self.model,
                torchvision_dataset=self.evaluator_config.torchvision_dataset,
                eval_frequency=self.evaluator_config.eval_frequency,
                custom_dataset=self.evaluator_config.custom_dataset,
            )
            job.to_server(evaluator, id="evaluator")

        if self.simulation_config:
            self._configure_simulation(job, self.simulation_config)

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
