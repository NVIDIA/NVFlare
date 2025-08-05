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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch.nn as nn

from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

# Import aggregators
from nvflare.edge.aggregators.model_update_dxo_factory import ModelUpdateDXOAggrFactory
from nvflare.edge.aggregators.num_dxo_factory import NumDXOAggrFactory
from nvflare.edge.assessors.async_num import AsyncNumAssessor
from nvflare.edge.assessors.buff_device_manager import BuffDeviceManager
from nvflare.edge.assessors.buff_model_manager import BuffModelManager

# Import assessors
from nvflare.edge.assessors.model_update import ModelUpdateAssessor
from nvflare.edge.assessors.num import NumAssessor

# Import edge job
from nvflare.edge.edge_job import EdgeJob

# Import executors
from nvflare.edge.executors.edge_model_executor import EdgeModelExecutor
from nvflare.edge.executors.et_edge_model_executor import ETEdgeModelExecutor
from nvflare.edge.models.model import DeviceModel

# Import other components
from nvflare.edge.widgets.evaluator import GlobalEvaluator
from nvflare.recipe.spec import Recipe


@dataclass
class AggregationConfig:
    """Configuration for model aggregation."""

    max_model_versions: int = 10
    max_model_history: int = 1
    num_updates_for_model: int = 16
    global_lr: float = 1.0
    staleness_weight: float = 1.0
    update_timeout: float = 1000.0


@dataclass
class DeviceConfig:
    """Configuration for device selection and management."""

    selection_size: int = 16
    min_hole_to_fill: int = 16
    device_reuse: bool = False
    const_selection: bool = False


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""

    enabled: bool = False
    model_path: Optional[str] = None
    dataset_config: Optional[Dict[str, Any]] = None
    frequency: int = 1


class EdgeJobRecipe(Recipe, ABC):
    """Base class for EdgeJob recipes."""

    def __init__(
        self,
        name: str,
        edge_method: str = "cnn",
        aggregation_config: AggregationConfig = None,
        device_config: DeviceConfig = None,
        evaluation_config: EvaluationConfig = None,
        min_clients: int = 1,
    ):
        self.name = name
        self.edge_method = edge_method
        self.aggregation_config = aggregation_config or AggregationConfig()
        self.device_config = device_config or DeviceConfig()
        self.evaluation_config = evaluation_config or EvaluationConfig()

        # Create EdgeJob
        self.job = EdgeJob(name=name, edge_method=edge_method, min_clients=min_clients)

        # Configure the job
        self._configure_job(self.job)

        Recipe.__init__(self, self.job)

    def configure_simulation_with_file(self, config_file=None):
        """Add simulation support - delegates to underlying job."""
        return self.job.configure_simulation_with_file(config_file)

    def configure_simulation(self, task_processor=None, num_devices=100, num_workers=10, job_timeout=60.0):
        """Add simulation support - delegates to underlying job."""
        return self.job.configure_simulation(task_processor, num_devices, num_workers, job_timeout)

    @abstractmethod
    def _configure_job(self, job: EdgeJob):
        """Configure the EdgeJob with recipe-specific components."""
        pass


class ModelUpdateEdgeRecipe(EdgeJobRecipe, ABC):
    """Base recipe for model-based edge jobs (PyTorch and ExecutorTorch)."""

    def __init__(
        self,
        name: str,
        model: Union[nn.Module, str, type],
        # Expose controller params
        num_rounds: int = 10,
        task_name: str = "train",
        assess_interval: float = 0.5,
        update_interval: float = 1.0,
        # Expose common aggregation params
        max_model_versions: int = 10,
        num_updates_for_model: int = 16,
        max_model_history: int = 1,
        global_lr: float = 1.0,
        staleness_weight: float = 1.0,
        update_timeout: float = 1000.0,
        # Expose common device params
        device_selection_size: int = 16,
        min_hole_to_fill: int = 16,
        device_reuse: bool = False,
        const_selection: bool = False,
        # Expose evaluation params
        enable_evaluation: bool = False,
        eval_model_path: Optional[str] = None,
        eval_dataset_config: Optional[Dict[str, Any]] = None,
        eval_frequency: int = 1,
        **kwargs,
    ):
        self.model = model
        self.num_rounds = num_rounds
        self.task_name = task_name
        self.assess_interval = assess_interval
        self.update_interval = update_interval

        # Create config objects from exposed parameters
        aggregation_config = kwargs.get("aggregation_config") or AggregationConfig(
            max_model_versions=max_model_versions,
            num_updates_for_model=num_updates_for_model,
            max_model_history=max_model_history,
            global_lr=global_lr,
            staleness_weight=staleness_weight,
            update_timeout=update_timeout,
        )

        device_config = kwargs.get("device_config") or DeviceConfig(
            selection_size=device_selection_size,
            min_hole_to_fill=min_hole_to_fill,
            device_reuse=device_reuse,
            const_selection=const_selection,
        )

        evaluation_config = kwargs.get("evaluation_config") or EvaluationConfig(
            enabled=enable_evaluation,
            model_path=eval_model_path,
            dataset_config=eval_dataset_config,
            frequency=eval_frequency,
        )

        super().__init__(
            name=name,
            aggregation_config=aggregation_config,
            device_config=device_config,
            evaluation_config=evaluation_config,
            **kwargs,
        )

    def _configure_job(self, job: EdgeJob):
        # Configure client using EdgeJob's clean method
        job.configure_client(
            aggregator_factory=ModelUpdateDXOAggrFactory(),
            executor=self._create_executor(),
            max_model_versions=self.aggregation_config.max_model_versions,
            update_timeout=self.aggregation_config.update_timeout,
            executor_task_name=self.task_name,
        )

        # Configure server using EdgeJob's clean method
        job.configure_server(
            assessor=self._create_assessor(job),  # Pass job to handle server components
            num_rounds=self.num_rounds,
            task_name=self.task_name,
            assess_interval=self.assess_interval,
            update_interval=self.update_interval,
        )

        # Configure evaluation if enabled
        if self.evaluation_config.enabled:
            self._configure_evaluation(job)

    @abstractmethod
    def _create_executor(self):
        """Create executor specific to model format."""
        pass

    def _create_assessor(self, job: EdgeJob):
        """Create ModelUpdateAssessor and configure server components."""
        # Configure model-based server components
        model_instance = self._create_model_instance()
        persistor = PTFileModelPersistor(model=model_instance)
        job.to_server(persistor, id="persistor")

        model_manager = BuffModelManager(
            num_updates_for_model=self.aggregation_config.num_updates_for_model,
            max_model_history=self.aggregation_config.max_model_history,
            global_lr=self.aggregation_config.global_lr,
            staleness_weight=self.aggregation_config.staleness_weight,
        )
        job.to_server(model_manager, id="model_manager")

        device_manager = BuffDeviceManager(
            device_selection_size=self.device_config.selection_size,
            min_hole_to_fill=self.device_config.min_hole_to_fill,
            device_reuse=self.device_config.device_reuse,
            const_selection=self.device_config.const_selection,
        )
        job.to_server(device_manager, id="device_manager")

        # Return the assessor
        return ModelUpdateAssessor(
            persistor_id="persistor",
            model_manager_id="model_manager",
            device_manager_id="device_manager",
            max_model_version=self.aggregation_config.max_model_versions,
        )

    def _create_model_instance(self):
        """Create model instance from config."""
        if isinstance(self.model, str):
            # Import model class from string path
            module_path, class_name = self.model.rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            model_class = getattr(module, class_name)
            return model_class()
        elif isinstance(self.model, type):
            # Model class provided
            return self.model()
        else:
            # Model instance provided
            return self.model

    def _configure_evaluation(self, job: EdgeJob):
        """Configure global evaluation."""
        if not self.evaluation_config.model_path:
            # Auto-determine model path
            if isinstance(self.model, str):
                model_path = self.model
            elif hasattr(self.model, "__module__") and hasattr(self.model, "__name__"):
                model_path = f"{self.model.__module__}.{self.model.__name__}"
            else:
                model_path = f"{self.model.__class__.__module__}.{self.model.__class__.__name__}"
        else:
            model_path = self.evaluation_config.model_path

        evaluator = GlobalEvaluator(
            model_path=model_path,
            torchvision_dataset=self.evaluation_config.dataset_config,
            eval_frequency=self.evaluation_config.frequency,
        )
        job.to_server(evaluator, id="evaluator")


# PyTorch Recipes
class PyTorchEdgeRecipe(ModelUpdateEdgeRecipe):
    """Recipe for PyTorch edge jobs (handles both sync and async via configuration)."""

    def _create_executor(self):
        return EdgeModelExecutor(
            aggr_factory_id="aggr_factory",  # EdgeJob will create this with this ID
            max_model_versions=self.aggregation_config.max_model_versions,
            update_timeout=self.aggregation_config.update_timeout,
        )


# ExecutorTorch Recipes
class ETEdgeRecipe(ModelUpdateEdgeRecipe):
    """Recipe for ExecutorTorch edge jobs (handles both sync and async via configuration)."""

    def __init__(self, name: str, et_model, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...], **kwargs):
        if not isinstance(et_model, DeviceModel):
            raise ValueError("et_model must be a DeviceModel for ExecutorTorch jobs")

        self.input_shape = input_shape
        self.output_shape = output_shape
        super().__init__(name=name, model=et_model, **kwargs)

    def _create_model_instance(self):
        """Create model instance for persistor - extract underlying model from DeviceModel."""
        # Handle DeviceModel - extract the underlying net for persistor
        return self.model.net

    def _create_executor(self):
        # Create proper DeviceModel config with nested net
        et_model_config = {
            "path": "nvflare.edge.models.model.DeviceModel",
            "args": {
                "net": {
                    "path": f"{self.model.net.__class__.__module__}.{self.model.net.__class__.__name__}",
                    "args": {},
                }
            },
        }

        return ETEdgeModelExecutor(
            et_model=et_model_config,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            aggr_factory_id="aggr_factory",  # EdgeJob will create this with this ID
            max_model_versions=self.aggregation_config.max_model_versions,
            update_timeout=self.aggregation_config.update_timeout,
        )


# Numpy Recipes
class NumpyEdgeRecipe(EdgeJobRecipe, ABC):
    """Base recipe for numpy-based edge jobs."""

    def __init__(
        self,
        name: str,
        # Expose controller params
        num_rounds: int = 10,
        task_name: str = "train",
        assess_interval: float = 0.5,
        update_interval: float = 1.0,
        # Expose common aggregation params
        num_updates_for_model: int = 10,
        max_model_history: int = 3,
        max_model_versions: int = 10,
        global_lr: float = 1.0,
        staleness_weight: float = 1.0,
        update_timeout: float = 1000.0,
        # Expose common device params
        device_selection_size: int = 30,
        min_hole_to_fill: int = 16,
        device_reuse: bool = False,
        const_selection: bool = False,
        **kwargs,
    ):
        self.num_rounds = num_rounds
        self.task_name = task_name
        self.assess_interval = assess_interval
        self.update_interval = update_interval

        # Create config objects from exposed parameters
        aggregation_config = kwargs.get("aggregation_config") or AggregationConfig(
            num_updates_for_model=num_updates_for_model,
            max_model_history=max_model_history,
            max_model_versions=max_model_versions,
            global_lr=global_lr,
            staleness_weight=staleness_weight,
            update_timeout=update_timeout,
        )

        device_config = kwargs.get("device_config") or DeviceConfig(
            selection_size=device_selection_size,
            min_hole_to_fill=min_hole_to_fill,
            device_reuse=device_reuse,
            const_selection=const_selection,
        )

        # Numpy jobs don't need evaluation
        evaluation_config = kwargs.get("evaluation_config") or EvaluationConfig()

        super().__init__(
            name=name,
            aggregation_config=aggregation_config,
            device_config=device_config,
            evaluation_config=evaluation_config,
            **kwargs,
        )

    def _configure_job(self, job: EdgeJob):
        # Configure client - numpy jobs don't use executors, just aggregator factory
        job.to_clients(NumDXOAggrFactory(), id="aggr_factory")

        # Configure server using EdgeJob's clean method
        job.configure_server(
            assessor=self._create_assessor(job),
            num_rounds=self.num_rounds,
            task_name=self.task_name,
            assess_interval=self.assess_interval,
            update_interval=self.update_interval,
        )

    @abstractmethod
    def _create_assessor(self, job: EdgeJob):
        """Create assessor specific to aggregation strategy."""
        pass


class NumpySyncEdgeRecipe(NumpyEdgeRecipe):
    """Recipe for synchronous numpy edge jobs."""

    def _create_assessor(self, job: EdgeJob):
        return NumAssessor(
            num_updates_for_model=self.aggregation_config.num_updates_for_model,
            max_model_history=self.aggregation_config.max_model_history,
            max_model_version=self.aggregation_config.max_model_versions,
            device_selection_size=self.device_config.selection_size,
        )


class NumpyAsyncEdgeRecipe(NumpyEdgeRecipe):
    """Recipe for asynchronous numpy edge jobs."""

    def _create_assessor(self, job: EdgeJob):
        return AsyncNumAssessor(
            num_updates_for_model=self.aggregation_config.num_updates_for_model,
            max_model_history=self.aggregation_config.max_model_history,
            max_model_version=self.aggregation_config.max_model_versions,
            device_selection_size=self.device_config.selection_size,
        )
