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

import base64
import logging
from abc import ABC, abstractmethod
from typing import Dict

from executorch.extension.training import _load_for_executorch_for_training_from_buffer, get_sgd_optimizer
from torch.utils.data import DataLoader, Dataset

from nvflare.apis.dxo import DXO, from_dict
from nvflare.edge.model_protocol import ModelBufferType, ModelEncoding, ModelNativeFormat, verify_payload
from nvflare.edge.simulation.device_task_processor import DeviceTaskProcessor
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.task_response import TaskResponse

log = logging.getLogger(__name__)


def tensor_dict_to_json(d):
    j = {}
    for k, v in d.items():
        entry = {}
        # Note: This needs to be compatible with the "NVFlare system" logic
        #       for example: nvflare/edge/executors/et_edge_model_executor.py
        entry["data"] = v.cpu().numpy().tolist()
        entry["sizes"] = list(v.size())
        j[k] = entry
    return j


def clone_params(et_params):
    params = {}
    for k, v in et_params.items():
        params[k] = v.clone()
    return params


def calc_params_diff(initial_p, last_p):
    diff_p = {}
    for k, v in initial_p.items():
        diff_p[k] = last_p[k] - v
    return diff_p


class ETTaskProcessor(DeviceTaskProcessor, ABC):
    """Base ExecutorTorch task processor."""

    def __init__(
        self,
        data_path: str,
        training_config: Dict = None,
    ):
        """Initialize the task processor.

        Args:
            data_path: Path to the dataset
            training_config: Configuration for training including:
                - batch_size (int): Size of each training batch (default: 32)
                - shuffle (bool): Whether to shuffle the dataset (default: True)
                - num_workers (int): Number of worker processes for data loading (default: 0)
                - learning_rate (float): Learning rate for optimization (default: 0.1)
                - momentum (float): Momentum factor (default: 0.0)
                - weight_decay (float): Weight decay factor (default: 0.0)
                - dampening (float): Dampening for momentum (default: 0.0)
                - nesterov (bool): Enables Nesterov momentum (default: False)
        """
        DeviceTaskProcessor.__init__(self)
        self.data_path = data_path
        self._dataset = None

        # Set default training configuration
        self.training_config = {
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 0,
            "learning_rate": 0.1,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "dampening": 0.0,
            "nesterov": False,
        }
        # Update with user-provided config
        if training_config:
            self.training_config.update(training_config)

    @abstractmethod
    def create_dataset(self, data_path: str) -> Dataset:
        """Create dataset for training.

        Note: This method may perform expensive I/O operations.

        Args:
            data_path: Path to dataset

        Returns:
            Dataset: PyTorch dataset for training
        """
        pass

    def get_dataset(self) -> Dataset:
        """Get the dataset, creating it if necessary (cached)."""
        if self._dataset is None:
            self._dataset = self.create_dataset(self.data_path)
        return self._dataset

    def setup(self, job: JobResponse) -> None:
        """Set up the task processor for a new job.

        Args:
            job: Job response containing job information and configuration
        """
        log.info(f"Setting up job {self.job_name} (ID: {self.job_id})")

        # Additional setup could be added here, such as:
        # - Loading job-specific configurations
        # - Setting up logging/monitoring
        # - Initializing job-specific resources

    def shutdown(self) -> None:
        """Clean up resources when shutting down."""
        log.info(f"Shutting down job {self.job_name} (ID: {self.job_id})")
        # Add cleanup code here if needed

    def run_training(self, et_model, total_epochs: int = 1) -> Dict:
        """Run training loop.

        Args:
            et_model: ExecutorTorch model
            total_epochs: Number of epochs to train

        Returns:
            dict: Training results with parameter differences
        """
        log.info(f"Starting training for {total_epochs} epochs")
        initial_params = None
        # Dataset and DataLoader setup
        dataloader = DataLoader(
            self.get_dataset(),
            batch_size=self.training_config["batch_size"],
            shuffle=self.training_config["shuffle"],
            num_workers=self.training_config["num_workers"],
            drop_last=True,
        )
        total_batches = len(dataloader)

        for epoch in range(total_epochs):
            log.info(f"Epoch {epoch + 1}/{total_epochs}")

            for batch_idx, batch in enumerate(dataloader):
                x, y = batch
                loss, pred = et_model.forward_backward("forward", (x, y))

                if initial_params is None:
                    initial_params = clone_params(et_model.named_parameters())

                optimizer = get_sgd_optimizer(
                    et_model.named_parameters(),
                    self.training_config["learning_rate"],
                    self.training_config["momentum"],
                    self.training_config["weight_decay"],
                    self.training_config["dampening"],
                    self.training_config["nesterov"],
                )

                optimizer.step(et_model.named_gradients())

                # Log progress periodically
                if batch_idx % max(1, total_batches // 10) == 0:
                    log.info(f"Epoch {epoch + 1}/{total_epochs} - Batch {batch_idx + 1}/{total_batches} - Loss: {loss}")

        log.info("Training completed")
        last_params = clone_params(et_model.named_parameters())
        param_diff = calc_params_diff(initial_params, last_params)
        result = tensor_dict_to_json(param_diff)
        return result

    def process_task(self, task: TaskResponse) -> dict:
        """Process received task and return results.

        Args:
            task: The task response containing model and instructions

        Returns:
            dict: Results from training

        Raises:
            ValueError: If task data is invalid or protocol validation fails
            RuntimeError: If training operations fail
        """
        log.info(f"Processing task {task.task_name=}")

        if task.task_name != "train":
            log.error(f"Received unknown task: {task.task_name}")
            raise ValueError(f"Unsupported task type: {task.task_name}")

        payload: DXO = from_dict(task.task_data)

        # Validate inputs first - fail fast if invalid
        verify_payload(
            payload,
            expected_type=ModelBufferType.EXECUTORCH,
            expected_format=ModelNativeFormat.BINARY,
            expected_encoding=ModelEncoding.BASE64,
        )

        try:
            model_bytes = base64.b64decode(payload.data)
            et_model = _load_for_executorch_for_training_from_buffer(model_bytes)
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            raise RuntimeError("Failed to load model") from e

        try:
            diff_dict = self.run_training(et_model)
            log.info("Training completed successfully")
            dxo_dict = {
                "meta": payload.meta,
                "data": diff_dict,
                "kind": "et_tensor_diff",
            }
            return dxo_dict
        except Exception as e:
            log.error(f"Training failed with unexpected error: {e}")
            raise RuntimeError("Training failed unexpectedly") from e
