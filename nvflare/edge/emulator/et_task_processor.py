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
from typing import Dict, Tuple

import torch
from executorch.extension.training import (
    _load_for_executorch_for_training_from_buffer,
    get_sgd_optimizer,
)
from torch.utils.data import DataLoader, Dataset

from nvflare.edge.constants import MsgKey
from nvflare.edge.emulator.device_task_processor import DeviceTaskProcessor
from nvflare.edge.model_protocol import (
    ModelBufferType,
    ModelEncoding,
    ModelExchangeFormat,
    ModelNativeFormat,
    verify_payload,
)
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.task_response import TaskResponse
from nvflare.edge.web.models.user_info import UserInfo

log = logging.getLogger(__name__)


def tensor_dict_to_json(d):
    j = {}
    for k, v in d.items():
        entry = {}
        # TODO: this needs to be compatible with the "Controller" side logic
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
        device_info: DeviceInfo,
        user_info: UserInfo,
        data_path: str,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
    ):
        super().__init__(device_info, user_info)
        self.job_id = None
        self.job_name = None
        self.device_info = device_info
        self.data_path = data_path

        # Dataset and DataLoader setup
        self.dataset = self.get_dataset(data_path)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.data_iterator = iter(self.dataloader)

    @abstractmethod
    def get_dataset(self, data_path: str) -> Dataset:
        """Get dataset for training.

        Args:
            data_path: Path to dataset

        Returns:
            Dataset: PyTorch dataset for training
        """
        pass

    def read_data_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read a batch of data using DataLoader.

        Returns:
            tuple: (inputs, targets) batch for training
        """
        try:
            batch = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.dataloader)
            batch = next(self.data_iterator)
        return batch

    def setup(self, job: JobResponse) -> None:
        self.job_id = job.job_id
        self.job_name = job.job_name

    def shutdown(self) -> None:
        pass

    def run_training(self, et_model, total_epochs: int = 1) -> Dict:
        """Run training loop.

        Args:
            et_model: ExecutorTorch model
            total_epochs: Number of epochs to train

        Returns:
            dict: Training results with parameter differences
        """
        initial_params = None
        for i in range(total_epochs):
            data = self.read_data_batch()
            loss, pred = et_model.forward_backward("forward", data)

            if initial_params is None:
                initial_params = clone_params(et_model.named_parameters())

            optimizer = get_sgd_optimizer(
                et_model.named_parameters(),
                0.1,
                0,
                0,
                0,
                False,
            )

            optimizer.step(et_model.named_gradients())

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

        # Validate inputs first - fail fast if invalid
        payload = verify_payload(
            task.task_data[MsgKey.PAYLOAD],
            expected_type=ModelBufferType.EXECUTORCH,
            expected_format=ModelNativeFormat.BINARY,
            expected_encoding=ModelEncoding.BASE64,
        )

        try:
            model_bytes = base64.b64decode(payload[ModelExchangeFormat.MODEL_BUFFER])
            et_model = _load_for_executorch_for_training_from_buffer(model_bytes)
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            raise RuntimeError("Failed to load model") from e

        try:
            diff_dict = self.run_training(et_model)
            log.info("Training completed successfully")
            return {"result": diff_dict}
        except Exception as e:
            log.error(f"Training failed with unexpected error: {e}")
            raise RuntimeError("Training failed unexpectedly") from e
