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
import hashlib
import logging
import random
import time
import uuid

import torch
from executorch.extension.training import _load_for_executorch_for_training_from_buffer, get_sgd_optimizer
from torch.utils.data import Subset
from torchvision import datasets, transforms

from nvflare.apis.dxo import DXO, DataKind
from nvflare.edge.constants import MsgKey
from nvflare.edge.model_protocol import (
    ModelBufferType,
    ModelEncoding,
    ModelExchangeFormat,
    ModelNativeFormat,
    verify_payload,
)
from nvflare.edge.simulation.device_task_processor import DeviceTaskProcessor
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.task_response import TaskResponse

log = logging.getLogger(__name__)


def tensor_dict_to_json(d):
    j = {}
    for k, v in d.items():
        # TODO: this needs to be compatible with the "Controller" side logic
        # entry = {}
        # entry["data"] = v.cpu().numpy().tolist()
        # entry["sizes"] = list(v.size())
        # j[k] = entry
        j[k] = v.cpu().numpy().tolist()
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


class ETCifar10Processor(DeviceTaskProcessor):

    def __init__(self, data_root: str, subset_size: int, min_train_time: float, max_train_time: float):
        DeviceTaskProcessor.__init__(self)
        self.data_root = data_root
        self.subset_size = subset_size
        self.min_train_time = min_train_time
        self.max_train_time = max_train_time
        self.train_set = None

    def setup(self, job: JobResponse) -> None:
        transform = transforms.Compose([transforms.ToTensor()])
        self.train_set = datasets.CIFAR10(root=self.data_root, train=True, download=True, transform=transform)

    def shutdown(self) -> None:
        pass

    def _uuid_to_seed(self, uuid_string):
        """Converts a UUID-4 string to a seed number (integer)."""
        try:
            uuid_obj = uuid.UUID(uuid_string)
        except ValueError:
            raise ValueError("Invalid UUID string.")
        # Get the bytes representation of the UUID
        uuid_bytes = uuid_obj.bytes
        # Hash the bytes using SHA-256 for better distribution
        hashed_bytes = hashlib.sha256(uuid_bytes).digest()
        # Convert the first 8 bytes of the hash to an integer (64-bit)
        seed_number = int.from_bytes(hashed_bytes[:8], byteorder="big")

        return seed_number

    def _executorch_training(self, et_model):
        batch_size = 4

        # Randomly select a subset of the training set
        device_uuid = self.device_info.device_id.split("_")[-1]  # device-prefix_[uuid]
        seed = self._uuid_to_seed(device_uuid)
        random.seed(seed)
        # generate a random indices list
        indices = list(range(len(self.train_set)))
        random.shuffle(indices)
        # select the first subset_size indices
        indices = indices[: self.subset_size]
        # create a new train_set from the selected indices
        train_subset = Subset(self.train_set, indices)
        # create a dataloader for the train_subset
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)

        initial_params = None

        # Training loop
        for epoch in range(4):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0], data[1]
                loss, pred = et_model.forward_backward("forward", (inputs, labels))

                if initial_params is None:
                    initial_params = clone_params(et_model.named_parameters())
                    log.info(f"Initial params: {initial_params.keys()}")

                optimizer = get_sgd_optimizer(
                    et_model.named_parameters(),
                    0.1,
                    0.0,
                    0.0,
                    0.0,
                    False,
                )

                optimizer.step(et_model.named_gradients())

        # Calculate the model param diff
        log.info("Training completed")
        last_params = clone_params(et_model.named_parameters())
        log.info(f"Last params: {last_params.keys()}")
        param_diff = calc_params_diff(initial_params, last_params)
        result = tensor_dict_to_json(param_diff)
        return result

    def process_task(self, task: TaskResponse) -> dict:
        log.info(f"Processing task {task.task_name=}")

        task_data = task.task_data

        if task.task_name != "train":
            log.error(f"Received unknown task: {task.task_name}")
            raise ValueError(f"Unsupported task type: {task.task_name}")

        # Validate inputs first - fail fast if invalid
        model_buffer = task_data["data"]

        try:
            model_bytes = base64.b64decode(model_buffer)
            et_model = _load_for_executorch_for_training_from_buffer(model_bytes)
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            raise RuntimeError("Failed to load model") from e

        try:
            diff_dict = self._executorch_training(et_model)
            log.info("Training completed successfully")

        except Exception as e:
            log.error(f"Training failed with unexpected error: {e}")
            raise RuntimeError("Training failed unexpectedly") from e

        return self._compose_result(diff_dict)

    def _compose_result(self, diff_dict):
        # TODO: make this dict a constant, this needs to be compatible with the
        #     "model_update_dxo" in the aggregator...
        result_dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data={"dict": diff_dict})

        # Random delay
        delay = random.uniform(self.min_train_time, self.max_train_time)
        time.sleep(delay)

        return result_dxo.to_dict()
