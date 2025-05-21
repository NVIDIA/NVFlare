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
import logging
import os

import filelock
import torch
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from nvflare.edge.constants import MsgKey
from nvflare.edge.model_protocol import (
    ModelBufferType,
    ModelEncoding,
    ModelExchangeFormat,
    ModelNativeFormat,
    verify_payload,
)
from nvflare.edge.models.model import Cifar10ConvNet
from nvflare.edge.simulation.device_task_processor import DeviceTaskProcessor
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.task_response import TaskResponse

log = logging.getLogger(__name__)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class PTCifar10TaskProcessor(DeviceTaskProcessor):
    def __init__(self, data_root: str, subset_size: int):
        DeviceTaskProcessor.__init__(self)
        self.data_root = data_root
        self.subset_size = subset_size

    def setup(self, job: JobResponse) -> None:
        device_io_dir = f"/tmp/nvflare/workspaces/edge_simulator_cifar10/{self.device_info.device_id}"
        os.makedirs(device_io_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(device_io_dir)

    def shutdown(self) -> None:
        pass

    def _pytorch_training(self, global_model, global_round):
        # Data loading code
        transform = transforms.Compose([transforms.ToTensor()])
        batch_size = 4

        # Add file lock to prevent multiple simultaneous downloads
        lock_file = os.path.join(self.data_root, "cifar10.lock")
        with filelock.FileLock(lock_file):
            train_set = datasets.CIFAR10(root=self.data_root, train=True, download=True, transform=transform)

        # Find the device ID numer
        device_id = int(self.device_info.device_id.split("-")[-1])
        indices = list(range(device_id * self.subset_size, (device_id + 1) * self.subset_size))
        train_subset = Subset(train_set, indices)
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)

        # Network loading
        net = Cifar10ConvNet()
        net.load_state_dict(global_model)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        net.to(DEVICE)

        # Training loop
        for epoch in range(4):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                # record loss every 250 mini-batches (1000 samples)
                if i % 250 == 249:
                    self.tb_writer.add_scalar(
                        "loss", running_loss / 250, (global_round * 4 + epoch) * len(train_loader) + i
                    )
                    running_loss = 0.0

        # Calculate the model param diff
        diff_dict = {}
        for key, param in net.state_dict().items():
            diff_dict[key] = param.cpu().numpy() - global_model[key].numpy()
            diff_dict[key] = diff_dict[key].tolist()
        return diff_dict

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
        log.info(f"Processing task {task.task_name}")

        if task.task_name != "train":
            log.error(f"Received unknown task: {task.task_name}")
            raise ValueError(f"Unsupported task type: {task.task_name}")

        # Validate inputs first - fail fast if invalid
        payload = verify_payload(
            task.task_data[MsgKey.PAYLOAD],
            expected_type=ModelBufferType.PYTORCH,
            expected_format=ModelNativeFormat.STRING,
            expected_encoding=ModelEncoding.NONE,
        )
        global_round = payload[ModelExchangeFormat.MODEL_VERSION]
        global_model = payload[ModelExchangeFormat.MODEL_BUFFER]

        # Convert list to numpy to tensor and run training
        global_model = {k: torch.tensor(v) for k, v in global_model.items()}
        diff_dict = self._pytorch_training(global_model, global_round)

        # Compose simple returning message
        return_msg = {MsgKey.WEIGHTS: diff_dict, MsgKey.MODE: "diff"}

        return return_msg
