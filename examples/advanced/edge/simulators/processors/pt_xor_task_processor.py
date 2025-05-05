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

import torch
from torch.utils.tensorboard import SummaryWriter

from nvflare.edge.constants import MsgKey
from nvflare.edge.model_protocol import (
    ModelBufferType,
    ModelEncoding,
    ModelExchangeFormat,
    ModelNativeFormat,
    verify_payload,
)
from nvflare.edge.models.model import XorNet
from nvflare.edge.simulation.device_task_processor import DeviceTaskProcessor
from nvflare.edge.simulation.simulated_device import SimulatedDevice
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.task_response import TaskResponse

log = logging.getLogger(__name__)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class PTXorTaskProcessor(DeviceTaskProcessor):
    def __init__(self):
        DeviceTaskProcessor.__init__(self)

    def setup(self, job: JobResponse) -> None:
        assert isinstance(self.device, SimulatedDevice)
        device_io_dir = f"/tmp/nvflare/workspaces/edge_simulator_xor/{self.device_info.device_id}"
        os.makedirs(device_io_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(device_io_dir)

    def shutdown(self) -> None:
        pass

    def _pytorch_training(self, global_model, global_round):
        # Define the XOR dataset
        xor_data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        xor_target = torch.tensor([0, 1, 1, 0], dtype=torch.long)

        # Network loading
        net = XorNet()
        net.load_state_dict(global_model)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        net.to(DEVICE)

        # Training loop for XOR
        local_epoch = 1000
        for epoch in range(local_epoch):
            inputs = xor_data.to(DEVICE)
            labels = xor_target.to(DEVICE)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # record loss
            self.tb_writer.add_scalar("loss", loss.item(), global_round * local_epoch + epoch)

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
