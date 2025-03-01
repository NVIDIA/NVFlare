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

import torch
from nvflare.edge.emulator.device_task_processor import DeviceTaskProcessor
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.task_response import TaskResponse
from nvflare.edge.web.models.user_info import UserInfo

log = logging.getLogger(__name__)

import torch.nn as nn
from torch.nn import functional as F

import numpy as np
default_model_updates = {
    "net.linear2.weight":
        {"sizes":[2,10],
         "strides":[10,1],
         "data":np.ones(20).tolist()},
    "net.linear.bias":
        {"sizes":[10],
         "strides":[1],
         "data":np.ones(10).tolist()},
    "net.linear2.bias":
        {"sizes":[2],
         "strides":[1],
         "data":np.ones(2).tolist()},
    "net.linear.weight":
        {"sizes":[10,2],
         "strides":[2,1],
         "data":np.ones(20).tolist()}
}

# Net training logic on device
# Basic Net for XOR
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 10)
        self.linear2 = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear2(F.sigmoid(self.linear(x)))

# XOR training data
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
Y = torch.tensor([0, 1, 1, 0], dtype=torch.long)


class XorTaskProcessor(DeviceTaskProcessor):
    def __init__(self, device_info: DeviceInfo, user_info: UserInfo):
        super().__init__(device_info, user_info)
        self.job_id = None
        self.job_name = None

    def setup(self, job: JobResponse) -> None:
        self.job_id = job.job_id
        self.job_name = job.job_name
        # Setup training items
        self.model = Net()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

    def shutdown(self) -> None:
        pass

    def process_task(self, task: TaskResponse) -> dict:
        log.info(f"Processing task {task.task_name}")

        # Load global model
        # weights here is a buffer, to be loaded by ExecuTorch
        global_weights = task.task_data["weights"]

        # In this simulation, we skip the load process
        #for key in global_weights:
        #    global_weights[key] = torch.tensor(global_weights[key])
        #self.model.load_state_dict(global_weights)

        # Validate global model
        pred = self.model(X)
        # Calculate accuracy
        correct = (pred.argmax(dim=1) == Y).sum().item()
        accuracy = correct / len(Y)

        # Local training or validation
        result = None
        if task.task_name == "train":
            for i in range(1):
                self.optimizer.zero_grad()
                pred = self.model(X)
                loss = self.loss_fn(pred, Y)
                loss.backward()
                self.optimizer.step()

            # Ignor the actual training at this time,
            # return the default model updates
            result = {
                "weights": default_model_updates,
                "accuracy": accuracy
            }
        elif task.task_name == "validate":
            # Validate the model on the data
            pred = self.model(X)
            # Calculate accuracy
            correct = (pred.argmax(dim=1) == Y).sum().item()
            accuracy = correct / len(Y)
            result = {
                "accuracy": accuracy
            }
        else:
            log.error(f"Received unknown task: {task.task_name}")

        return result
