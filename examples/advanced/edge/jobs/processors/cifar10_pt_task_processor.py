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
import random
import time

import filelock
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

from nvflare.apis.dxo import DXO, DataKind, from_dict
from nvflare.edge.simulation.device_task_processor import DeviceTaskProcessor
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.task_response import TaskResponse

from .models.cifar10_model import Cifar10ConvNet

log = logging.getLogger(__name__)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class Cifar10PTTaskProcessor(DeviceTaskProcessor):

    def __init__(
        self,
        data_root: str,
        subset_size: int,
        communication_delay: dict,
        device_speed: dict,
        local_batch_size: int = 4,
        local_epochs: int = 4,
        local_lr: float = 0.001,
        local_momentum: float = 0.9,
    ):
        DeviceTaskProcessor.__init__(self)
        self.data_root = data_root
        self.subset_size = subset_size
        self.communication_delay = communication_delay
        self.device_speed = device_speed
        # device_speed_type is the index of the device speed in the list of device speeds
        # this is used to simulate different devices having different training speeds
        # it is set to None initially and will be randomly assigned when the device is initialized
        mean_speed = self.device_speed.get("mean")
        self.device_speed_type = random.randint(0, len(mean_speed) - 1)
        self.local_batch_size = local_batch_size
        self.local_epochs = local_epochs
        self.local_lr = local_lr
        self.local_momentum = local_momentum

    def setup(self, job: JobResponse) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def _pytorch_training(self, global_model):
        # Data loading code
        transform = transforms.Compose([transforms.ToTensor()])
        batch_size = self.local_batch_size

        # Add file lock to prevent multiple simultaneous downloads
        lock_file = os.path.join(self.data_root, "cifar10.lock")
        with filelock.FileLock(lock_file):
            train_set = datasets.CIFAR10(root=self.data_root, train=True, download=True, transform=transform)

        # Generate seed according to device_id
        # Randomly select a subset of the training set
        # generate a random indices list
        indices = list(range(len(train_set)))
        random.shuffle(indices)
        # select the first subset_size indices
        indices = indices[: self.subset_size]
        # create a new train_set from the selected indices
        train_subset = Subset(train_set, indices)
        # create a dataloader for the train_subset
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)

        # Network loading
        net = Cifar10ConvNet()
        net.load_state_dict(global_model)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.local_lr, momentum=self.local_momentum)
        net.to(DEVICE)

        # Training loop
        # Let's do 4 local epochs
        for epoch in range(self.local_epochs):
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

        # Calculate the model param diff
        diff_dict = {}
        for key, param in net.state_dict().items():
            numpy_param = param.cpu().numpy() - global_model[key].numpy()
            # Convert numpy array to list for serialization
            diff_dict[key] = numpy_param.tolist()
        return diff_dict

    def process_task(self, task: TaskResponse) -> dict:
        task_data = task.task_data
        assert isinstance(task_data, dict)
        model = from_dict(task_data)
        if not isinstance(model, DXO):
            self.logger.error(f"expect model to be DXO but got {type(model)}")
            raise ValueError("bad model data")

        if model.data_kind != DataKind.WEIGHTS:
            self.logger.error(f"expect model data kind to be {DataKind.WEIGHTS} but got {model.data_kind}")
            raise ValueError("bad model data kind")

        global_model = model.data
        if not isinstance(global_model, dict):
            self.logger.error(f"expect global_model to be dict but got {type(global_model)}")
            raise ValueError("bad global model")

        # do the training
        # convert the global_model to a dict of tensors
        global_model = {k: torch.tensor(v) for k, v in global_model.items()}
        diff_dict = self._pytorch_training(global_model)

        # create the result DXO
        result_dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data={"dict": diff_dict})

        # random delay to simulate the training time difference
        # get mean and std from device_speed
        mean_speed = self.device_speed.get("mean")
        std_speed = self.device_speed.get("std")
        # mean and std should be of same length
        if len(mean_speed) != len(std_speed):
            self.logger.error("mean and std speed should be of same length")
            raise ValueError("bad device speed data")
        # sample the value from the normal distribution
        delay_speed = random.gauss(mean_speed[self.device_speed_type], std_speed[self.device_speed_type])
        if delay_speed < 0:
            delay_speed = 0

        # get mean and std from communication_delay
        mean_comm = self.communication_delay.get("mean")
        std_comm = self.communication_delay.get("std")
        # sample the communication delay from the normal distribution
        delay_comm = random.gauss(mean_comm, std_comm)
        if delay_comm < 0:
            delay_comm = 0

        delay = delay_speed + delay_comm

        # Log the delay with the device id
        device_id = self.device.device_id
        self.logger.info(f"Device {device_id} training delay: {delay:.2f} seconds")

        time.sleep(delay)
        return result_dxo.to_dict()
