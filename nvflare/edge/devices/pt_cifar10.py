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
import hashlib
import os
import random
import time
import uuid

import filelock
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

from nvflare.apis.dxo import DXO, DataKind, from_dict
from nvflare.edge.models.model import Cifar10ConvNet
from nvflare.edge.simulated_device import DeviceFactory, SimulatedDevice
from nvflare.edge.web.models.task_response import TaskResponse

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class PTCifar10Device(SimulatedDevice):

    def __init__(self, device_id, data_root, subset_size, min_train_time=1.0, max_train_time=5.0):
        SimulatedDevice.__init__(self, device_id)
        self.min_train_time = min_train_time
        self.max_train_time = max_train_time
        self.data_root = data_root
        self.subset_size = subset_size
        self.random_seed = self._uuid_to_seed(device_id)

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

    def _pytorch_training(self, global_model):
        # Data loading code
        transform = transforms.Compose([transforms.ToTensor()])
        batch_size = 4

        # Add file lock to prevent multiple simultaneous downloads
        lock_file = os.path.join(self.data_root, "cifar10.lock")
        with filelock.FileLock(lock_file):
            train_set = datasets.CIFAR10(root=self.data_root, train=True, download=True, transform=transform)

        # Randomly select a subset of the training set
        # using the random seed
        random.seed(self.random_seed)
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
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        net.to(DEVICE)

        # Training loop
        # Let's do 4 local epochs
        for epoch in range(4):
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

                # print the loss every ${subset_size / batch_size}/10 iterations
                # if i % (self.subset_size / batch_size / 10) == 0:
                #    print(f"Epoch {epoch}, Step {i}, Loss: {loss.item()}")

        # Calculate the model param diff
        diff_dict = {}
        for key, param in net.state_dict().items():
            diff_dict[key] = param.cpu().numpy() - global_model[key].numpy()
        return diff_dict

    def do_task(self, data: TaskResponse) -> dict:
        task_data = data.task_data
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
        delay = random.uniform(self.min_train_time, self.max_train_time)
        time.sleep(delay)
        return result_dxo.to_dict()


class PTCifar10DeviceFactory(DeviceFactory):

    def __init__(self, data_root, subset_size, min_train_time=1.0, max_train_time=5.0):
        DeviceFactory.__init__(self)
        self.min_train_time = min_train_time
        self.max_train_time = max_train_time
        self.data_root = data_root
        self.subset_size = subset_size

    def make_device(self) -> SimulatedDevice:
        return PTCifar10Device(
            str(uuid.uuid4()), self.data_root, self.subset_size, self.min_train_time, self.max_train_time
        )
