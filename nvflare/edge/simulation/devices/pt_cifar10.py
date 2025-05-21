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
from torchvision import datasets, transforms

from nvflare.edge.models.model import Cifar10ConvNet
from nvflare.edge.simulation.device_task_processor import DeviceTaskProcessor
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.task_response import TaskResponse

log = logging.getLogger(__name__)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class PTCifar10Processor(DeviceTaskProcessor):

    def __init__(self, data_root: str, subset_size: int):
        DeviceTaskProcessor.__init__(self)
        self.data_root = data_root
        self.subset_size = subset_size
        self.min_train_time = min_train_time
        self.max_train_time = max_train_time

    def setup(self, job: JobResponse) -> None:
        pass

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
        task_data = task.task_data
        assert isinstance(task_data, dict)
        model = from_dict(task_data)
        if not isinstance(model, DXO):
            self.logger.error(f"expect model to be DXO but got {type(model)}")
            raise ValueError("bad model data")

        # Convert list to numpy to tensor and run training
        global_model = {k: torch.tensor(v) for k, v in global_model.items()}
        diff_dict = self._pytorch_training(global_model, global_round)

        # Random delay
        delay = random.uniform(self.min_train_time, self.max_train_time)
        time.sleep(delay)

        # Compose simple returning message
        result_dxo = DXO(data_kind="model", data={"value": diff_dict})

        return result_dxo.to_dict()
