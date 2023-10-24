# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Union

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from net import Net
from torchvision import transforms

from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.abstract.model_learner import ModelLearner
from nvflare.app_common.app_constant import ModelName


class CIFAR10ModelLearner(ModelLearner):
    def __init__(
        self,
        epochs: int = 2,
        lr: float = 1e-2,
        momentum: float = 0.9,
        batch_size: int = 4,
        num_workers: int = 1,
        dataset_path: str = "/tmp/nvflare/data/cifar10",
        model_path: str = "/tmp/nvflare/data/cifar10/cifar_net.pth",
        device: str = "cuda:0",
    ):
        """CIFAR-10 Trainer.

        Args:
            epochs: the number of training epochs for a round. Defaults to 1.
            lr: local learning rate. Float number. Defaults to 1e-2.
            momentum (float, optional): Momentum. Defaults to 0.9
            batch_size: batch size for training and validation.
            num_workers: number of workers for data loaders.
            dataset_path: path to dataset
            model_path: path to save model
            device: (optional) We change to use GPU to speed things up. if you want to use CPU, change DEVICE="cpu"

        Returns:
            an FLModel with the updated local model differences after running `train()`, the metrics after `validate()`,
            or the best local model depending on the specified task.
        """
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_path = dataset_path
        self.model_path = model_path

        self.train_dataset = None
        self.train_loader = None
        self.valid_dataset = None
        self.valid_loader = None

        self.net = None
        self.optimizer = None
        self.criterion = None
        self.device = device
        self.best_acc = 0.0

    def initialize(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.trainset = torchvision.datasets.CIFAR10(
            root=self.dataset_path, train=True, download=True, transform=transform
        )
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

        self.testset = torchvision.datasets.CIFAR10(
            root=self.dataset_path, train=False, download=True, transform=transform
        )
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

        self.net = Net()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum)

    def get_model(self, model_name: str) -> Union[str, FLModel]:
        # Retrieve the best local model saved during training.
        if model_name == ModelName.BEST_MODEL:
            try:
                model_data = torch.load(self.model_path, map_location="cpu")
                np_model_data = {k: v.cpu().numpy() for k, v in model_data.items()}

                return FLModel(params_type=ParamsType.FULL, params=np_model_data)
            except Exception as e:
                raise ValueError("Unable to load best model") from e
        else:
            raise ValueError(f"Unknown model_type: {model_name}")  # Raised errors are caught in LearnerExecutor class.

    def train(self, model: FLModel) -> Union[str, FLModel]:
        self.info(f"Current/Total Round: {self.current_round + 1}/{self.total_rounds}")
        self.info(f"Client identity: {self.site_name}")

        pt_input_params = {k: torch.as_tensor(v) for k, v in model.params.items()}
        self._local_train(pt_input_params)

        pt_output_params = {k: torch.as_tensor(v) for k, v in self.net.cpu().state_dict().items()}
        accuracy = self._local_validate(pt_output_params)

        if accuracy > self.best_acc:
            self.best_acc = accuracy
            torch.save(self.net.state_dict(), self.model_path)

        np_output_params = {k: v.cpu().numpy() for k, v in self.net.cpu().state_dict().items()}
        return FLModel(
            params=np_output_params,
            metrics={"accuracy": accuracy},
            meta={"NUM_STEPS_CURRENT_ROUND": 2 * len(self.trainloader)},
        )

    def _local_train(self, input_weights):
        self.net.load_state_dict(input_weights)
        # (optional) use GPU to speed things up
        self.net.to(self.device)

        for epoch in range(self.epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                # (optional) use GPU to speed things up
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    self.info(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                    running_loss = 0.0

        self.info("Finished Training")

    def validate(self, model: FLModel) -> Union[str, FLModel]:
        pt_params = {k: torch.as_tensor(v) for k, v in model.params.items()}
        val_accuracy = self._local_validate(pt_params)

        return FLModel(metrics={"val_accuracy": val_accuracy})

    def _local_validate(self, input_weights):
        self.net.load_state_dict(input_weights)
        # (optional) use GPU to speed things up
        self.net.to(self.device)

        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.testloader:
                # (optional) use GPU to speed things up
                images, labels = data[0].to(self.device), data[1].to(self.device)
                # calculate outputs by running images through the network
                outputs = self.net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct // total
        self.info(f"Accuracy of the network on the 10000 test images: {val_accuracy} %")
        return val_accuracy
