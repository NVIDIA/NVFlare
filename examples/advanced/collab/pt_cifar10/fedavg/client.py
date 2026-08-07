# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Client-side CIFAR-10 training shared by FedAvg and FedProx."""

import torch
import torch.nn as nn

from nvflare.collab import collab

from ..data import make_data_loader
from ..model import SimpleNetwork, get_model_state

BATCH_SIZE = 64
LEARNING_RATE = 0.01
LOCAL_EPOCHS = 1


class FedAvgClient:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.train_loader = None

    # @collab.init runs once on each client before published methods are called.
    @collab.init
    def initialize(self):
        self.model = SimpleNetwork().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE, momentum=0.9)
        self.train_loader = make_data_loader(train=True, batch_size=BATCH_SIZE)

    def _before_train(self) -> None:
        pass

    def _compute_loss(self, inputs, labels, criterion) -> torch.Tensor:
        return criterion(self.model(inputs), labels)

    def _after_optimizer_step(self) -> None:
        pass

    def _local_train(self) -> int:
        criterion = nn.CrossEntropyLoss()
        self._before_train()
        self.model.train()
        num_steps = 0

        for _epoch in range(LOCAL_EPOCHS):
            for inputs, labels in self.train_loader:
                if collab.is_aborted:
                    raise RuntimeError("Training aborted")
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                loss = self._compute_loss(inputs, labels, criterion)
                loss.backward()
                self.optimizer.step()
                self._after_optimizer_step()
                num_steps += 1
        return num_steps

    # @collab.publish exposes train so the server can call it through collab.clients.
    @collab.publish
    def train(self, global_weights: dict[str, torch.Tensor]) -> dict:
        self.model.load_state_dict(global_weights)
        num_steps = self._local_train()
        print(f"[{collab.site_name}] completed {num_steps} local steps")
        return {"weights": get_model_state(self.model), "num_steps": num_steps}
