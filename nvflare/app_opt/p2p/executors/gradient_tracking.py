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
import time
from abc import abstractmethod

import torch

from nvflare.apis.dxo import from_shareable
from nvflare.app_opt.p2p.executors.sync_executor import SyncAlgorithmExecutor
from nvflare.app_opt.p2p.utils.metrics import compute_loss_over_dataset
from nvflare.app_opt.p2p.utils.utils import get_device


class GTExecutor(SyncAlgorithmExecutor):
    """An executor that implements Stochastic Gradient Tracking (GT) in a peer-to-peer (P2P) learning setup.

    Each client maintains its own local model and synchronously exchanges model parameters with its neighbors
    at each iteration. The model parameters are updated based on the neighbors' parameters and local gradient descent steps.
    The executor also tracks and records training, validation and test losses over time.

    The number of iterations and the learning rate must be provided by the controller when asing to run the algorithm.
    They can be set in the extra parameters of the controller's config with the "iterations" and "stepsize" keys.

    Note:
        Subclasses must implement the __init__ method to initialize the model, loss function, and data loaders.

    Args:
        model (torch.nn.Module, optional): The neural network model used for training.
        loss (torch.nn.modules.loss._Loss, optional): The loss function used for training.
        train_dataloader (torch.utils.data.DataLoader, optional): DataLoader for the training dataset.
        test_dataloader (torch.utils.data.DataLoader, optional): DataLoader for the testing dataset.
        val_dataloader (torch.utils.data.DataLoader, optional): DataLoader for the validation dataset.

    Attributes:
        model (torch.nn.Module): The neural network model.
        loss (torch.nn.modules.loss._Loss): The loss function.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for testing data.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
        train_loss_sequence (list[tuple]): Records of training loss over time.
        test_loss_sequence (list[tuple]): Records of testing loss over time.
    """

    @abstractmethod
    def __init__(
        self,
        model: torch.nn.Module | None = None,
        loss: torch.nn.modules.loss._Loss | None = None,
        train_dataloader: torch.utils.data.DataLoader | None = None,
        test_dataloader: torch.utils.data.DataLoader | None = None,
        val_dataloader: torch.utils.data.DataLoader | None = None,
    ):
        super().__init__()
        self.device = get_device()
        self.model = model.to(self.device)
        self.loss = loss.to(self.device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.val_dataloader = val_dataloader

        # metrics
        self.train_loss_sequence = []
        self.test_loss_sequence = []

    def run_algorithm(self, fl_ctx, shareable, abort_signal):
        start_time = time.time()
        iter_dataloader = iter(self.train_dataloader)

        for iteration in range(self._iterations):
            self.log_info(fl_ctx, f"iteration: {iteration}/{self._iterations}")
            if abort_signal.triggered:
                break

            try:
                data, label = next(iter_dataloader)
                data, label = data.to(self.device), label.to(self.device)
            except StopIteration:
                # 3. store metrics
                current_time = time.time() - start_time
                self.train_loss_sequence.append(
                    (
                        current_time,
                        compute_loss_over_dataset(self.model, self.loss, self.train_dataloader, self.device),
                    )
                )
                self.test_loss_sequence.append(
                    (
                        current_time,
                        compute_loss_over_dataset(self.model, self.loss, self.test_dataloader, self.device),
                    )
                )
                # restart after an epoch
                iter_dataloader = iter(self.train_dataloader)
                data, label = next(iter_dataloader)
                data, label = data.to(self.device), label.to(self.device)

            # run algorithm step
            with torch.no_grad():
                # 1. exchange trainable parameters and tracker
                value_to_exchange = {
                    "parameters": self.model.parameters(),
                    "tracker": self.tracker,
                }
                self._exchange_values(fl_ctx, value=value_to_exchange, iteration=iteration)

                # 2. Update trainable parameters
                # - a. compute consensus value
                for idx, param in enumerate(self.model.parameters()):
                    if param.requires_grad:
                        param.mul_(self._weight)
                        for neighbor in self.neighbors:
                            neighbor_param = self.neighbors_values[iteration][neighbor.id]["parameters"][idx].to(
                                self.device
                            )
                            param.add_(
                                neighbor_param,
                                alpha=neighbor.weight,
                            )

                # - b. update local parameters
                self._update_local_state(self._stepsize)

                # 3. Update tracker
                # - a. consensus on tracker
                for idx, tracker in enumerate(iter(self.tracker)):
                    tracker.mul_(self._weight)
                    for neighbor in self.neighbors:
                        neighbor_tracker = self.neighbors_values[iteration][neighbor.id]["tracker"][idx].to(self.device)
                        tracker.add_(
                            neighbor_tracker,
                            alpha=neighbor.weight,
                        )

            # -b. compute new gradients
            self.model.zero_grad()
            pred = self.model(data)
            loss = self.loss(pred, label)
            loss.backward()

            gradient = [param.grad.clone() for param in self.model.parameters()]

            # - c. update tracker
            with torch.no_grad():
                for i in range(len(self.tracker)):
                    self.tracker[i].add_(gradient[i], alpha=1.0)
                    self.tracker[i].sub_(self.old_gradient[i], alpha=1.0)

            self.old_gradient = [g.clone() for g in gradient]

            # 4. free memory that's no longer needed
            del self.neighbors_values[iteration]

    def _update_local_state(self, stepsize):
        for idx, param in enumerate(self.model.parameters()):
            if param.requires_grad:
                param.add_(self.tracker[idx], alpha=-stepsize)

    def _to_message(self, x):
        return {
            "parameters": [param.cpu().numpy() for param in iter(x["parameters"])],
            "tracker": [z.cpu().numpy() for z in iter(x["tracker"])],
        }

    def _from_message(self, x):
        return {
            "parameters": [torch.from_numpy(param) for param in x["parameters"]],
            "tracker": [torch.from_numpy(z) for z in x["tracker"]],
        }

    def _pre_algorithm_run(self, fl_ctx, shareable, abort_signal):
        data = from_shareable(shareable).data
        self._iterations = data["iterations"]
        self._stepsize = data["stepsize"]

        init_train_loss = compute_loss_over_dataset(self.model, self.loss, self.train_dataloader, self.device)
        init_test_loss = compute_loss_over_dataset(self.model, self.loss, self.test_dataloader, self.device)

        self.train_loss_sequence.append((0, init_train_loss))
        self.test_loss_sequence.append((0, init_test_loss))

        # initialize tracker
        self.old_gradient = [torch.zeros_like(param, device=self.device) for param in self.model.parameters()]
        self.tracker = [torch.zeros_like(param, device=self.device) for param in self.model.parameters()]

    def _post_algorithm_run(self, *args, **kwargs):
        torch.save(torch.tensor(self.train_loss_sequence), "train_loss_sequence.pt")
        torch.save(torch.tensor(self.test_loss_sequence), "test_loss_sequence.pt")
