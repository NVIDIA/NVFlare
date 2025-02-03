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
import torch

from nvflare.apis.dxo import from_shareable
from nvflare.app_opt.p2p.executors.gradient_tracking import GTExecutor


class GTADAMExecutor(GTExecutor):
    """An executor that implements GTAdam in a peer-to-peer (P2P) learning setup.

    Each client maintains its own local model and synchronously exchanges model parameters with its neighbors
    at each iteration. The model parameters are updated based on the neighbors' parameters and local gradient descent steps.
    The executor also tracks and records training, validation and test losses over time.

    The number of iterations, the learning rate and the beta1, beta2 and epsilon hyperparameters must be provided
    by the controller when asing to run the algorithm.  They can be set in the extra parameters of the controller's
    config with the "iterations", "stepsize", "beta1", "beta2", and "epsilon" keys.

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

    def _pre_algorithm_run(self, fl_ctx, shareable, abort_signal):
        super()._pre_algorithm_run(fl_ctx, shareable, abort_signal)

        data = from_shareable(shareable).data
        self.beta1 = data["beta1"]
        self.beta2 = data["beta2"]
        self.epsilon = data["epsilon"]
        self.G = torch.tensor(1e6, device=self.device)
        self.m = [torch.zeros_like(param, device=self.device) for param in self.model.parameters()]
        self.v = [torch.zeros_like(param, device=self.device) for param in self.model.parameters()]

    def _update_local_state(self, stepsize):
        for i in range(len(self.tracker)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * self.tracker[i]
            self.v[i] = torch.minimum(self.beta2 * self.v[i] + (1 - self.beta2) * self.tracker[i] ** 2, self.G)

        with torch.no_grad():
            for idx, param in enumerate(self.model.parameters()):
                if param.requires_grad:
                    descent = torch.divide(self.m[idx], torch.sqrt(self.v[idx] + self.epsilon))
                    param.add_(descent, alpha=-stepsize)
