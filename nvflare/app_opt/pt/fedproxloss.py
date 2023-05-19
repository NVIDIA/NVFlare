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

import torch
from torch.nn.modules.loss import _Loss


class PTFedProxLoss(_Loss):
    def __init__(self, mu: float = 0.01) -> None:
        """Compute FedProx loss: a loss penalizing the deviation from global model.

        Args:
            mu: weighting parameter
        """
        super().__init__()
        if mu < 0.0:
            raise ValueError("mu should be no less than 0.0")
        self.mu = mu

    def forward(self, input, target) -> torch.Tensor:
        """Forward pass in training.

        Args:
            input (nn.Module): the local pytorch model
            target (nn.Module): the copy of global pytorch model when local clients received it
                                at the beginning of each local round

        Returns:
            FedProx loss term
        """
        prox_loss: torch.Tensor = 0.0
        for param, ref in zip(input.named_parameters(), target.named_parameters()):
            prox_loss += (self.mu / 2) * torch.sum((param[1] - ref[1]) ** 2)

        return prox_loss
