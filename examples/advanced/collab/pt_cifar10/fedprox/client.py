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

"""FedProx client: FedAvg training plus a proximal loss."""

from ..fedavg.client import FedAvgClient

MU = 0.01


class FedProxClient(FedAvgClient):
    def _before_train(self) -> None:
        self.global_parameters = [parameter.detach().clone() for parameter in self.model.parameters()]

    def _compute_loss(self, inputs, labels, criterion):
        loss = super()._compute_loss(inputs, labels, criterion)
        proximal_term = sum(
            (local - global_parameter).square().sum()
            for local, global_parameter in zip(self.model.parameters(), self.global_parameters)
        )
        return loss + 0.5 * MU * proximal_term
