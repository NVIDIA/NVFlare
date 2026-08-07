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

"""Server-side synchronous FedAvg workflow."""

import torch

from nvflare.collab import collab

from ..aggregation import aggregate_result
from ..data import make_data_loader
from ..model import SimpleNetwork, get_model_state
from .client import BATCH_SIZE


class FedAvgServer:
    def __init__(self, num_rounds: int):
        self.num_rounds = num_rounds

    @staticmethod
    def evaluate(model, test_loader) -> float:
        model.eval()
        device = next(model.parameters()).device
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                correct += model(inputs).argmax(dim=1).eq(labels).sum().item()
                total += labels.size(0)
        return correct / total

    # @collab.main marks the single server entry point that drives the workflow.
    @collab.main
    def run(self) -> dict[str, torch.Tensor]:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = SimpleNetwork().to(device)
        test_loader = make_data_loader(train=False, batch_size=BATCH_SIZE)
        global_weights = get_model_state(model)

        for round_number in range(self.num_rounds):
            # This calls every client's @collab.publish train method and collects results by site.
            client_results = collab.clients.train(global_weights)
            global_weights = aggregate_result(client_results, "weights", round_number)
            model.load_state_dict(global_weights)
            accuracy = self.evaluate(model, test_loader)
            print(f"Round {round_number + 1}: test accuracy={accuracy:.4f}")

        return global_weights
