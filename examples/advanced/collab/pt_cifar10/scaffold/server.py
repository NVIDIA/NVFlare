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

"""Server-side synchronous SCAFFOLD workflow."""

import torch

from nvflare.collab import collab

from ..aggregation import aggregate_result
from ..data import make_data_loader
from ..fedavg.client import BATCH_SIZE
from ..fedavg.server import FedAvgServer
from ..model import SimpleNetwork, get_model_state


class ScaffoldServer(FedAvgServer):
    # @collab.main is the server entry point for the SCAFFOLD round loop.
    @collab.main
    def run(self) -> dict[str, torch.Tensor]:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = SimpleNetwork().to(device)
        test_loader = make_data_loader(train=False, batch_size=BATCH_SIZE)
        global_weights = get_model_state(model)
        global_controls = {
            name: torch.zeros_like(parameter, device="cpu") for name, parameter in model.named_parameters()
        }

        for round_number in range(self.num_rounds):
            # The call maps to ScaffoldClient.train and sends both model and control state.
            client_results = collab.clients.train(global_weights, global_controls)
            global_weights = aggregate_result(client_results, "weights", round_number)
            control_delta = aggregate_result(client_results, "control_delta", round_number)
            for name, delta in control_delta.items():
                global_controls[name].add_(delta)

            model.load_state_dict(global_weights)
            accuracy = self.evaluate(model, test_loader)
            print(f"Round {round_number + 1}: test accuracy={accuracy:.4f}")

        return global_weights
