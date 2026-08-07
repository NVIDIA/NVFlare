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

"""SCAFFOLD client with a persistent local control variate."""

import torch

from nvflare.collab import collab

from ..fedavg.client import FedAvgClient
from ..model import get_model_state


class ScaffoldClient(FedAvgClient):
    def __init__(self):
        super().__init__()
        self.local_controls = None
        self.control_correction = None
        self.lr_exposure = 0.0

    def _after_optimizer_step(self) -> None:
        learning_rate = self.optimizer.param_groups[0]["lr"]
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                parameter.add_(self.control_correction[name], alpha=-learning_rate)
        self.lr_exposure += learning_rate

    def _update_local_controls(self, global_weights, global_controls):
        control_delta = {}
        for name, parameter in self.model.named_parameters():
            old_local = self.local_controls[name]
            new_local = (
                old_local
                - global_controls[name].to(old_local)
                + (global_weights[name].to(old_local) - parameter.detach().cpu()) / self.lr_exposure
            )
            self.local_controls[name] = new_local
            control_delta[name] = new_local - old_local
        return control_delta

    # SCAFFOLD publishes the same train operation with one extra control argument.
    @collab.publish
    def train(self, global_weights: dict[str, torch.Tensor], global_controls: dict[str, torch.Tensor]) -> dict:
        self.model.load_state_dict(global_weights)
        parameters = dict(self.model.named_parameters())
        if self.local_controls is None:
            self.local_controls = {
                name: torch.zeros_like(parameter, device="cpu") for name, parameter in parameters.items()
            }

        self.control_correction = {
            name: global_controls[name].to(parameter) - self.local_controls[name].to(parameter)
            for name, parameter in parameters.items()
        }
        self.lr_exposure = 0.0
        num_steps = self._local_train()
        control_delta = self._update_local_controls(global_weights, global_controls)
        print(f"[{collab.site_name}] completed {num_steps} local steps")
        return {
            "weights": get_model_state(self.model),
            "control_delta": control_delta,
            "num_steps": num_steps,
        }
