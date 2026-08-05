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

"""Synchronous CIFAR-10 FedProx as a small extension of Collab FedAvg."""

import math
from pathlib import Path

import torch
from fedavg import FedAvgClient
from fedavg import define_parser as define_fedavg_parser
from fedavg import make_recipe as make_fedavg_recipe
from fedavg import run_example

JOB_NAME = "collab_pt_sync_cifar10_fedprox"
EXAMPLE_DIR = Path(__file__).resolve().parent


class FedProxClient(FedAvgClient):
    def __init__(self, *args, mu: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = mu
        self._global_parameters = None

    def _prepare_local_training(self) -> None:
        self._global_parameters = [parameter.detach().clone() for parameter in self.model.parameters()]

    def _compute_loss(self, inputs, labels, criterion) -> torch.Tensor:
        loss = super()._compute_loss(inputs, labels, criterion)
        proximal_term = sum(
            (local - global_parameter).square().sum()
            for local, global_parameter in zip(self.model.parameters(), self._global_parameters)
        )
        return loss + 0.5 * self.mu * proximal_term


def define_parser():
    parser = define_fedavg_parser()
    parser.description = __doc__
    parser.set_defaults(output_root="/tmp/nvflare/collab/pt_sync_cifar10/fedprox")
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx proximal coefficient")
    return parser


def make_recipe(args):
    return make_fedavg_recipe(
        args,
        client_class=FedProxClient,
        client_options={"mu": args.mu},
        job_name=JOB_NAME,
        extra_files=(EXAMPLE_DIR / "fedavg.py",),
    )


def main():
    args = define_parser().parse_args()
    if not math.isfinite(args.mu) or args.mu <= 0:
        raise ValueError("--mu must be finite and greater than 0; use fedavg.py when no proximal term is needed")
    run_example(args, recipe_factory=make_recipe, algorithm_name="FEDPROX", extra_summary=(("Proximal mu", args.mu),))


if __name__ == "__main__":
    main()
