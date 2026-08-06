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

"""Synchronous CIFAR-10 SCAFFOLD with direct model and control exchange."""

from pathlib import Path

import torch
from data import make_test_loader
from fedavg import FedAvgClient, FedAvgServer, collect_client_updates
from fedavg import define_parser as define_fedavg_parser
from fedavg import make_recipe as make_fedavg_recipe
from fedavg import resolve_device, run_example, seed_everything, weighted_average
from model import Cifar10CNN, get_model_state

from nvflare.collab import collab

JOB_NAME = "collab_pt_sync_cifar10_scaffold"
EXAMPLE_DIR = Path(__file__).resolve().parent


def weighted_control_average(client_results) -> dict[str, torch.Tensor]:
    """Average control deltas by local steps, matching NVFlare's framework SCAFFOLD path.

    Canonical SCAFFOLD uses an unweighted participating-client average. This
    example intentionally uses the framework-compatible variant so its Collab
    and standard-recipe comparisons have the same aggregation behavior.
    """

    updates = dict(client_results)
    if not updates:
        raise ValueError("SCAFFOLD control aggregation requires at least one client update")
    total_weight = sum(update["num_steps"] for update in updates.values())
    if total_weight <= 0:
        raise ValueError("SCAFFOLD client updates contain no training steps")
    control_names = set(next(iter(updates.values()))["control_delta"])
    averaged = {}
    for name in sorted(control_names):
        reference = next(iter(updates.values()))["control_delta"][name]
        accumulator = torch.zeros_like(reference, device="cpu")
        for site_name, update in updates.items():
            if update["num_steps"] <= 0:
                raise ValueError(f"{site_name} returned an invalid step count")
            delta = update["control_delta"]
            if set(delta) != control_names or delta[name].shape != reference.shape:
                raise ValueError(f"{site_name} returned incompatible SCAFFOLD controls")
            accumulator.add_(delta[name].cpu(), alpha=update["num_steps"])
        averaged[name] = accumulator / total_weight
    return averaged


class ScaffoldClient(FedAvgClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_controls = None
        self._control_correction = None
        self._lr_exposure = 0.0

    def _after_optimizer_step(self, optimizer) -> None:
        step_lr = optimizer.param_groups[0]["lr"]
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                parameter.add_(self._control_correction[name], alpha=-step_lr)
        self._lr_exposure += step_lr

    def _update_local_controls(self, global_weights, global_controls) -> dict[str, torch.Tensor]:
        if self._lr_exposure <= 0:
            raise RuntimeError("SCAFFOLD local training completed without a positive learning-rate exposure")

        new_local_controls = {}
        control_delta = {}
        for name, parameter in self.model.named_parameters():
            old_local = self.local_controls[name]
            new_local = (
                old_local
                - global_controls[name].to(old_local)
                + (global_weights[name].to(old_local) - parameter.detach().cpu()) / self._lr_exposure
            )
            new_local_controls[name] = new_local
            control_delta[name] = new_local - old_local
        self.local_controls = new_local_controls
        return control_delta

    @collab.publish
    def train(
        self,
        round_number: int,
        global_weights: dict[str, torch.Tensor],
        global_controls: dict[str, torch.Tensor],
    ) -> dict:
        if collab.is_aborted:
            raise RuntimeError("Training aborted")

        self.model.load_state_dict(global_weights, strict=True)
        parameters = dict(self.model.named_parameters())
        if set(global_controls) != set(parameters):
            raise ValueError("Global SCAFFOLD controls do not match the model parameters")
        if self.local_controls is None:
            self.local_controls = {
                name: torch.zeros_like(parameter, device="cpu") for name, parameter in parameters.items()
            }

        self._control_correction = {}
        for name, parameter in parameters.items():
            global_control = global_controls[name]
            if global_control.shape != parameter.shape:
                raise ValueError(f"Global SCAFFOLD control has an incompatible shape for {name}")
            self._control_correction[name] = global_control.to(parameter) - self.local_controls[name].to(parameter)

        self._lr_exposure = 0.0
        train_loss, local_steps = self._local_train()
        control_delta = self._update_local_controls(global_weights, global_controls)
        self._control_correction = None

        print(f"[{collab.site_name}] round={round_number} train_loss={train_loss:.4f}")
        return {
            "weights": get_model_state(self.model),
            "control_delta": control_delta,
            "num_steps": local_steps,
            "train_loss": train_loss,
        }


class ScaffoldServer(FedAvgServer):
    @collab.main
    def run(self) -> dict[str, torch.Tensor]:
        seed_everything(self.seed)
        device = resolve_device(self.device_name)
        model = Cifar10CNN().to(device)
        test_loader = make_test_loader(self.data_root, self.eval_batch_size, self.num_workers)
        global_weights = get_model_state(model)
        global_controls = {
            name: torch.zeros_like(parameter, device="cpu") for name, parameter in model.named_parameters()
        }
        history = [{"round": 0, "test_accuracy": self.evaluate(model, test_loader)}]
        print(f"Initial test_accuracy={history[0]['test_accuracy']:.4f}")

        for round_number in range(1, self.num_rounds + 1):
            print(f"=== SCAFFOLD round {round_number}/{self.num_rounds} ===")
            client_results = collab.clients.train(round_number, global_weights, global_controls)
            updates = collect_client_updates(client_results, self.min_clients)
            global_weights, train_loss = weighted_average(updates, self.min_clients)
            control_delta = weighted_control_average(updates)
            for name, delta in control_delta.items():
                global_controls[name].add_(delta)

            model.load_state_dict(global_weights, strict=True)
            accuracy = self.evaluate(model, test_loader)
            print(f"Aggregated train_loss={train_loss:.4f}, test_accuracy={accuracy:.4f}")
            history.append({"round": round_number, "test_accuracy": accuracy, "train_loss": train_loss})

        self.save_outputs(global_weights, history)
        return global_weights


def define_parser():
    parser = define_fedavg_parser()
    parser.description = __doc__
    parser.set_defaults(output_root="/tmp/nvflare/collab/pt_sync_cifar10/scaffold")
    return parser


def make_recipe(args):
    return make_fedavg_recipe(
        args,
        server_class=ScaffoldServer,
        client_class=ScaffoldClient,
        job_name=JOB_NAME,
        extra_files=(EXAMPLE_DIR / "fedavg.py",),
    )


def main():
    run_example(define_parser().parse_args(), recipe_factory=make_recipe, algorithm_name="SCAFFOLD")


if __name__ == "__main__":
    main()
