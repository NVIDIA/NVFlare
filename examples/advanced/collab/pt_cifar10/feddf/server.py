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

"""Synchronous rounds with replaceable weight averaging or KD aggregation."""

import json
import time
from pathlib import Path

import torch

from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper
from nvflare.collab import collab

from ..fedavg.client import BATCH_SIZE, LEARNING_RATE, LOCAL_EPOCHS
from ..fedavg.server import TRAIN_TIMEOUT, FedAvgServer
from ..loader import make_data_loader
from ..model import SimpleNetwork, get_model_state
from .distiller import Cifar10DistillerFactory


def _teacher(weights):
    model = SimpleNetwork()
    model.load_state_dict(weights)
    return model


class MeanLogitEnsemble(torch.nn.Module):
    """Expose current-round mean logits as one teacher for the shared trainer."""

    def __init__(self, states):
        super().__init__()
        self.models = torch.nn.ModuleList([_teacher(state) for state in states])

    def forward(self, inputs):
        return torch.stack([model(inputs) for model in self.models]).mean(dim=0)


class FedDFServer:
    def __init__(
        self,
        num_rounds=20,
        aggregation="feddf",
        split_root="/tmp/cifar10_splits/pt_cifar10_feddf",
        seed=10,
        kd_epochs=1,
        kd_lr=0.001,
        temperature=1.0,
        distiller_factory=None,
    ):
        if aggregation not in ("fedavg", "feddf"):
            raise ValueError("aggregation must be fedavg or feddf")
        if num_rounds < 1 or kd_epochs < 1 or kd_lr <= 0 or temperature <= 0:
            raise ValueError("Round count, KD epochs, learning rate and temperature must be positive")
        self.num_rounds = num_rounds
        self.aggregation = aggregation
        self.split_root = split_root
        self.seed = seed
        self.kd_epochs = kd_epochs
        self.kd_lr = kd_lr
        self.temperature = temperature
        self.distiller_factory = distiller_factory

    def aggregate(self, model, client_results, proxy_loader, round_number):
        """Average current client parameters, then optionally distill mean logits.

        Both modes use uniform averaging. Non-parameter buffers come from the
        last client in stable site order; the shared simple model has none.
        """
        if client_results.failures:
            raise next(iter(client_results.failures.values()))
        # Stable teacher order avoids completion timing changing reduction order.
        results = sorted(client_results)
        if not results:
            raise ValueError("No client models were returned")
        # Average parameters only; retain the final teacher's buffers explicitly.
        helper = WeightedAggregationHelper()
        parameter_names = dict(model.named_parameters())
        for site, result in results:
            helper.add(
                data={name: result["weights"][name] for name in parameter_names},
                weight=1,
                contributor_name=site,
                contribution_round=round_number,
            )
        model.load_state_dict(dict(results[-1][1]["weights"], **helper.get_result()))
        if self.aggregation == "fedavg":
            return {}

        factory = (
            Cifar10DistillerFactory(kd_lr=self.kd_lr, temperature=self.temperature)
            if self.distiller_factory is None
            else self.distiller_factory
        )
        distiller = factory(student=model, teacher=MeanLogitEnsemble([result["weights"] for _, result in results]))
        total_loss, samples, steps = 0.0, 0, 0
        try:
            for _ in range(self.kd_epochs):
                if collab.is_aborted:
                    raise RuntimeError("Distillation aborted")
                metrics = distiller.fit(proxy_loader)
                total_loss += metrics["loss"] * metrics["samples"]
                samples += metrics["samples"]
                steps += metrics["steps"]
            # Backends may own a wrapped/copied student; publish only its snapshot.
            model.load_state_dict(distiller.export_student_state())
        finally:
            distiller.close()
        return {"kd_loss": total_loss / samples, "kd_steps": steps, "kd_samples": samples}

    # Collab invokes this decorated method as the server workflow entry point.
    @collab.main
    def run(self):
        manifest = json.loads((Path(self.split_root) / "manifest.json").read_text())
        if manifest["num_clients"] != len(collab.clients):
            raise ValueError("The client count must match the prepared manifest")
        torch.manual_seed(self.seed)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = SimpleNetwork().to(device)
        test_loader = make_data_loader(train=False, batch_size=BATCH_SIZE)
        proxy_loader = None
        if self.aggregation == "feddf":
            # An independent loader generator prevents teacher construction from
            # changing proxy order. The same prepared transform is used by clients.
            proxy_loader = make_data_loader(
                train=True,
                batch_size=BATCH_SIZE,
                site_name="proxy",
                split_root=self.split_root,
                generator=torch.Generator(device="cpu").manual_seed(self.seed),
            )
        run_dir = Path(collab.workspace.get_run_dir(collab.fl_ctx.get_job_id()))
        run_dir.mkdir(parents=True, exist_ok=True)
        config = dict(
            {key: value for key, value in vars(self).items() if key != "distiller_factory"},
            data=manifest,
            local_batch_size=BATCH_SIZE,
            local_lr=LEARNING_RATE,
            local_epochs=LOCAL_EPOCHS,
            distiller_factory=(
                type(self.distiller_factory).__name__
                if self.distiller_factory is not None
                else "Cifar10DistillerFactory"
            ),
            distiller_backend=getattr(
                self.distiller_factory, "backend", "native" if self.distiller_factory is None else "custom"
            ),
            kd_target="uniform_mean_logits",
            parameter_weighting="uniform",
            buffer_policy="last_site_in_sorted_order",
        )
        (run_dir / "run_config.json").write_text(json.dumps(config, indent=2))
        torch.save(get_model_state(model), run_dir / "initial_model.pt")
        history = []
        for round_number in range(self.num_rounds):
            if collab.is_aborted:
                raise RuntimeError("Training aborted")
            # Invoke the FedAvgClient published train method and wait for replies.
            results = collab.clients(timeout=TRAIN_TIMEOUT).train(get_model_state(model))
            started = time.perf_counter()
            metrics = self.aggregate(model, results, proxy_loader, round_number)
            metrics["aggregation_seconds"] = time.perf_counter() - started
            accuracy = FedAvgServer.evaluate(model, test_loader)
            history.append(
                dict(
                    round=round_number + 1,
                    accuracy=accuracy,
                    clients=sorted(name for name, _ in results),
                    local_steps=sum(result["num_steps"] for _, result in results),
                    **metrics,
                )
            )
            (run_dir / "metrics.json").write_text(json.dumps(history, indent=2))
            print(f"Round {round_number + 1}: {self.aggregation}, test accuracy={accuracy:.4f}")
        weights = get_model_state(model)
        torch.save(weights, run_dir / "final_model.pt")
        return weights
