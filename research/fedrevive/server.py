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

"""FedRevive model operations, initialization and experiment artifacts."""

import json
from dataclasses import asdict
from pathlib import Path

import torch
from data import test_transform
from fedrevive import model_delta
from model import create_model, get_model_params, load_model_params, reset_tracker_stats_in_state, set_seed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets

from nvflare.apis.simulation_plan import SIMULATION_PLAN_PROP
from nvflare.collab import collab


class FedReviveModelOps:
    """Research arithmetic, including the reference tracker reset on publication.

    FedRevive clients upload full models. These models contain only floating
    parameters/statistics; tensor shape/dtype/finite checks precede acceptance.
    """

    def snapshot(self, model):
        return {name: value.detach().cpu().clone() for name, value in model.items()}

    def validate(self, model, reference):
        if model.keys() != reference.keys():
            raise ValueError("model keys do not match reference")
        for name, value in model.items():
            base = reference[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != base.shape
                or value.dtype != base.dtype
                or not value.is_floating_point()
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"incompatible or non-finite model entry: {name}")

    def difference(self, model, base):
        return model_delta(model, base)

    def apply(self, model, delta, scale):
        result = {name: value.detach().cpu().clone() + scale * delta[name] for name, value in model.items()}
        return reset_tracker_stats_in_state(result)


class FedReviveModelLoader:
    """Construct the initial model using the reference setup seed."""

    def __init__(self, setup_seed=10):
        self.setup_seed = setup_seed

    def load_model(self):
        torch.set_num_threads(1)
        set_seed(self.setup_seed)
        return get_model_params(create_model(), target_device="cpu")


class FedReviveObserver:
    """Evaluation and artifacts; no transport or aggregation state."""

    def __init__(
        self,
        data_root,
        run_config,
        device=None,
        eval_batch_size=300,
        eval_interval=10,
    ):
        if eval_batch_size < 1 or eval_interval < 1:
            raise ValueError("evaluation sizes and interval must be positive")
        self.data_root = data_root
        self.run_config = run_config
        self.device = device
        self.eval_batch_size = eval_batch_size
        self.eval_interval = eval_interval

    def _device(self):
        return torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def initialize(self):
        self.run_dir = Path(collab.workspace.get_run_dir(collab.fl_ctx.get_job_id()))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        config = self.run_dir / "run_config.json"
        with config.open("x") as stream:
            json.dump(self.run_config, stream, indent=2)
        self.writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))
        self.history = []
        self.simulated_time = 0.0 if self.run_config["arrival_mode"] == "simulated" else None
        self._eval_model = None
        self.test_loader = DataLoader(
            datasets.CIFAR10(root=self.data_root, train=False, download=False, transform=test_transform()),
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=0,
            # DataLoader draws an iterator seed even for sequential evaluation.
            generator=torch.Generator(device="cpu").manual_seed(0),
        )

    def on_time(self, simulated_time):
        self.simulated_time = simulated_time

    def on_event(self, event):
        with (self.run_dir / "schedule.jsonl").open("a") as stream:
            stream.write(json.dumps({**asdict(event), "simulated_time": self.simulated_time}) + "\n")

    def on_update(self, update):
        # Actual-arrival assignment metadata is empty; the uploaded metadata
        # still records the client-applied seed and persistent shard identity.
        with (self.run_dir / "updates.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "assignment_id": update.assignment.assignment_id,
                        "current_version": update.current_version,
                        "metadata": update.metadata,
                        "metrics": update.metrics,
                    }
                )
                + "\n"
            )

    def on_model(self, model, version):
        if version % self.eval_interval == 0:
            self._evaluate(model, version)

    def _evaluate(self, model, version):
        if self._eval_model is None:
            # Evaluation loads published weights. Discard the RNG draws for
            # the temporary CPU initialization so they cannot shift KD draws.
            with torch.random.fork_rng(devices=[]):
                self._eval_model = create_model()
        load_model_params(self._eval_model, model, target_device=self._device())
        self._eval_model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self._device()), labels.to(self._device())
                outputs = self._eval_model(inputs)
                loss_sum += torch.nn.functional.cross_entropy(outputs, labels, reduction="sum").item()
                correct += outputs.argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        record = {
            "model_version": version,
            "simulated_time": self.simulated_time,
            "accuracy": correct / total,
            "test_loss": loss_sum / total,
            "final": False,
        }
        self.history.append(record)
        self.writer.add_scalar("test/accuracy", record["accuracy"], version, walltime=self.simulated_time)
        self.writer.flush()
        (self.run_dir / "accuracy_history.json").write_text(json.dumps(self.history, indent=2))

    def on_finish(self, model, version):
        # The replay horizon can advance beyond the last observed phase event.
        runtime = collab.fl_ctx.get_prop(SIMULATION_PLAN_PROP)
        if runtime is not None and runtime.virtual_time is not None:
            self.simulated_time = runtime.virtual_time
        if not self.history or self.history[-1]["model_version"] != version:
            self._evaluate(model, version)
        self.history[-1]["final"] = True
        (self.run_dir / "accuracy_history.json").write_text(json.dumps(self.history, indent=2))
        result = {
            **self.run_config,
            "model_version": version,
            "simulated_time": self.simulated_time,
            "history": self.history,
        }
        (self.run_dir / "results.json").write_text(json.dumps(result, indent=2))
        (self.run_dir / "final_model").mkdir(exist_ok=True)
        torch.save(model, self.run_dir / "final_model" / "global_model.pt")

    def close(self):
        writer = getattr(self, "writer", None)
        if writer is not None:
            writer.close()
        self._eval_model = None
