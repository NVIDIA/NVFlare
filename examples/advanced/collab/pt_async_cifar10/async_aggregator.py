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

"""CIFAR-10 model, evaluation, and artifact hooks for AsyncRecipe."""

import gc
import json
import os
import random

import numpy as np
import torch
from model import ModerateCNN, get_model_params, load_model_params
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from nvflare.collab import collab
from nvflare.fuel.utils.log_utils import get_obj_logger

_TEST_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        ),
    ]
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Cifar10AsyncApplication:
    """Own CIFAR concerns while AsyncRecipe owns scheduling and aggregation."""

    def __init__(
        self,
        data_root: str,
        buffer_size: int = 4,
        device: str | None = None,
        eval_batch_size: int = 300,
        setup_seed: int = 10,
        run_seed: int = 10,
        checkpoint_interval: int = 0,
        num_rounds: int = 100,
    ):
        if eval_batch_size < 1:
            raise ValueError("eval_batch_size must be >= 1")
        if checkpoint_interval < 0:
            raise ValueError("checkpoint_interval must be >= 0")
        self.data_root = data_root
        self.buffer_size = buffer_size
        self.device = device
        self.eval_batch_size = eval_batch_size
        self.setup_seed = setup_seed
        self.run_seed = run_seed
        self.checkpoint_interval = checkpoint_interval
        self.num_rounds = num_rounds
        self.logger = get_obj_logger(self)
        self._test_loader = None
        self._eval_model = None
        self._writer = None
        self._test_history = []
        self._run_dir = None

    def _new_model(self):
        return ModerateCNN(seed=self.setup_seed)

    def _device(self) -> torch.device:
        return torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def _init_outputs(self):
        self._run_dir = collab.workspace.get_run_dir(collab.fl_ctx.get_job_id())
        tensorboard_dir = os.path.join(self._run_dir, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        self._writer = SummaryWriter(log_dir=tensorboard_dir)
        self._eval_snapshot_dir = os.path.join(self._run_dir, "eval_snapshots")
        os.makedirs(self._eval_snapshot_dir, exist_ok=True)
        # Start a fresh trace for this run.
        with open(os.path.join(self._run_dir, "schedule.jsonl"), "w"):
            pass
        self.logger.info(f"TensorBoard logs: {tensorboard_dir}")

    def _init_test_loader(self):
        if self._test_loader is None:
            test_set = datasets.CIFAR10(
                root=self.data_root,
                train=False,
                download=False,
                transform=_TEST_TRANSFORM,
            )
            self._test_loader = DataLoader(test_set, batch_size=self.eval_batch_size, shuffle=False, num_workers=0)

    def _evaluate_model(self, global_model: dict[str, torch.Tensor], round_index: int) -> float:
        self._init_test_loader()
        device = self._device()
        if self._eval_model is None:
            self._eval_model = self._new_model()
        load_model_params(self._eval_model, global_model, target_device=device)
        self._eval_model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self._test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                predictions = self._eval_model(inputs).argmax(dim=1)
                correct += predictions.eq(labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total else 0.0
        display_round = "initial" if round_index < 0 else str(round_index + 1)
        self.logger.info(f"[{collab.call_info}] round {display_round}: test accuracy={accuracy:.4f}")
        self._writer.add_scalar("test/accuracy", accuracy, round_index + 1)
        self._writer.flush()
        self._test_history.append(
            {
                "aggregation": round_index + 1,
                "accepted_updates": max(0, round_index + 1) * self.buffer_size,
                "accuracy": accuracy,
            }
        )
        with open(os.path.join(self._run_dir, "accuracy_history.json"), "w") as history_file:
            json.dump(self._test_history, history_file, indent=2)
        return accuracy

    def _evaluate_snapshots(self) -> None:
        self.logger.info(f"evaluating {self.num_rounds + 1} saved global model versions on the server")
        for model_version in range(self.num_rounds + 1):
            snapshot_path = os.path.join(self._eval_snapshot_dir, f"model_version_{model_version}.pt")
            snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=True)
            self._evaluate_model(snapshot, round_index=model_version - 1)

    def initialize(self):
        seed_everything(self.setup_seed)
        self._init_outputs()
        initial = get_model_params(self._new_model(), target_device="cpu")
        seed_everything(self.run_seed)
        return initial

    def on_model(self, model, version):
        torch.save(model, os.path.join(self._eval_snapshot_dir, f"model_version_{version}.pt"))
        if (
            version
            and self.checkpoint_interval
            and (version % self.checkpoint_interval == 0 or version == self.num_rounds)
        ):
            round_dir = os.path.join(self._run_dir, "round_models")
            os.makedirs(round_dir, exist_ok=True)
            torch.save(model, os.path.join(round_dir, f"model_round_{version}.pt"))
        gc.collect()

    def on_event(self, event):
        from dataclasses import asdict

        with open(os.path.join(self._run_dir, "schedule.jsonl"), "a") as trace:
            trace.write(json.dumps(asdict(event)) + "\n")

    def on_finish(self, model, version):
        self._evaluate_snapshots()
        final_dir = os.path.join(self._run_dir, "final_model")
        os.makedirs(final_dir, exist_ok=True)
        torch.save(model, os.path.join(final_dir, "global_model.pt"))

    def close(self):
        if self._writer:
            self._writer.close()
