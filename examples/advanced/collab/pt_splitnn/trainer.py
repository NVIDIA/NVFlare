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

"""Image-side and label-side SplitNN operations exposed through CollabAPI."""

import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
from data import Cifar10SplitDataset, validate_prepared_data
from model import BottomModel, TopModel
from torch.utils.tensorboard import SummaryWriter

from nvflare.collab import collab
from nvflare.fuel.utils.log_utils import get_obj_logger


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class SplitNNTrainer:
    """Run the model half assigned to this site.

    ``site-1`` is configured with the image role and ``site-2`` with the label
    role. The methods are deliberately ordinary tensor-in/tensor-out methods;
    CollabAPI publishes them without application-level message conversion.
    """

    def __init__(
        self,
        data_root: str,
        learning_rate: float = 0.01,
        fp16: bool = True,
        device: str | None = None,
        seed: int = 42,
        num_threads: int | None = None,
    ):
        self.data_root = data_root
        self.learning_rate = learning_rate
        self.fp16 = fp16
        self.device = device
        self.seed = seed
        self.num_threads = num_threads
        self.logger = get_obj_logger(self)

        self.role = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_dataset = None
        self.valid_dataset = None
        self._train_activations = None

    @collab.init
    def initialize(self):
        if self.num_threads is not None:
            torch.set_num_threads(self.num_threads)
        self.data_root = collab.get_app_prop("data_root", self.data_root)
        self.role = collab.get_app_prop("role")
        if self.role not in ("image", "label"):
            raise ValueError(f"[{collab.site_name}] expected configured role 'image' or 'label', got {self.role!r}")

        validate_prepared_data(self.data_root)
        _seed_everything(self.seed)
        self.device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # Build both halves in the original ModerateCNN order so each site's
        # selected half has the same seeded initialization as the full model.
        split_models = {"image": BottomModel(), "label": TopModel()}
        self.model = split_models[self.role].to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss() if self.role == "label" else None
        self.train_dataset = Cifar10SplitDataset(self.data_root, role=self.role, train=True)
        self.valid_dataset = Cifar10SplitDataset(self.data_root, role=self.role, train=False)
        self.logger.info(
            f"[{collab.site_name}] initialized {self.role}-side model on {self.device} with "
            f"{len(self.train_dataset)} aligned training samples"
        )

    def _require_role(self, expected: str) -> None:
        if self.role != expected:
            raise RuntimeError(f"[{collab.site_name}] operation requires role {expected!r}, got {self.role!r}")

    def _get_label_client(self):
        clients = {client.name: client for client in collab.clients}
        label_client = clients.get("site-2")
        if label_client is None:
            raise RuntimeError("SplitNN requires label-side client 'site-2'")
        return label_client

    def _validate_splitnn(self, label_client, step, test_size, batch_size, call_timeout, writer):
        losses = []
        correct = 0
        count = 0
        num_batches = math.ceil(test_size / batch_size)
        for batch_indices in np.array_split(np.arange(test_size), num_batches):
            activations = self.validation_forward(batch_indices)
            metrics = label_client(timeout=call_timeout).validation_metrics(batch_indices, activations)
            losses.append(metrics["loss"])
            correct += metrics["correct"]
            count += metrics["count"]

        result = {"loss": float(np.mean(losses)), "accuracy": correct / count}
        self.logger.info(
            f"step {step + 1}: validation loss={result['loss']:.4f}, " f"accuracy={result['accuracy']:.4f}"
        )
        writer.add_scalar("val_loss", result["loss"], step)
        writer.add_scalar("val_accuracy", result["accuracy"], step)
        return result

    @collab.publish
    def run_splitnn(
        self,
        train_size: int,
        test_size: int,
        num_steps: int,
        batch_size: int,
        validation_frequency: int,
        call_timeout: float,
        seed: int,
        log_frequency: int,
    ):
        self._require_role("image")
        label_client = self._get_label_client()
        run_dir = collab.workspace.get_run_dir(collab.fl_ctx.get_job_id())
        tensorboard_dir = os.path.join(run_dir, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        rng = np.random.RandomState(seed)
        gradients = None
        last_metrics = None
        steps_completed = 0

        self.logger.info(
            f"starting image-side SplitNN loop for {num_steps} steps with batch size {batch_size}; "
            f"TensorBoard logs: {tensorboard_dir}"
        )
        try:
            for step in range(num_steps):
                if collab.is_aborted:
                    self.logger.info("SplitNN run aborted")
                    break

                if gradients is not None:
                    self.backward(gradients)
                batch_indices = rng.randint(0, train_size - 1, size=batch_size)
                activations = self.forward(batch_indices)
                gradients, metrics = label_client(timeout=call_timeout).compute_loss(batch_indices, activations)
                last_metrics = metrics
                steps_completed = step + 1
                writer.add_scalar("train_loss", metrics["loss"], step)
                writer.add_scalar("train_accuracy", metrics["accuracy"], step)
                if step % log_frequency == 0:
                    self.logger.info(
                        f"step {step + 1}/{num_steps}: train loss={metrics['loss']:.4f}, "
                        f"accuracy={metrics['accuracy']:.4f}"
                    )
                if validation_frequency > 0 and step % validation_frequency == 0:
                    self._validate_splitnn(label_client, step, test_size, batch_size, call_timeout, writer)

            models = {
                "bottom_model": self.get_model(),
                "top_model": label_client(timeout=call_timeout).get_model(),
            }
            model_dir = os.path.join(run_dir, "final_model")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "splitnn_model.pt")
            torch.save(models, model_path)
            self.logger.info(f"saved trained SplitNN model to {model_path}")
            return {
                "steps_completed": steps_completed,
                "model_path": model_path,
                "last_metrics": last_metrics,
            }
        finally:
            writer.close()

    @collab.publish
    def forward(self, batch_indices):
        """Run the image-side forward pass and retain its autograd graph."""
        self._require_role("image")
        if collab.is_aborted:
            return None

        self.model.train()
        inputs = self.train_dataset.get_batch(batch_indices).to(self.device)
        self._train_activations = self.model(inputs)
        cut_activations = self._train_activations.detach().flatten(start_dim=1)
        if self.fp16:
            cut_activations = cut_activations.to(dtype=torch.float16)
        return cut_activations.cpu()

    @collab.publish
    def compute_loss(self, batch_indices, activations):
        """Update the label-side model and return cut-layer gradients."""
        self._require_role("label")
        if activations is None:
            raise RuntimeError("image-side forward pass returned no activations")

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        labels = self.train_dataset.get_batch(batch_indices).to(self.device)
        activations = activations.to(device=self.device, dtype=torch.float32).detach().requires_grad_(True)
        predictions = self.model(activations)
        loss = self.criterion(predictions, labels)
        loss.backward()

        if activations.grad is None:
            raise RuntimeError("label-side backward pass did not produce cut-layer gradients")
        gradients = activations.grad.detach()
        accuracy = predictions.detach().argmax(dim=1).eq(labels).float().mean().item()
        self.optimizer.step()

        if self.fp16:
            gradients = gradients.to(dtype=torch.float16)
        return gradients.cpu(), {"loss": loss.item(), "accuracy": accuracy}

    @collab.publish
    def backward(self, gradients):
        """Apply label-side cut-layer gradients to the image-side model."""
        self._require_role("image")
        if self._train_activations is None:
            raise RuntimeError("backward() called before a successful forward()")

        self.optimizer.zero_grad(set_to_none=True)
        gradients = gradients.to(device=self.device, dtype=torch.float32)
        self._train_activations.backward(gradients.reshape(self._train_activations.shape))
        self.optimizer.step()
        self._train_activations = None

    @collab.publish
    def validation_forward(self, batch_indices):
        """Return image-side validation activations."""
        self._require_role("image")
        self.model.eval()
        with torch.no_grad():
            inputs = self.valid_dataset.get_batch(batch_indices).to(self.device)
            activations = self.model(inputs).flatten(start_dim=1)
        if self.fp16:
            activations = activations.to(dtype=torch.float16)
        return activations.cpu()

    @collab.publish
    def validation_metrics(self, batch_indices, activations):
        """Compute label-side validation loss and correct predictions."""
        self._require_role("label")
        self.model.eval()
        with torch.no_grad():
            labels = self.valid_dataset.get_batch(batch_indices).to(self.device)
            activations = activations.to(device=self.device, dtype=torch.float32)
            predictions = self.model(activations)
            loss = self.criterion(predictions, labels).item()
            correct = predictions.argmax(dim=1).eq(labels).sum().item()
        return {"loss": loss, "correct": correct, "count": len(labels)}

    @collab.publish
    def get_model(self):
        """Return this site's trained model parameters."""
        return {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()}
