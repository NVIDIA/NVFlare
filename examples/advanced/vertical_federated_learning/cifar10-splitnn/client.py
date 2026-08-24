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

"""Client-side data, model-half ownership, and SplitNN training.

Only methods decorated with @collab.publish cross a site boundary. The other
methods are ordinary local Python calls, which keeps the algorithm flow visible.
"""

import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
from data import Cifar10SplitDataset
from model import BottomModel, TopModel
from torch.utils.tensorboard import SummaryWriter

from nvflare.collab import collab
from nvflare.fuel.utils.log_utils import get_obj_logger

NUM_STEPS = 15_625
BATCH_SIZE = 64
LEARNING_RATE = 0.01
VALIDATION_FREQUENCY = 1_000
LOG_FREQUENCY = 100
CALL_TIMEOUT = 600.0
TRAINING_SEED = 42
MODEL_FILE_NAMES = {"image": "bottom_model.pt", "label": "top_model.pt"}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _to_wire(tensor: torch.Tensor) -> torch.Tensor:
    """Encode a tensor once for cross-site transfer."""
    return tensor.detach().to(device="cpu", dtype=torch.float16)


def _from_wire(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Restore a transferred tensor to the local training precision."""
    return tensor.to(device=device, dtype=torch.float32)


class SplitNNClient:
    """Run the model half assigned to this site.

    The recipe assigns one site the image role and another the label role.
    The methods are deliberately ordinary tensor-in/tensor-out methods;
    Collab API publishes them without application-level message conversion.
    """

    def __init__(self):
        self.dataset_root = None
        self.intersection_file = None
        self.device = None
        self.logger = get_obj_logger(self)

        self.role = None
        self.label_site = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_dataset = None
        self.valid_dataset = None
        self._train_activations = None

    # @collab.init runs once after the Collab runtime is ready at each site.
    # Client-specific configuration, data, and model state are initialized here.
    @collab.init
    def initialize(self):
        # get_app_prop reads common and per-site values configured by job.py.
        self.dataset_root = collab.get_app_prop("dataset_root")
        self.intersection_file = collab.get_app_prop("intersection_file")
        self.role = collab.get_app_prop("role")
        self.label_site = collab.get_app_prop("label_site")
        if self.role not in ("image", "label"):
            raise ValueError(f"[{collab.site_name}] expected configured role 'image' or 'label', got {self.role!r}")

        _seed_everything(TRAINING_SEED)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Build both halves in the ModerateCNN order so each site's
        # selected half has the same seeded initialization as the full model.
        split_models = {"image": BottomModel(), "label": TopModel()}
        self.model = split_models[self.role].to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss() if self.role == "label" else None
        self.train_dataset = Cifar10SplitDataset(
            self.dataset_root, intersection_file=self.intersection_file, role=self.role, train=True
        )
        self.valid_dataset = Cifar10SplitDataset(
            self.dataset_root, intersection_file=self.intersection_file, role=self.role, train=False
        )
        self.logger.info(
            f"[{collab.site_name}] initialized {self.role}-side model on {self.device} with "
            f"{len(self.train_dataset)} aligned training samples"
        )

    def _require_role(self, expected: str) -> None:
        if self.role != expected:
            raise RuntimeError(f"[{collab.site_name}] operation requires role {expected!r}, got {self.role!r}")

    def _get_label_client(self):
        # get_clients validates the configured site name and returns its proxy.
        # Calling the proxy invokes a published method on that remote client.
        return collab.get_clients([self.label_site])[0]

    def _validate_splitnn(self, label_client, step, writer):
        loss_sum = 0.0
        correct = 0
        count = 0
        test_size = len(self.valid_dataset)
        num_batches = math.ceil(test_size / BATCH_SIZE)
        for batch_indices in np.array_split(np.arange(test_size), num_batches):
            activations = self._validation_forward(batch_indices)
            # Native tensors, indices, and dict results need no application-level
            # DXO, Shareable, or serialization conversion.
            metrics = label_client(timeout=CALL_TIMEOUT).validation_metrics(batch_indices, activations)
            loss_sum += metrics["loss"] * metrics["count"]
            correct += metrics["correct"]
            count += metrics["count"]

        result = {"loss": loss_sum / count, "accuracy": correct / count}
        self.logger.info(
            f"step {step + 1}: validation loss={result['loss']:.4f}, " f"accuracy={result['accuracy']:.4f}"
        )
        writer.add_scalar("val_loss", result["loss"], step)
        writer.add_scalar("val_accuracy", result["accuracy"], step)
        return result

    # @collab.publish exposes this method to remote Collab callers. The
    # server invokes it on the image site, which then owns the complete training loop.
    @collab.publish
    def run_splitnn(self):
        train_size = len(self.train_dataset)
        self._require_role("image")
        label_client = self._get_label_client()
        # The Collab context provides the current job ID and its site-local run directory.
        run_dir = collab.workspace.get_run_dir(collab.fl_ctx.get_job_id())
        tensorboard_dir = os.path.join(run_dir, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        rng = np.random.RandomState(TRAINING_SEED)
        gradients = None
        last_metrics = None
        steps_completed = 0

        self.logger.info(
            f"starting image-side SplitNN loop for {NUM_STEPS} steps with batch size {BATCH_SIZE}; "
            f"TensorBoard logs: {tensorboard_dir}"
        )
        try:
            for step in range(NUM_STEPS):
                # is_aborted lets long-running published methods stop cooperatively.
                if collab.is_aborted:
                    self.logger.info("SplitNN run aborted")
                    break

                if gradients is not None:
                    self._backward(gradients)
                batch_indices = rng.randint(0, train_size, size=BATCH_SIZE)
                activations = self._forward(batch_indices)
                if activations is None:
                    self.logger.info("SplitNN run aborted during image-side forward pass")
                    break
                # This is the key client-to-client interaction: the image site directly
                # consumes the tensor tuple returned by the label site.
                gradients, metrics = label_client(timeout=CALL_TIMEOUT).compute_loss(batch_indices, activations)
                last_metrics = metrics
                steps_completed = step + 1
                writer.add_scalar("train_loss", metrics["loss"], step)
                writer.add_scalar("train_accuracy", metrics["accuracy"], step)
                if step % LOG_FREQUENCY == 0:
                    self.logger.info(
                        f"step {step + 1}/{NUM_STEPS}: train loss={metrics['loss']:.4f}, "
                        f"accuracy={metrics['accuracy']:.4f}"
                    )
                if step % VALIDATION_FREQUENCY == 0:
                    self._validate_splitnn(label_client, step, writer)

            if collab.is_aborted:
                return {
                    "steps_completed": steps_completed,
                    "model_paths": None,
                    "last_metrics": last_metrics,
                }

            model_paths = {
                "bottom_model": self._save_model(),
                "top_model": label_client(timeout=CALL_TIMEOUT).save_model(),
            }
            return {
                "steps_completed": steps_completed,
                "model_paths": model_paths,
                "last_metrics": last_metrics,
            }
        finally:
            writer.close()

    def _forward(self, batch_indices):
        """Run the image-side forward pass and retain its autograd graph."""
        self._require_role("image")
        if collab.is_aborted:
            return None

        self.model.train()
        inputs = self.train_dataset.get_batch(batch_indices).to(self.device)
        self._train_activations = self.model(inputs)
        cut_activations = self._train_activations.flatten(start_dim=1)
        return _to_wire(cut_activations)

    # Published because the image site calls this method on the label site.
    @collab.publish
    def compute_loss(self, batch_indices, activations):
        """Update the label-side model and return cut-layer gradients."""
        self._require_role("label")
        if activations is None:
            raise RuntimeError("image-side forward pass returned no activations")

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        labels = self.train_dataset.get_batch(batch_indices).to(self.device)
        activations = _from_wire(activations, self.device).detach().requires_grad_(True)
        predictions = self.model(activations)
        loss = self.criterion(predictions, labels)
        loss.backward()

        if activations.grad is None:
            raise RuntimeError("label-side backward pass did not produce cut-layer gradients")
        gradients = activations.grad
        accuracy = predictions.detach().argmax(dim=1).eq(labels).float().mean().item()
        self.optimizer.step()

        return _to_wire(gradients), {"loss": loss.item(), "accuracy": accuracy}

    def _backward(self, gradients):
        """Apply label-side cut-layer gradients to the image-side model."""
        self._require_role("image")
        if self._train_activations is None:
            raise RuntimeError("backward() called before a successful forward()")

        self.optimizer.zero_grad(set_to_none=True)
        gradients = _from_wire(gradients, self.device)
        self._train_activations.backward(gradients.reshape(self._train_activations.shape))
        self.optimizer.step()
        self._train_activations = None

    def _validation_forward(self, batch_indices):
        """Return image-side validation activations."""
        self._require_role("image")
        self.model.eval()
        with torch.no_grad():
            inputs = self.valid_dataset.get_batch(batch_indices).to(self.device)
            activations = self.model(inputs).flatten(start_dim=1)
        return _to_wire(activations)

    # Published because validation also crosses from site-1 to site-2.
    @collab.publish
    def validation_metrics(self, batch_indices, activations):
        """Compute label-side validation loss and correct predictions."""
        self._require_role("label")
        self.model.eval()
        with torch.no_grad():
            labels = self.valid_dataset.get_batch(batch_indices).to(self.device)
            activations = _from_wire(activations, self.device)
            predictions = self.model(activations)
            loss = self.criterion(predictions, labels).item()
            correct = predictions.argmax(dim=1).eq(labels).sum().item()
        return {"loss": loss, "correct": correct, "count": len(labels)}

    def _save_model(self) -> str:
        """Persist this site's model half without transferring its parameters."""
        run_dir = collab.workspace.get_run_dir(collab.fl_ctx.get_job_id())
        model_dir = os.path.join(run_dir, "final_model")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, MODEL_FILE_NAMES[self.role])
        torch.save(self.model.state_dict(), model_path)
        self.logger.info(f"saved trained {self.role}-side model to {model_path}")
        return model_path

    # Published so the image-side coordinator can request site-local persistence.
    @collab.publish
    def save_model(self):
        """Save this site's model half locally and return only its path."""
        return self._save_model()
