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

"""The original CIFAR-10 SplitNN computation expressed as Collab calls.

``site-1`` owns images and the convolutional half of ModerateCNN. ``site-2``
owns labels and the fully connected half. The controller/executor/DXO plumbing
from the original example becomes ordinary calls::

    if gradient is not None:
        image_site.backward(gradient)
    sample_ids, activations = image_site.forward()
    gradient, loss, accuracy = label_site.compute_gradient(sample_ids, activations)

Run from the ``examples`` directory::

    python -m collab.split_learning.split_learning
"""

import random
import threading
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

from nvflare.collab import collab

IMAGE_SITE = "site-1"
LABEL_SITE = "site-2"
_RNG_LOCK = threading.RLock()


def set_seed(seed):
    """Use the same deterministic setup as the original ModerateCNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ModerateCNN comes from IBM FedMA:
# https://github.com/IBM/FedMA/blob/master/model.py
#
# MIT License
#
# Copyright (c) 2020 International Business Machines
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


class ModerateCNN(nn.Module):
    """The same CNN used by the original CIFAR-10 SplitNN example."""

    def __init__(self, seed=42):
        set_seed(seed)
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10),
        )

    def forward(self, inputs):
        outputs = self.conv_layer(inputs)
        outputs = outputs.view(outputs.size(0), -1)
        return self.fc_layer(outputs)


class SplitNN(ModerateCNN):
    """Expose one half of ModerateCNN while retaining its full state dict."""

    def __init__(self, split_id, seed=42):
        super().__init__(seed=seed)
        if split_id not in (0, 1):
            raise ValueError(f"split_id must be 0 or 1, but was {split_id}")
        self.split_id = split_id
        self.split_forward = self.conv_layer if split_id == 0 else self.fc_layer

    def forward(self, inputs):
        return self.split_forward(inputs)


class CIFAR10SplitDataset:
    """CIFAR-10 batch access identical to the original SplitNN dataset."""

    def __init__(self, root, train, transform, download, returns, intersect_indices=None):
        self.transform = transform
        self.returns = returns
        if intersect_indices is not None:
            intersect_indices = np.sort(intersect_indices).astype(np.int64)

        cifar = datasets.CIFAR10(root=root, train=train, transform=transform, download=download)
        self.data = cifar.data
        self.targets = np.asarray(cifar.targets)
        self.original_size = len(self.data)
        if intersect_indices is not None:
            self.data = self.data[intersect_indices]
            self.targets = self.targets[intersect_indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        target = self.targets[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def get_batch(self, sample_ids):
        images = []
        targets = []
        for sample_id in sample_ids:
            image, target = self[sample_id]
            images.append(image)
            targets.append(torch.tensor(target, dtype=torch.long))
        image_batch = torch.stack(images, dim=0)
        target_batch = torch.stack(targets, dim=0)
        if self.returns == "image":
            return image_batch
        if self.returns == "label":
            return target_batch
        raise ValueError(f"returns must be 'image' or 'label', but was {self.returns!r}")


class SplitNNClient:
    """Become the image or label party based on the Collab site name."""

    def __init__(
        self,
        dataset_root: str,
        batch_size: int,
        learning_rate: float,
        intersection_file: str,
        fp16: bool,
        seed: int,
        download: bool,
    ):
        self.dataset_root = dataset_root
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.intersection_file = intersection_file
        self.fp16 = fp16
        self.seed = seed
        self.download = download

        self.role = None
        self.device = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_dataset = None
        self.valid_dataset = None
        self.train_size = 0
        self._activations = None
        self._validation_losses = []
        self._validation_labels = []
        self._validation_predictions = []
        self._rng_state = None

    @collab.init
    def initialize(self):
        if collab.site_name == IMAGE_SITE:
            self.role = "image"
            split_id = 0
        elif collab.site_name == LABEL_SITE:
            self.role = "label"
            split_id = 1
        else:
            raise RuntimeError(f"this example only supports {IMAGE_SITE} and {LABEL_SITE}, not {collab.site_name}")

        # Local-process Collab clients share global RNGs, while the original
        # sites run in separate processes. Capture an independent state for
        # each client so direct calls reproduce the original random stream.
        with _RNG_LOCK:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = SplitNN(split_id=split_id, seed=self.seed).to(self.device)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
            self.criterion = nn.CrossEntropyLoss()
            self._rng_state = self._capture_rng_state()

        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Pad(4, padding_mode="reflect"),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[value / 255.0 for value in (125.3, 123.0, 113.9)],
                    std=[value / 255.0 for value in (63.0, 62.1, 66.7)],
                ),
            ]
        )
        valid_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[value / 255.0 for value in (125.3, 123.0, 113.9)],
                    std=[value / 255.0 for value in (63.0, 62.1, 66.7)],
                ),
            ]
        )
        intersect_indices = np.loadtxt(self.intersection_file) if self.intersection_file else None
        returns = "image" if self.role == "image" else "label"
        self.train_dataset = CIFAR10SplitDataset(
            root=self.dataset_root,
            train=True,
            transform=train_transform,
            download=self.download,
            returns=returns,
            intersect_indices=intersect_indices,
        )
        self.valid_dataset = CIFAR10SplitDataset(
            root=self.dataset_root,
            train=False,
            transform=valid_transform,
            download=False,
            returns=returns,
        )
        self.train_size = len(self.train_dataset)
        if self.train_size <= 0:
            raise ValueError("training dataset must contain at least one sample")
        print(
            f"{collab.site_name}: initialized split_id={split_id} on {self.device}; "
            f"training with {self.train_size}/{self.train_dataset.original_size} samples"
        )

    def _require_role(self, expected):
        if self.role != expected:
            raise RuntimeError(
                f"{collab.site_name} is the {self.role} site; this function requires the {expected} site"
            )

    @staticmethod
    def _capture_rng_state():
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state().clone(),
            "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
        }

    @staticmethod
    def _restore_rng_state(state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state["cuda"]:
            torch.cuda.set_rng_state_all(state["cuda"])

    @contextmanager
    def _site_rng(self):
        """Give this client the process-local RNG stream of the original site."""
        if self._rng_state is None:
            raise RuntimeError("client RNG state has not been initialized")
        with _RNG_LOCK:
            ambient_state = self._capture_rng_state()
            self._restore_rng_state(self._rng_state)
            try:
                yield
            finally:
                self._rng_state = self._capture_rng_state()
                self._restore_rng_state(ambient_state)

    @collab.publish
    def load_model(self, weights):
        """Load the same full initial model on both split sites."""
        local_weights = self.model.state_dict()
        loaded = 0
        for name in local_weights:
            if name in weights:
                local_weights[name] = torch.as_tensor(np.reshape(weights[name], local_weights[name].shape))
                loaded += 1
        if loaded == 0:
            raise ValueError("no initial model weights were loaded")
        self.model.load_state_dict(local_weights)

    @collab.publish
    def backward(self, gradient):
        """Apply the previous round's cut-layer gradient on the image site."""
        self._require_role("image")
        if self._activations is None:
            raise RuntimeError("forward() must be called before backward()")
        self.model.train()
        self.optimizer.zero_grad()
        if self.fp16:
            gradient = gradient.type(torch.float32)
        gradient = gradient.to(self.device)
        self._activations.backward(gradient=gradient.reshape(self._activations.shape))
        self.optimizer.step()
        self._activations = None

    @collab.publish
    def forward(self):
        """Sample a minibatch and return flattened cut-layer activations."""
        self._require_role("image")
        with self._site_rng():
            self.model.train()
            sample_ids = np.random.randint(0, self.train_size - 1, self.batch_size)
            images = self.train_dataset.get_batch(sample_ids).to(self.device)
            self._activations = self.model(images)
            transferred = self._activations.detach().requires_grad_().flatten(start_dim=1, end_dim=-1)
        return sample_ids, transferred.type(torch.float16) if self.fp16 else transferred

    @collab.publish
    def compute_gradient(self, sample_ids, activations):
        """Update the label-side model and return its cut-layer gradient."""
        self._require_role("label")
        with self._site_rng():
            self.model.train()
            self.optimizer.zero_grad()
            labels = self.train_dataset.get_batch(sample_ids).to(self.device)
            # A real Collab transport round-trip, like the original FOBS exchange,
            # returns a tensor detached from the image-side computation graph.
            activations = activations.detach()
            if self.fp16:
                activations = activations.type(torch.float32)
            activations = activations.to(self.device)
            activations.requires_grad_(True)

            predictions = self.model(activations)
            loss = self.criterion(predictions, labels)
            loss.backward()
            predicted_labels = torch.max(predictions, 1)[1]
            accuracy = (predicted_labels == labels).sum() / len(labels)
            self.optimizer.step()

        if not isinstance(activations.grad, torch.Tensor):
            raise ValueError("no activation gradient was produced")
        gradient = activations.grad.type(torch.float16) if self.fp16 else activations.grad
        return gradient.detach(), loss.item(), accuracy.item()

    @collab.publish
    def validation_size(self):
        self._require_role("image")
        return len(self.valid_dataset)

    @collab.publish
    def validation_forward(self, sample_ids):
        self._require_role("image")
        self.model.eval()
        with torch.no_grad():
            images = self.valid_dataset.get_batch(sample_ids).to(self.device)
            activations = self.model(images).detach().flatten(start_dim=1, end_dim=-1)
        # Unlike training activations, the original sends validation
        # activations at their native FP32 precision.
        return activations

    @collab.publish
    def validation_step(self, sample_ids, activations, first_batch, last_batch):
        """Accumulate validation exactly as the original label-side learner."""
        self._require_role("label")
        if first_batch:
            self._validation_losses = []
            self._validation_labels = []
            self._validation_predictions = []

        self.model.eval()
        with torch.no_grad():
            labels = self.valid_dataset.get_batch(sample_ids).to(self.device)
            if self.fp16:
                activations = activations.type(torch.float32)
            predictions = self.model(activations.to(self.device))
            self._validation_losses.append(self.criterion(predictions, labels).unsqueeze(0))
            self._validation_predictions.append(torch.max(predictions, 1)[1])
            self._validation_labels.append(labels)

        if not last_batch:
            return None
        loss = torch.mean(torch.cat(self._validation_losses))
        predicted_labels = torch.cat(self._validation_predictions)
        labels = torch.cat(self._validation_labels)
        accuracy = (predicted_labels == labels).sum() / len(labels)
        return loss.item(), accuracy.item()


class SplitLearning:
    def __init__(
        self,
        num_rounds: int,
        batch_size: int,
        validation_frequency: int,
        log_every: int,
        call_timeout: float,
        seed: int,
    ):
        self.num_rounds = num_rounds
        self.batch_size = batch_size
        self.validation_frequency = validation_frequency
        self.log_every = log_every
        self.call_timeout = call_timeout
        self.seed = seed

        initial_model = ModerateCNN(seed=seed)
        self.initial_weights = {
            name: value.detach().cpu().numpy().copy() for name, value in initial_model.state_dict().items()
        }

    def _validate(self, image_site, label_site):
        indices = np.arange(image_site.validation_size())
        num_batches = int(np.ceil(len(indices) / self.batch_size))
        result = None
        for batch_number, sample_ids in enumerate(np.array_split(indices, num_batches)):
            activations = image_site.validation_forward(sample_ids)
            result = label_site.validation_step(
                sample_ids,
                activations,
                first_batch=batch_number == 0,
                last_batch=batch_number == num_batches - 1,
            )
        return result

    @collab.main
    def run(self):
        if len(collab.clients) != 2:
            raise RuntimeError(f"split learning requires exactly two clients, but found {len(collab.clients)}")

        image_proxy, label_proxy = collab.get_clients([IMAGE_SITE, LABEL_SITE])
        image_site = image_proxy(timeout=self.call_timeout)
        label_site = label_proxy(timeout=self.call_timeout)
        image_site.load_model(self.initial_weights)
        label_site.load_model(self.initial_weights)

        print("Training: site-1 has images; site-2 has labels")
        gradient = None
        train_loss = None
        train_accuracy = None
        validation_loss = None
        validation_accuracy = None
        for current_round in range(self.num_rounds):
            # The original learner applies the preceding round's gradient
            # immediately before the next forward pass.
            if gradient is not None:
                image_site.backward(gradient)
            sample_ids, activations = image_site.forward()
            gradient, train_loss, train_accuracy = label_site.compute_gradient(sample_ids, activations)

            if current_round % self.log_every == 0:
                print(
                    f"round {current_round:>5}/{self.num_rounds}: "
                    f"loss={train_loss:.4f}, accuracy={train_accuracy:.2%}"
                )
            if self.validation_frequency > 0 and current_round % self.validation_frequency == 0:
                validation_loss, validation_accuracy = self._validate(image_site, label_site)
                print(
                    f"round {current_round:>5}/{self.num_rounds} validation: "
                    f"loss={validation_loss:.4f}, accuracy={validation_accuracy:.2%}"
                )

        # Intentionally do not apply the final gradient: this is how the
        # original learner ends its delayed image-side update sequence.
        return {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }


def make_recipe(args):
    # Recipe construction is only needed by the launcher, not by the exported
    # server/client module at runtime.
    from nvflare.collab import CollabRecipe

    return CollabRecipe(
        job_name="collab_split_learning",
        server=SplitLearning(
            num_rounds=args.num_rounds,
            batch_size=args.batch_size,
            validation_frequency=args.validation_frequency,
            log_every=args.log_every,
            call_timeout=args.call_timeout,
            seed=args.seed,
        ),
        client=SplitNNClient(
            dataset_root=args.dataset_root,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            intersection_file=args.intersection_file,
            fp16=not args.no_fp16,
            seed=args.seed,
            download=not args.no_download,
        ),
        min_clients=2,
        sync_task_timeout=args.call_timeout,
    )


def main():
    # Keep the examples-only CLI helper out of the exported job's imports.
    from collab.common.runner import make_parser, run_recipe

    parser = make_parser("CIFAR-10 SplitNN equivalent using direct Collab calls")
    parser.add_argument("--dataset-root", default="/tmp/cifar10", help="CIFAR-10 data directory on each site")
    parser.add_argument("--intersection-file", help="optional aligned training sample IDs, as produced by PSI")
    parser.add_argument("--num-rounds", type=int, default=15625, help="number of split-learning minibatches")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--validation-frequency", type=int, default=1000, help="round interval; <= 0 disables it")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--call-timeout", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-fp16", action="store_true", help="exchange FP32 activations and gradients")
    parser.add_argument("--no-download", action="store_true", help="require CIFAR-10 to already exist")
    parser.set_defaults(num_clients=2)
    args = parser.parse_args()

    if args.num_clients != 2:
        parser.error("this split-learning example requires --num-clients 2")
    if args.num_rounds <= 0:
        parser.error("--num-rounds must be greater than zero")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be greater than zero")
    if args.log_every <= 0:
        parser.error("--log-every must be greater than zero")
    if args.call_timeout <= 0:
        parser.error("--call-timeout must be greater than zero")

    run_recipe(make_recipe(args), args)


if __name__ == "__main__":
    main()
