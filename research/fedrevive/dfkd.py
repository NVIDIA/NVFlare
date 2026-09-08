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

"""Server-only DFKD implementation for the FedRevive CIFAR-10 experiment."""

import os
import time
from dataclasses import dataclass

import kornia.augmentation as K
import kornia.enhance as KE
import torch
import torch.nn as nn
import torch.nn.functional as F
from fedrevive import ModelState, TeacherState
from model import StatTracker, create_model, get_model_diff, load_model_params, reset_model_state


@dataclass(frozen=True)
class DFKDConfig:
    teacher_buffer_size: int = 8
    max_images: int = 16000
    synth_batch_size: int = 64
    kd_batch_size: int = 32
    kd_lr: float = 1e-4
    temperature: float = 1.0
    generator_lr: float = 3e-3
    latent_lr: float = 1e-3
    generator_steps: int = 2
    generation_interval: int = 1
    warmup_versions: int = 100
    freeze_versions: int = 50
    adversarial_weight: float = 0.1
    feature_weight: float = 0.003
    one_hot_weight: float = 1.0
    class_embedding_weight: float = 0.2
    latent_size: int = 256
    generator_features: int = 128
    kd_iterations: int = 10
    proxy_num_uploads: int = 2
    proxy_temperature: float = 0.8
    proxy_batch_size: int = 64

    def __post_init__(self):
        if self.teacher_buffer_size < 1 or self.generator_steps < 1 or self.generation_interval < 1:
            raise ValueError("teacher_buffer_size, generator_steps, and generation_interval must be positive")
        if self.kd_iterations < 1:
            raise ValueError("kd_iterations must be positive")
        if self.proxy_num_uploads < 1 or self.proxy_temperature <= 0 or self.proxy_batch_size < 1:
            raise ValueError("proxy_num_uploads, proxy_temperature, and proxy_batch_size must be positive")

    def should_generate(self, model_version: int) -> bool:
        return model_version % self.generation_interval == 0


class ClassProportionProxyEstimator:
    """Estimate and freeze per-client class proportions from regular uploads.

    Paper-aligned FedRevive probes each client's first two uploaded models with
    Gaussian inputs, averages the temperature-scaled softmax predictions, and
    then fixes that proxy.  The estimator owns one reusable model and only a
    ten-value running sum plus count per logical client, so enabling it does
    not turn the 1,000-client population into 1,000 resident models.
    """

    def __init__(
        self,
        device: torch.device,
        num_uploads: int = 2,
        temperature: float = 0.8,
        batch_size: int = 64,
        seed: int = 10,
    ):
        if num_uploads < 1 or batch_size < 1 or temperature <= 0:
            raise ValueError("num_uploads, batch_size, and temperature must be positive")
        self.device = device
        self.num_uploads = int(num_uploads)
        self.temperature = float(temperature)
        self.batch_size = int(batch_size)
        self._model = create_model().to(device)
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)
        # Keep probe randomness independent from synthesis and KD randomness.
        # This makes the online estimate reproducible without changing the
        # server's DFKD random stream when a client reaches its second upload.
        self._probe_rng = torch.Generator(device="cpu").manual_seed(seed)
        self._proxy_sums: dict[str, torch.Tensor] = {}
        self._upload_counts: dict[str, int] = {}

    def estimate(self, client_name: str, uploaded_model: ModelState) -> torch.Tensor:
        """Return the running proxy, probing only the first ``num_uploads`` models."""

        count = self._upload_counts.get(client_name, 0)
        if count < self.num_uploads:
            load_model_params(self._model, uploaded_model, target_device=self.device)
            self._model.eval()
            probes = torch.randn(
                self.batch_size,
                3,
                32,
                32,
                generator=self._probe_rng,
                device="cpu",
            ).to(self.device)
            with torch.no_grad():
                logits = self._model(probes)
                upload_proxy = torch.softmax(logits / self.temperature, dim=1).mean(dim=0).cpu()
            self._proxy_sums[client_name] = (
                self._proxy_sums.get(client_name, torch.zeros_like(upload_proxy)) + upload_proxy
            )
            count += 1
            self._upload_counts[client_name] = count

        proxy = self._proxy_sums[client_name] / count
        return proxy / proxy.sum().clamp_min(1e-8)

    def num_observed_uploads(self, client_name: str) -> int:
        return self._upload_counts.get(client_name, 0)


class DFKDGenerator(nn.Module):
    """Class-conditional generator used by the reference implementation."""

    def __init__(self, latent_size: int = 256, features: int = 128, num_classes: int = 10):
        super().__init__()
        self.latent_size = latent_size
        self.features = features
        self.num_classes = num_classes
        self.initial_size = 8
        self.l1 = nn.Sequential(nn.Linear(latent_size, features * 2 * self.initial_size**2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(features * 2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(features * 2, features * 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(features * 2, features, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, 3, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, noise, targets, class_embeddings, class_embedding_weight: float):
        conditioned = noise + class_embedding_weight * class_embeddings[targets]
        output = self.l1(conditioned)
        output = output.view(output.size(0), self.features * 2, self.initial_size, self.initial_size)
        return self.conv_blocks(output)

    def clone(self):
        result = DFKDGenerator(self.latent_size, self.features, self.num_classes)
        result.load_state_dict(self.state_dict())
        return result.to(next(self.parameters()).device)


class SyntheticPool:
    """Bounded in-memory synthetic set with class-aware sampling.

    Synthetic images are genuine FedRevive algorithm state, unlike completed
    RPC payloads waiting only for sequence replay.  They remain resident for KD
    sampling, but ``max_images`` caps their memory independently of run length.
    The optional checkpoint is for observability/recovery and does not create a
    second in-memory pool.
    """

    def __init__(self, max_images: int, output_dir: str | None = None):
        self.max_images = max_images
        self.output_dir = output_dir
        self.images: list[torch.Tensor] = []
        self.labels: list[int] = []
        self.by_label = {label: [] for label in range(10)}
        self.additions = 0

    def __len__(self):
        return len(self.images)

    def _reindex(self):
        self.by_label = {label: [] for label in range(10)}
        for index, label in enumerate(self.labels):
            self.by_label[label].append(index)

    def add(self, images: torch.Tensor, labels: torch.Tensor):
        for image, label in zip(images.detach().cpu(), labels.detach().cpu()):
            self.images.append(image.clone())
            self.labels.append(int(label))
        if len(self.images) > self.max_images:
            excess = len(self.images) - self.max_images
            del self.images[:excess]
            del self.labels[:excess]
        self._reindex()
        self.additions += 1
        if self.output_dir and self.additions % 100 == 1:
            os.makedirs(self.output_dir, exist_ok=True)
            torch.save(
                {"images": self.images, "labels": self.labels},
                os.path.join(self.output_dir, "synthetic_dataset.pt"),
            )

    def sample(self, distribution: torch.Tensor, batch_size: int):
        if not self.images:
            return None, None
        probabilities = distribution.detach().cpu().float()
        probabilities = probabilities / probabilities.sum()
        requested_labels = torch.multinomial(probabilities, batch_size, replacement=True).tolist()
        indices = []
        for label in requested_labels:
            candidates = self.by_label[label]
            if candidates:
                selected = candidates[torch.randint(len(candidates), (1,)).item()]
            else:
                selected = torch.randint(len(self.images), (1,)).item()
            indices.append(selected)
        return (
            torch.stack([self.images[index] for index in indices]),
            torch.tensor([self.labels[index] for index in indices], dtype=torch.long),
        )


def _kldiv(student_logits, teacher_logits, temperature: float, reduction="batchmean"):
    student = F.log_softmax(student_logits / temperature, dim=1)
    teacher = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student, teacher, reduction=reduction) * temperature**2


def _feature_loss(model: nn.Module, batch_features) -> torch.Tensor:
    trackers = [module for module in model.modules() if isinstance(module, StatTracker)]
    if len(trackers) != len(batch_features):
        raise ValueError(f"Expected {len(trackers)} feature-stat pairs, received {len(batch_features)}")
    loss = torch.zeros((), device=next(model.parameters()).device)
    for tracker, (mean, variance) in zip(trackers, batch_features):
        loss = loss + (tracker.running_mean - mean).norm(2) + (tracker.running_var - variance).norm(2)
    return loss


class DFKDReviver:
    """Persistent generator and fixed-size reusable models for one server.

    Teacher and student module objects are allocated once, then loaded in place
    on each revival.  Reuse avoids repeatedly constructing CUDA modules and
    allocator pools; their count is bounded by ``teacher_buffer_size`` rather
    than the number of accepted updates.
    """

    def __init__(self, device: torch.device, output_dir: str | None = None, config: DFKDConfig | None = None):
        self.config = config or DFKDConfig()
        self.device = device
        self.generator = DFKDGenerator(
            latent_size=self.config.latent_size,
            features=self.config.generator_features,
        ).to(device)
        self.generator.train()
        self.meta_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config.generator_lr,
            betas=(0.5, 0.999),
            eps=0,
        )
        self.pool = SyntheticPool(self.config.max_images, output_dir=output_dir)
        self.teacher_models = [create_model().to(device) for _ in range(self.config.teacher_buffer_size)]
        self.student_model = create_model().to(device)
        self.synthesis_count = 0
        self.augmentation = nn.Sequential(
            K.RandomCrop((32, 32), padding=4, padding_mode="reflect"),
            K.RandomHorizontalFlip(),
            KE.Normalize(
                mean=torch.tensor([0.4914, 0.4822, 0.4465]),
                std=torch.tensor([0.2023, 0.1994, 0.2010]),
            ),
        ).to(device)
        self._loss_windows = {name: [] for name in ("feature", "one_hot", "adversarial", "kd")}

    def _stabilize(self, loss: torch.Tensor, name: str) -> torch.Tensor:
        if not torch.isfinite(loss):
            # Multiplying NaN by zero remains NaN, so replace non-finite values
            # while retaining an autograd path with zero gradient at them.
            return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        values = self._loss_windows[name]
        value = loss.item()
        if len(values) >= 10:
            mean = sum(values) / len(values)
            if mean > 0 and value > 3 * mean:
                return loss * 0.0
        values.append(value)
        del values[:-20]
        return loss

    def _load_teachers(self, teachers: list[TeacherState]):
        models = []
        distributions = []
        for index, teacher in enumerate(teachers):
            model = self.teacher_models[index]
            reset_model_state(model, reset_tracker_stats=True)
            load_model_params(model, teacher.model, target_device=self.device)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            models.append(model)
            distributions.append(teacher.class_proxy.to(self.device))
        return models, distributions

    def _synthesize(self, teachers, distributions, student, current_version: int):
        started = time.time()
        self.synthesis_count += 1
        batch_size = self.config.synth_batch_size
        combined_distribution = torch.stack(distributions).mean(dim=0)
        targets = torch.multinomial(combined_distribution, batch_size, replacement=True)
        noise = torch.randn(batch_size, self.config.latent_size, device=self.device, requires_grad=True)
        embeddings = torch.randn(10, self.config.latent_size, device=self.device, requires_grad=True)

        distribution_tensor = torch.stack(distributions)
        teacher_weights = distribution_tensor / distribution_tensor.sum(dim=0, keepdim=True).clamp_min(1e-8)
        sample_weights = teacher_weights[:, targets].T
        fast_generator = self.generator.clone()
        optimizer = torch.optim.Adam(
            [
                {"params": fast_generator.parameters()},
                {"params": [noise], "lr": self.config.latent_lr},
                {"params": [embeddings], "lr": self.config.latent_lr},
            ],
            lr=self.config.generator_lr,
            betas=(0.5, 0.999),
        )

        best_cost = float("inf")
        best_inputs = None
        components = {}
        for _ in range(self.config.generator_steps):
            inputs = fast_generator(noise, targets, embeddings, self.config.class_embedding_weight)
            augmented = self.augmentation(inputs)
            outputs = []
            features = []
            for teacher in teachers:
                output, feature = teacher(augmented, return_features=True)
                outputs.append(output)
                features.append(feature)
            stacked_outputs = torch.stack(outputs)

            feature_loss = torch.zeros((), device=self.device)
            for index, teacher in enumerate(teachers):
                feature_loss = feature_loss + sample_weights[:, index].sum() * _feature_loss(teacher, features[index])
            feature_loss = self._stabilize(feature_loss / batch_size, "feature")

            ce_losses = torch.stack([F.cross_entropy(output, targets, reduction="none") for output in outputs])
            one_hot_loss = self._stabilize((sample_weights * ce_losses.T).sum() / batch_size, "one_hot")

            adversarial_loss = torch.zeros((), device=self.device)
            if self.config.adversarial_weight > 0 and current_version >= self.config.warmup_versions:
                student_output = student(augmented)
                per_teacher = []
                for teacher_output in stacked_outputs:
                    mask = student_output.argmax(1).eq(teacher_output.argmax(1)).float()
                    per_teacher.append(-(_kldiv(student_output, teacher_output, 1.0, reduction="none").sum(1) * mask))
                adversarial_loss = self._stabilize(
                    (sample_weights * torch.stack(per_teacher).T).sum() / batch_size,
                    "adversarial",
                )

            loss = (
                self.config.feature_weight * feature_loss
                + self.config.one_hot_weight * one_hot_loss
                + self.config.adversarial_weight * adversarial_loss
            )
            if loss.item() < best_cost:
                best_cost = loss.item()
                best_inputs = inputs.detach().clone()
            components = {
                "feature_loss": feature_loss.item(),
                "one_hot_loss": one_hot_loss.item(),
                "adversarial_loss": adversarial_loss.item(),
            }
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.meta_optimizer.zero_grad()
        for parameter, fast_parameter in zip(self.generator.parameters(), fast_generator.parameters()):
            parameter.grad = parameter.detach() - fast_parameter.detach()
        self.meta_optimizer.step()
        self.pool.add(best_inputs, targets)
        components.update({"synthesis_loss": best_cost, "synthesis_time": time.time() - started})
        return components

    def revive(
        self,
        global_model: ModelState,
        teacher_states: list[TeacherState],
        current_version: int,
    ) -> tuple[ModelState, dict[str, float]]:
        """Refresh synthetic data when due, then return the per-update student delta."""

        if not teacher_states:
            raise ValueError("FedRevive requires at least one teacher")
        teachers, distributions = self._load_teachers(teacher_states)
        reset_model_state(self.student_model, reset_tracker_stats=True)
        load_model_params(self.student_model, global_model, target_device=self.device)
        self.student_model.eval()
        for parameter in self.student_model.parameters():
            parameter.requires_grad_(False)

        if self.config.should_generate(current_version):
            metrics = self._synthesize(teachers, distributions, self.student_model, current_version)
            metrics["synthesis_performed"] = True
        else:
            # Paper-aligned T_gen=10 reuses the bounded synthetic pool between
            # generator updates while still performing KD for every eligible
            # client arrival.  No model snapshot is retained for this reuse.
            metrics = {"synthesis_performed": False, "synthesis_time": 0.0}
        if current_version > self.config.warmup_versions:
            for parameter in self.student_model.parameters():
                parameter.requires_grad_(True)
            self.student_model.train()
            optimizer = torch.optim.Adam(self.student_model.parameters(), lr=self.config.kd_lr, weight_decay=1e-5)
            losses = []
            for _ in range(self.config.kd_iterations):
                loss = torch.zeros((), device=self.device)
                samples = 0
                for teacher, distribution in zip(teachers, distributions):
                    images, _ = self.pool.sample(distribution, self.config.kd_batch_size)
                    if images is None:
                        continue
                    images = self.augmentation(images.to(self.device))
                    student_output = self.student_model(images.detach())
                    with torch.no_grad():
                        teacher_output = teacher(images)
                    loss = loss + _kldiv(
                        student_output,
                        teacher_output,
                        self.config.temperature,
                        reduction="sum",
                    )
                    samples += images.size(0)
                if samples:
                    loss = self._stabilize(loss / samples, "kd")
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
            metrics["kd_loss"] = sum(losses) / len(losses) if losses else 0.0
        else:
            metrics["kd_loss"] = 0.0

        delta = get_model_diff(self.student_model, global_model, target_device="cpu")
        for parameter in self.student_model.parameters():
            parameter.requires_grad_(True)
        return delta, metrics
