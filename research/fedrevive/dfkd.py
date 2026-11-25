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
from collections import Counter
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from fedrevive import ModelState, SynthesisLabelDistribution, TeacherState
from model import StatTracker, create_model, get_model_diff, load_model_params, reset_model_state

from nvflare.app_opt.pt.distillation.classification import mean_class_probabilities
from nvflare.app_opt.pt.distillation.features import feature_statistics_loss
from nvflare.app_opt.pt.distillation.losses import kl_divergence


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
    generation_interval: int = 10
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
    synthesis_label_distribution: str = SynthesisLabelDistribution.UNIFORM.value

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

    The estimator probes each client's first two uploaded models with Gaussian
    inputs, averages the temperature-scaled softmax predictions, and then fixes
    that estimate. It owns one reusable model and only a ten-value running sum
    plus count per logical client, so enabling it does not turn the 1,000-client
    population into 1,000 resident models.
    """

    def __init__(
        self,
        device: torch.device,
        num_uploads: int = 2,
        temperature: float = 0.8,
        batch_size: int = 64,
        forward_batch_size: int = 64,
        seed: int = 10,
    ):
        if num_uploads < 1 or batch_size < 1 or forward_batch_size < 1 or temperature <= 0:
            raise ValueError("num_uploads, batch_size, forward_batch_size, and temperature must be positive")
        self.device = device
        self.num_uploads = int(num_uploads)
        self.temperature = float(temperature)
        self.batch_size = int(batch_size)
        self.forward_batch_size = int(forward_batch_size)
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

            def probes():
                for offset in range(0, self.batch_size, self.forward_batch_size):
                    chunk_size = min(self.forward_batch_size, self.batch_size - offset)
                    yield torch.randn(chunk_size, 3, 32, 32, generator=self._probe_rng, device="cpu")

            upload_proxy = mean_class_probabilities(
                self._model,
                probes(),
                temperature=self.temperature,
                device=self.device,
            )
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
    """Class-conditional generator used by FedRevive."""

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
        # The copied weights replace the CPU initialization. Preserve the RNG used for KD sampling.
        with torch.random.fork_rng(devices=[]):
            result = DFKDGenerator(self.latent_size, self.features, self.num_classes)
        result.load_state_dict(self.state_dict())
        return result.to(next(self.parameters()).device)


class SyntheticPool:
    """Bounded CPU image storage with class-aware KD sampling.

    Clone only new images when appending. Plan KD batches using indices so the
    images are materialized one minibatch at a time.
    """

    def __init__(self, max_images: int, output_dir: str | None = None):
        if isinstance(max_images, bool) or not isinstance(max_images, int) or max_images < 1:
            raise ValueError("max_images must be a positive integer")
        self.max_images = max_images
        self.output_dir = output_dir
        self.images: list[torch.Tensor] = []
        self.labels: list[int] = []
        self.by_label: dict[int, list[int]] = {}
        self.additions = 0

    def __len__(self):
        return len(self.images)

    def add(self, images: torch.Tensor, labels: torch.Tensor):
        if images.ndim < 1 or not len(images) or labels.ndim != 1 or len(images) != len(labels):
            raise ValueError("images and labels must be nonempty batches of the same size")
        if self.images and (images.shape[1:] != self.images[0].shape or images.dtype != self.images[0].dtype):
            raise ValueError("new images must match the pool's shape and dtype")
        new_images = [image.clone() for image in images[-self.max_images :].detach().cpu()]
        new_labels = labels[-self.max_images :].detach().cpu().tolist()
        self.images.extend(new_images)
        self.labels.extend(new_labels)
        excess = len(self.images) - self.max_images
        if excess > 0:
            del self.images[:excess]
            del self.labels[:excess]
        self.by_label.clear()
        for index, label in enumerate(self.labels):
            self.by_label.setdefault(label, []).append(index)
        self.additions += 1
        if self.output_dir and self.additions % 100 == 1:
            os.makedirs(self.output_dir, exist_ok=True)
            torch.save(
                {"images": self.images, "labels": self.labels},
                os.path.join(self.output_dir, "synthetic_dataset.pt"),
            )

    def sample_batch_indices(self, distribution: torch.Tensor, batch_size: int, num_batches: int):
        """Draw labels for all steps, then sample matching images per class."""

        probabilities = distribution.detach().cpu().float()
        probabilities = probabilities / probabilities.sum()
        requested = torch.multinomial(probabilities, batch_size * num_batches, replacement=True).tolist()
        indices = []
        for label, count in Counter(requested).items():
            candidates = self.by_label.get(label)
            if not candidates:
                indices.extend(torch.randint(len(self.images), (count,), device="cpu").tolist())
                continue
            if len(candidates) >= count:
                selected = torch.randperm(len(candidates), device="cpu")[:count]
            else:
                selected = torch.randint(len(candidates), (count,), device="cpu")
            indices.extend(candidates[index] for index in selected.tolist())
        shuffled = torch.tensor(indices, device="cpu")[torch.randperm(len(indices), device="cpu")]
        return shuffled.reshape(num_batches, batch_size).tolist()

    def load_batch(self, indices: list[int]) -> torch.Tensor:
        return torch.stack([self.images[index] for index in indices])


def _kldiv(student_logits, teacher_logits, temperature: float, reduction="batchmean"):
    """Synthesis KL: retain teacher-input gradients through frozen models.

    Ordinary student training uses the shared detached-target KD loss below.
    Synthesis optimizes inputs, so replacing this function with that loss would
    silently remove the teacher's contribution to the input gradient.
    """
    student = F.log_softmax(student_logits / temperature, dim=1)
    teacher = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student, teacher, reduction=reduction) * temperature**2


def _feature_loss(model: nn.Module, batch_features) -> torch.Tensor:
    trackers = [module for module in model.modules() if isinstance(module, StatTracker)]
    if len(trackers) != len(batch_features):
        raise ValueError(f"Expected {len(trackers)} feature-stat pairs, received {len(batch_features)}")
    loss = torch.zeros((), device=next(model.parameters()).device)
    for tracker, (mean, variance) in zip(trackers, batch_features):
        loss = loss + feature_statistics_loss([(mean, variance)], [(tracker.running_mean, tracker.running_var)])
    return loss


class DFKDReviver:
    """Persistent generator and fixed-size reusable models for one server.

    Teacher and student module objects are allocated once, then loaded in place
    on each revival.  Reuse avoids repeatedly constructing CUDA modules and
    allocator pools; their count is bounded by ``teacher_buffer_size`` rather
    than the number of accepted updates.
    """

    def __init__(self, device: torch.device, output_dir: str | None = None, config: DFKDConfig | None = None):
        import kornia.augmentation as K
        import kornia.enhance as KE

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

    def _synthesize(self, teachers, distributions, student):
        started = time.time()
        self.synthesis_count += 1
        batch_size = self.config.synth_batch_size
        noise = torch.randn(batch_size, self.config.latent_size, device=self.device, requires_grad=True)
        distribution_tensor = torch.stack(distributions)
        if self.config.synthesis_label_distribution == SynthesisLabelDistribution.TEACHER_MEAN.value:
            targets = torch.multinomial(distribution_tensor.mean(dim=0), batch_size, replacement=True)
        else:
            targets = torch.randint(distribution_tensor.size(1), (batch_size,), device=self.device)
        embeddings = torch.randn(10, self.config.latent_size, device=self.device, requires_grad=True)

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
            # The generator has its own warmup, measured in synthesis calls.
            if self.config.adversarial_weight > 0 and self.synthesis_count >= self.config.warmup_versions:
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
    ) -> tuple[ModelState | None, dict[str, float]]:
        """Refresh synthetic data when due and return a delta only after KD runs."""

        if not teacher_states:
            raise ValueError("FedRevive requires at least one teacher")
        teachers, distributions = self._load_teachers(teacher_states)
        reset_model_state(self.student_model, reset_tracker_stats=True)
        load_model_params(self.student_model, global_model, target_device=self.device)
        self.student_model.eval()
        for parameter in self.student_model.parameters():
            parameter.requires_grad_(False)

        if self.config.should_generate(current_version):
            metrics = self._synthesize(teachers, distributions, self.student_model)
            metrics["synthesis_performed"] = True
        else:
            # The periodic configuration reuses the bounded synthetic pool
            # between generator updates while still performing KD for every
            # eligible client arrival. No model snapshot is retained for reuse.
            metrics = {"synthesis_performed": False, "synthesis_time": 0.0}
        distilled_update = None
        if current_version > self.config.warmup_versions and len(self.pool):
            for parameter in self.student_model.parameters():
                parameter.requires_grad_(True)
            self.student_model.train()
            optimizer = torch.optim.Adam(self.student_model.parameters(), lr=self.config.kd_lr, weight_decay=1e-5)
            losses = []
            # Plan every KD batch teacher by teacher. Each optimizer step
            # still uses one minibatch from each teacher.
            batches = [
                self.pool.sample_batch_indices(distribution, self.config.kd_batch_size, self.config.kd_iterations)
                for distribution in distributions
            ]
            for step_batches in zip(*batches):
                loss = torch.zeros((), device=self.device)
                samples = 0
                for teacher, indices in zip(teachers, step_batches):
                    images = self.augmentation(self.pool.load_batch(indices).to(self.device))
                    student_output = self.student_model(images.detach())
                    with torch.no_grad():
                        teacher_output = teacher(images)
                    loss = loss + kl_divergence(
                        student_output,
                        teacher_output,
                        self.config.temperature,
                        reduction="sum",
                    )
                    samples += images.size(0)
                loss = self._stabilize(loss / samples, "kd")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            metrics["kd_loss"] = sum(losses) / len(losses)
            distilled_update = get_model_diff(self.student_model, global_model, target_device="cpu")
        else:
            metrics["kd_loss"] = 0.0

        for parameter in self.student_model.parameters():
            parameter.requires_grad_(True)
        return distilled_update, metrics
