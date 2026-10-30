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

"""Data-free input synthesis with replaceable task-specific components."""

from copy import deepcopy

import torch
from torch import nn

from .classification import ClassificationObjective
from .image import ImageGenerator, ImageTransform
from .teachers import _snapshot_state
from .trainer import _model


class DataFreeSynthesizer:
    """Optimize a generator through frozen models and return model-ready inputs.

    Defaults are an ImageGenerator producing RGB 32x32 images, confidence plus
    diversity ClassificationObjective, Adam(lr=3e-3, betas=(0.5, 0.999)), and
    ImageTransform crop/flip. input_shape configures the default generator;
    no dataset normalization is assumed.

    Replace generator, objective, optimizer_factory and transform for another
    task. generator is a module/factory mapping a latent tensor to a batched
    input tensor; it is moved/trained in place. latent_sampler(batch_size, rng)
    may replace CPU Gaussian noise of shape [batch_size, latent_size].
    objective(inputs, teachers, student, targets) returns a scalar to MINIMIZE.

    Teacher/student modules or factory results are copied, frozen and put in
    eval mode. Their forwards remain differentiable with respect to inputs.
    Only generator parameter gradients are accumulated. Custom callables own
    their configuration; module transforms are moved to device in place.

    rng is an optional caller-owned CPU generator shared by default noise and
    augmentation, independent of the process default device. Noise is moved
    to device for model execution. Global/model and custom-callable RNG remain
    caller-owned too.
    Basic KD imports do not import this optional module.

    Example::

        from nvflare.app_opt.pt.distillation.synthesis import DataFreeSynthesizer
        from nvflare.app_opt.pt.distillation.pool import SyntheticPool

        synthesis = DataFreeSynthesizer([teacher], input_shape=(1, 16, 16))
        inputs, metrics = synthesis.synthesize(batch_size=4, steps=2)
        pool = SyntheticPool(max_samples=128)
        pool.add(inputs)
        # Samples already include the synthesis transform/normalization.
        trainer.fit([pool.sample(4)[0]])
    """

    def __init__(
        self,
        teachers,
        generator=None,
        objective=None,
        optimizer_factory=None,
        transform=None,
        student=None,
        device="cpu",
        input_shape=(3, 32, 32),
        latent_size=128,
        latent_sampler=None,
        rng=None,
    ):
        teachers = list(teachers)
        if not teachers:
            raise ValueError("at least one teacher is required")
        self.device = torch.device(device)
        self.teachers = [deepcopy(_model(value)).to(self.device).eval().requires_grad_(False) for value in teachers]
        self.student = (
            None if student is None else deepcopy(_model(student)).to(self.device).eval().requires_grad_(False)
        )
        self.generator = (ImageGenerator(latent_size, input_shape) if generator is None else _model(generator)).to(
            self.device
        )
        self.parameters = [value for value in self.generator.parameters() if value.requires_grad]
        if not self.parameters:
            raise ValueError("generator must have trainable parameters")
        self.objective = ClassificationObjective() if objective is None else objective
        self.transform = ImageTransform(generator=rng) if transform is None else transform
        if isinstance(self.transform, nn.Module):
            self.transform = self.transform.to(self.device)
        self.optimizer = (
            torch.optim.Adam(self.parameters, lr=3e-3, betas=(0.5, 0.999))
            if optimizer_factory is None
            else optimizer_factory(self.parameters)
        )
        if not isinstance(self.optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer_factory must return a torch optimizer")
        self.latent_size, self.latent_sampler, self.rng = latent_size, latent_sampler, rng
        self.num_steps = 0

    def synthesize(self, batch_size=64, steps=20, targets=None):
        """Return (best_transformed_CPU_inputs, loss/steps metrics).

        One latent batch and optional targets are fixed across the optimization
        steps. targets may have any task-appropriate shape with leading
        batch_size. The returned inputs were evaluated by the objective BEFORE
        the corresponding optimizer step, with transforms already applied.
        Empty/invalid/non-finite/disconnected objectives raise; completed steps
        remain applied on a later failure. There is no implicit student update.
        """
        for name, value in (("batch_size", batch_size), ("steps", steps)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        noise = (
            (
                torch.randn(batch_size, self.latent_size, generator=self.rng, device="cpu")
                if self.latent_sampler is None
                else self.latent_sampler(batch_size, self.rng)
            )
            .detach()
            .to(self.device)
        )
        if targets is not None:
            if not isinstance(targets, torch.Tensor) or targets.ndim < 1 or len(targets) != batch_size:
                raise ValueError("targets must have leading dimension batch_size")
            targets = targets.detach().to(self.device)
        best, best_loss = None, float("inf")
        self.generator.train()
        for _ in range(steps):
            inputs = self.transform(self.generator(noise))
            if not isinstance(inputs, torch.Tensor) or inputs.ndim < 1 or len(inputs) != batch_size:
                raise ValueError("generator/transform must produce batch_size input tensors")
            loss = self.objective(inputs, self.teachers, self.student, targets)
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError("synthesis objective must return a finite scalar")
            if not loss.requires_grad:
                raise ValueError("synthesis objective must depend on generator parameters")
            cost = loss.detach().item()
            if cost < best_loss:
                best, best_loss = inputs.detach().cpu().clone(), cost
            gradients = torch.autograd.grad(loss, self.parameters, allow_unused=True)
            if all(value is None for value in gradients):
                raise ValueError("synthesis objective must depend on generator parameters")
            self.optimizer.zero_grad(set_to_none=True)
            for parameter, gradient in zip(self.parameters, gradients):
                parameter.grad = gradient
            self.optimizer.step()
            self.num_steps += 1
        return best, {"loss": best_loss, "steps": steps}

    def state_dict(self):
        """Copy model/optimizer/step state; RNG, objectives and factories stay external.

        Transforms that are modules also export their tensor state. Reconstruct
        matching models, objective, transforms, latent sampler and optimizer
        before loading, then restore caller-owned RNG. This is local component
        restoration, not distributed workflow recovery.
        """
        return {
            "generator": _snapshot_state(self.generator.state_dict()),
            "teachers": [_snapshot_state(model.state_dict()) for model in self.teachers],
            "student": None if self.student is None else _snapshot_state(self.student.state_dict()),
            "transform": (
                _snapshot_state(self.transform.state_dict()) if isinstance(self.transform, nn.Module) else None
            ),
            "optimizer": deepcopy(self.optimizer.state_dict()),
            "num_steps": self.num_steps,
        }

    def load_state_dict(self, state):
        """Restore matching components; restore caller-owned RNG after loading."""
        if len(state["teachers"]) != len(self.teachers) or (state["student"] is None) != (self.student is None):
            raise ValueError("checkpoint teacher/student configuration does not match")
        if (state["transform"] is None) != (not isinstance(self.transform, nn.Module)):
            raise ValueError("checkpoint transform configuration does not match")
        count = state["num_steps"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("checkpoint num_steps must be a nonnegative integer")
        self.generator.load_state_dict(state["generator"])
        for model, teacher_state in zip(self.teachers, state["teachers"]):
            model.load_state_dict(teacher_state)
        if self.student is not None:
            self.student.load_state_dict(state["student"])
        if isinstance(self.transform, nn.Module):
            self.transform.load_state_dict(state["transform"])
        self.optimizer.load_state_dict(deepcopy(state["optimizer"]))
        self.num_steps = count
