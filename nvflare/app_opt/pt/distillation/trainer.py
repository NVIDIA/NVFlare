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

"""A local distillation loop over caller-supplied input batches."""

from copy import deepcopy
from itertools import islice

import torch
from torch import nn

from .losses import kl_divergence
from .teachers import _snapshot_state


def _model(value):
    model = value if isinstance(value, nn.Module) else value()
    if not isinstance(model, nn.Module):
        raise TypeError("model instances/factories must produce torch.nn.Module")
    return model


class DistillationTrainer:
    """Train a student against the mean per-teacher loss.

    student and teachers accept modules or zero-argument factories. The student
    is moved to device and updated in place, retaining its supplied weights;
    factories define initialization explicitly. Teachers are independently copied
    once, moved to device, frozen and put in eval mode. Originals are untouched.

    batch_adapter extracts a single input tensor from each supplied batch.
    Output adapters convert model outputs to aligned [batch, classes] logits.
    Wrap a model for multiple inputs or different per-teacher output conventions.
    loss_fn(student_logits, teacher_logits) must return a scalar batch-mean loss.
    Use functools.partial(kl_divergence, temperature=...) to configure KD, or
    supply another loss. Multiple teachers contribute equally; logits are not
    averaged. No synthetic data, class weighting or optimizer reset is implicit.

    Example::

        from functools import partial
        import torch
        from torch import nn
        from nvflare.app_opt.pt.distillation import DistillationTrainer

        trainer = DistillationTrainer(
            student=nn.Linear(4, 3),
            teachers=[nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))],
            optimizer_factory=partial(torch.optim.SGD, lr=0.1),
        )
        metrics = trainer.fit([torch.randn(8, 4)], max_steps=1)
    """

    def __init__(
        self,
        student,
        teachers,
        optimizer_factory,
        device="cpu",
        batch_adapter=None,
        student_output_adapter=None,
        teacher_output_adapter=None,
        loss_fn=kl_divergence,
    ):
        teachers = list(teachers)
        if not teachers:
            raise ValueError("at least one teacher is required")
        self.device = torch.device(device)
        # Copy before moving/training the student, even if a supplied teacher
        # instance aliases that student.
        self.teachers = [deepcopy(_model(value)).to(self.device).eval().requires_grad_(False) for value in teachers]
        self.student = _model(student).to(self.device)
        self.optimizer = optimizer_factory(self.student.parameters())
        if not isinstance(self.optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer_factory must return a torch optimizer")
        self.batch_adapter = batch_adapter
        self.student_output_adapter = student_output_adapter
        self.teacher_output_adapter = teacher_output_adapter
        self.loss_fn = loss_fn
        self.num_steps = 0

    def fit(self, batches, max_steps: int | None = None) -> dict:
        """Consume one pass, optionally capped, returning loss/steps/samples.

        Input tensors are detached and moved to device. Labels are unused.
        The student runs in train mode and remains there; teacher copies stay
        frozen in eval mode. max_steps never cycles or consumes an extra batch.
        An empty input or non-finite/non-scalar loss raises. Completed optimizer
        steps remain applied if a later batch fails; there is no rollback.
        """
        if max_steps is not None and (isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1):
            raise ValueError("max_steps must be a positive integer or None")
        self.student.train()
        total_loss, samples, steps = 0.0, 0, 0
        selected = batches if max_steps is None else islice(batches, max_steps)
        for batch in selected:
            inputs = self.batch_adapter(batch) if self.batch_adapter else batch
            if not isinstance(inputs, torch.Tensor) or inputs.ndim < 1 or inputs.shape[0] == 0:
                raise ValueError("batch_adapter must return a nonempty batched input tensor")
            inputs = inputs.detach().to(self.device)
            student_output = self.student(inputs)
            if self.student_output_adapter:
                student_output = self.student_output_adapter(student_output)
            loss = None
            for teacher in self.teachers:
                with torch.no_grad():
                    target = teacher(inputs)
                    if self.teacher_output_adapter:
                        target = self.teacher_output_adapter(target)
                term = self.loss_fn(student_output, target)
                loss = term if loss is None else loss + term
            loss = loss / len(self.teachers)
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError("loss_fn must return a finite scalar batch-mean loss")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            count = inputs.shape[0]
            total_loss += loss.detach().item() * count
            samples += count
            steps += 1
            self.num_steps += 1
        if not steps:
            raise ValueError("no batches were supplied")
        return {"loss": total_loss / samples, "steps": steps, "samples": samples}

    def state_dict(self) -> dict:
        """Copy model/optimizer states and step count for explicit checkpointing.

        Model states are CPU copies; optimizer state follows PyTorch's native
        format/devices. Use torch.load(..., map_location=...) when loading files.
        Architecture/factories, adapters, loss configuration and RNG/data-loader
        state are caller-owned and are not serialized.
        """
        return {
            "student": _snapshot_state(self.student.state_dict()),
            "teachers": [_snapshot_state(model.state_dict()) for model in self.teachers],
            "optimizer": deepcopy(self.optimizer.state_dict()),
            "num_steps": self.num_steps,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore into matching architectures, teacher order and configuration.

        Construct models/optimizer first, then restore caller-owned RNG and data
        state after loading. This is component restoration, not workflow recovery.
        """
        if len(state["teachers"]) != len(self.teachers):
            raise ValueError("checkpoint teacher count does not match this trainer")
        count = state["num_steps"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("checkpoint num_steps must be a nonnegative integer")
        self.student.load_state_dict(state["student"])
        for model, teacher_state in zip(self.teachers, state["teachers"]):
            model.load_state_dict(teacher_state)
        self.optimizer.load_state_dict(deepcopy(state["optimizer"]))
        self.num_steps = count
