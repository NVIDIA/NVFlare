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

"""Run local KD from a small MLP teacher into a linear student."""

import argparse
from functools import partial

import torch
from torch import nn

from nvflare.app_opt.pt.distillation import DistillationTrainer, TeacherBuffer, kl_divergence


def teacher_model():
    return nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(args.seed)
    # These supplied proxy inputs belong to the application, not a KD synthesizer.
    inputs = torch.randn(128, 4, generator=generator)
    labels = (inputs @ torch.randn(4, 3, generator=generator)).argmax(dim=1)

    teacher = teacher_model()
    optimizer = torch.optim.Adam(teacher.parameters(), lr=0.02)
    history = TeacherBuffer(max_size=2)
    for step in range(100):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(teacher(inputs), labels)
        loss.backward()
        optimizer.step()
        if step in (49, 99):
            history.add(teacher, metadata={"training_steps": step + 1, "architecture": "mlp"})

    teachers = []
    for snapshot in history:
        model = teacher_model()
        model.load_state_dict(snapshot.model_state)
        teachers.append(model)
    trainer = DistillationTrainer(
        student=lambda: nn.Linear(4, 3),
        teachers=teachers,
        optimizer_factory=partial(torch.optim.Adam, lr=0.03),
        batch_adapter=lambda batch: batch[0],
        loss_fn=partial(kl_divergence, temperature=2),
    )

    def evaluate():
        trainer.student.eval()
        with torch.no_grad():
            return sum(
                trainer.loss_fn(trainer.student(inputs), model(inputs)) for model in trainer.teachers
            ).item() / len(trainer.teachers)

    initial = evaluate()
    metrics = trainer.fit(((inputs, labels) for _ in range(args.steps)))
    final = evaluate()
    print(f"Teacher snapshots: {len(history)}; student: Linear(4, 3)")
    print(f"Mean teacher KL: {initial:.6f} -> {final:.6f}; KD steps: {metrics['steps']}")
    if final >= initial:
        raise RuntimeError("distillation did not reduce the teacher/student KL")


if __name__ == "__main__":
    main()
