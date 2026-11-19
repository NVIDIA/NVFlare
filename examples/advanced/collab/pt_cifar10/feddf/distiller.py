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

"""Bind backend configuration without exposing backend internals to the server."""

from functools import partial

import torch

from nvflare.app_opt.pt.distillation import DistillationTrainer, kl_divergence


class Cifar10DistillerFactory:
    """Exportable factory: each call returns one current-round KD session.

    The server supplies an already-averaged student and ONE mean-logit ensemble
    teacher. All backends use pure temperature-scaled batchmean KL and Adam.
    Optional libraries are imported only when their backend is actually selected.
    """

    def __init__(self, backend="native", kd_lr=0.001, temperature=1.0):
        if backend not in ("native", "modelopt", "torchdistill"):
            raise ValueError("distiller must be native, modelopt or torchdistill")
        self.backend = backend
        self.kd_lr = kd_lr
        self.temperature = temperature

    def __call__(self, student, teacher):
        common = {"device": next(student.parameters()).device, "batch_adapter": lambda batch: batch[0]}
        optimizer = partial(torch.optim.Adam, lr=self.kd_lr)
        if self.backend == "native":
            return DistillationTrainer(
                student=student,
                teachers=[teacher],
                optimizer_factory=optimizer,
                loss_fn=partial(kl_divergence, temperature=self.temperature),
                **common,
            )
        from nvflare.app_opt.pt.distillation.backends import ModelOptDistiller, TorchDistillDistiller

        if self.backend == "modelopt":
            return ModelOptDistiller(
                student,
                teacher,
                optimizer,
                temperature=self.temperature,
                **common,
            )
        return TorchDistillDistiller(
            student,
            teacher,
            train_config={
                "criterion": {
                    "key": "WeightedSumLoss",
                    "kwargs": {
                        "sub_terms": {
                            "kd": {
                                "criterion": {
                                    "key": "KDLoss",
                                    "kwargs": {
                                        "student_module_path": ".",
                                        "student_module_io": "output",
                                        "teacher_module_path": ".",
                                        "teacher_module_io": "output",
                                        "temperature": self.temperature,
                                        "alpha": 0.0,
                                        "reduction": "batchmean",
                                    },
                                },
                                "weight": self.temperature**2,
                            },
                        }
                    },
                },
                "optimizer": {"key": "Adam", "kwargs": {"lr": self.kd_lr}},
            },
            **common,
        )
