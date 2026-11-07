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

"""Optional supplied-data distillers; external packages are imported on construction.

ModelOptDistiller uses modelopt's wrapped model, feature hooks and KD loss.
TorchDistillDistiller uses torchdistill's single-stage box, optimizer and training
callbacks. Neither reimplements a third-party loss or installs a dependency.
Tested versions: nvidia-modelopt 0.48.0 and torchdistill 1.1.5. Install the selected
package in the same environment as NVFlare; basic KD imports require neither.

Both adapters accept ONE teacher module/factory. For mean-logit KD supply an
ensemble module whose forward averages logits. Multiple independent teacher
losses must not silently become mean-logit KD. Bind constructor arguments with
functools.partial when injecting a distiller_factory into an application.
Backend configuration, including feature objectives, is caller-owned policy.
These adapters target FP32, one optimizer update per batch, same inputs for
student/teacher and caller-controlled training passes. Synthesis is separate.
"""

import importlib
import math
from copy import deepcopy

import torch
from torch import nn

from .teachers import _snapshot_state
from .trainer import _model


def _backend(module, package):
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"This distiller requires {package} and its dependencies; install it before selection"
        ) from exc


def _finite_loss(loss):
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError("Backend loss must be a finite scalar batch-mean loss")


class _Session:
    def __init__(self, student, teacher, device, batch_adapter):
        self._closed = False
        self._device = torch.device(device)
        self._batch_adapter = batch_adapter
        # Wrapping, freezing and backend cleanup must never mutate caller models.
        self._student = deepcopy(_model(student)).to(self._device)
        self._teacher = deepcopy(_model(teacher)).to(self._device).eval().requires_grad_(False)

    def _check_open(self):
        if self._closed:
            raise RuntimeError("Distiller is closed")

    def export_student_state(self) -> dict:
        """Snapshot only original student tensors, without dismantling backend hooks."""
        self._check_open()
        return _snapshot_state(self._student.state_dict())

    def fit(self, batches) -> dict:
        """One pass; steps counts optimizer updates, samples counts input rows.

        Empty/nonfinite batches fail. Prior completed updates are retained on
        failure; close in the caller's finally block to release backend resources.
        Checkpoints/gradient accumulation are outside these adapters' contract.
        """
        self._check_open()
        self._start_pass()
        total_loss, samples, steps = 0.0, 0, 0
        for batch in batches:
            inputs = self._batch_adapter(batch) if self._batch_adapter else batch
            if not isinstance(inputs, torch.Tensor) or inputs.ndim < 1 or not inputs.shape[0]:
                raise ValueError("batch_adapter must return a nonempty batched input tensor")
            inputs = inputs.detach().to(self._device)
            value = self._step(inputs)
            total_loss += value * inputs.shape[0]
            samples += inputs.shape[0]
            steps += 1
        self._finish_pass()
        if not steps:
            raise ValueError("no batches were supplied")
        return {"loss": total_loss / samples, "steps": steps, "samples": samples}

    def _finish_pass(self):
        pass


class _StudentShell(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        # ModelOpt registers teacher/loss modules on this shell, not inside a
        # Sequential student's forward iteration or original state dictionary.
        return self.model(*args, **kwargs)


class ModelOptDistiller(_Session):
    """Optional modelopt KD with caller-owned models preserved.

    optimizer_factory receives only trainable student/loss parameters. The
    default objective is pure T**2 * KL(teacher || student), batchmean reduction.
    criterion may instead be a modelopt loss or mapping from (student layer,
    teacher layer) names to losses. Student layer names are relative to the
    ORIGINAL student; the adapter prefixes its private shell automatically.
    A caller-supplied loss_balancer controls custom loss combinations; its default
    gives all weight to KD. These objects are copied before backend registration.

    export_student_state never calls modelopt.export: snapshots can be taken
    between fit calls. close performs final backend export to remove its hooks,
    then releases owned references. Backend optimizer/checkpoint restoration is
    not exposed by this adapter. No modelopt dependency is loaded until selected.

    Example (models, optimizer factory and batches are supplied by the caller)::

        session = ModelOptDistiller(student, teacher, optimizer_factory, temperature=2.0)
        try:
            metrics = session.fit(batches)
            student.load_state_dict(session.export_student_state())
        finally:
            session.close()
    """

    def __init__(
        self,
        student,
        teacher,
        optimizer_factory,
        *,
        temperature=1.0,
        criterion=None,
        loss_balancer=None,
        device="cpu",
        batch_adapter=None,
    ):
        if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self._mtd = _backend("modelopt.torch.distill", "nvidia-modelopt")
        super().__init__(student, teacher, device, batch_adapter)
        self._session = _StudentShell(self._student)
        self._optimizer = None
        losses = self._mtd.LogitsDistillationLoss(temperature=temperature, reduction="batchmean")
        if criterion is not None:
            losses = deepcopy(criterion)
        if isinstance(losses, nn.Module):
            losses = {("model", ""): losses}
        else:
            losses = {(f"model.{s}" if s else "model", t): loss for (s, t), loss in losses.items()}
        for loss in losses.values():
            if isinstance(loss, nn.Module):
                loss.to(self._device)
        balancer = (
            self._mtd.StaticLossBalancer(kd_loss_weight=1.0) if loss_balancer is None else deepcopy(loss_balancer)
        )
        try:
            self._session = self._mtd.convert(
                self._session,
                mode=[
                    (
                        "kd_loss",
                        {
                            "teacher_model": self._teacher,
                            "criterion": losses,
                            "loss_balancer": balancer,
                            "expose_minimal_state_dict": True,
                        },
                    )
                ],
            )
            self._session.to(self._device)
            self._optimizer = optimizer_factory(p for p in self._session.parameters() if p.requires_grad)
            if not isinstance(self._optimizer, torch.optim.Optimizer):
                raise TypeError("optimizer_factory must return a torch optimizer")
        except BaseException:
            self.close()
            raise

    def _start_pass(self):
        self._session.train()

    def _step(self, inputs):
        self._optimizer.zero_grad()
        self._session(inputs)
        loss = self._session.compute_kd_loss()
        _finite_loss(loss)
        value = loss.detach().item()
        loss.backward()
        self._optimizer.step()
        return value

    def close(self) -> None:
        """Idempotently remove modelopt hooks and release owned model/optimizer references."""
        if not self._closed:
            self._closed = True
            try:
                if isinstance(self._session, self._mtd.DistillationModel):
                    self._mtd.export(self._session)
            finally:
                self._session = self._optimizer = self._student = self._teacher = None


class TorchDistillDistiller(_Session):
    """Optional torchdistill single-stage, supplied-batch training.

    train_config is a copied DistillationBox config: criterion and optimizer are
    explicit, num_epochs defaults to 1, and teacher/student hooks and forward_proc
    are supported. Teacher gradients, structural model redesign, checkpoint
    loading, custom process callbacks, schedulers, acceleration and gradient
    accumulation are rejected rather than changing the common session contract.
    The adapter forces teacher eval mode and uses forward_batch_only by default.

    Example pure KL criterion entry (within WeightedSumLoss kwargs/sub_terms)::

        {"kd": {"criterion": {"key": "KDLoss", "kwargs": {
            "student_module_path": ".", "student_module_io": "output",
            "teacher_module_path": ".", "teacher_module_io": "output",
            "temperature": 2.0, "alpha": 0.0, "reduction": "batchmean",
        }}, "weight": 4.0}}

    torchdistill 1.1.5's soft-only KDLoss does not apply T**2; the explicit weight
    above restores that convention. Other criteria define their own scaling.
    Model/state export never unfreezes teachers or removes hooks. close calls
    clean_modules on OWNED copies. Optimizer/checkpoint restoration is not exposed.
    """

    def __init__(self, student, teacher, *, train_config, device="cpu", batch_adapter=None):
        config = deepcopy(train_config)
        allowed = {"num_epochs", "teacher", "student", "criterion", "optimizer"}
        if set(config) - allowed or config.get("num_epochs", 1) != 1:
            raise ValueError("Only a single-stage criterion/optimizer and model hooks are supported")
        for name in ("teacher", "student"):
            options = config.setdefault(name, {})
            if set(options) - {"requires_grad", "forward_proc", "forward_hook"}:
                raise ValueError("Model redesign, wrappers and checkpoint loading are unsupported")
            if name == "teacher" and options.get("requires_grad", False):
                raise ValueError("Teacher must remain frozen")
            if name == "student" and not options.get("requires_grad", True):
                raise ValueError("Student must be trainable")
            options.setdefault("forward_proc", "forward_batch_only")
        config["teacher"]["requires_grad"] = False
        config.setdefault("num_epochs", 1)
        optimizer = config.get("optimizer", {})
        if set(optimizer) - {"key", "kwargs", "filters_params", "max_grad_norm", "grad_accum_step"}:
            raise ValueError("Unsupported optimizer configuration")
        if optimizer.get("grad_accum_step", 1) != 1:
            raise ValueError("Gradient accumulation is unsupported: steps means optimizer updates")
        module = _backend("torchdistill.core.distillation", "torchdistill")
        super().__init__(student, teacher, device, batch_adapter)
        self._box = None
        # A box constructor can register hooks before raising and returning no
        # box. Retain the original hook IDs on our private copies for cleanup.
        self._initial_hooks = [
            (m, set(m._forward_hooks)) for model in (self._student, self._teacher) for m in model.modules()
        ]
        try:
            self._box = module.DistillationBox(
                self._teacher,
                self._student,
                {},
                config,
                self._device,
                [],
                False,
                1.0,
            )
            if not isinstance(self._box.optimizer, torch.optim.Optimizer):
                raise ValueError("train_config must supply an optimizer")
        except BaseException:
            self.close()
            raise

    def _start_pass(self):
        self._teacher.eval()
        self._box.pre_epoch_process(epoch=0)

    def _step(self, inputs):
        loss = self._box.forward_process(inputs, targets=None, supp_dict={})
        _finite_loss(loss)
        value = loss.detach().item()
        self._box.post_forward_process(loss)
        return value

    def _finish_pass(self):
        self._box.post_epoch_process()

    def close(self) -> None:
        """Idempotently remove box hooks and release owned references."""
        if not self._closed:
            self._closed = True
            try:
                if self._box is not None:
                    self._box.clean_modules()
            finally:
                for module, original_ids in self._initial_hooks:
                    for hook_id in set(module._forward_hooks) - original_ids:
                        del module._forward_hooks[hook_id]
                self._initial_hooks = []
                self._box = self._student = self._teacher = None
