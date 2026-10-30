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

"""Optional classification objectives and sampling/probing helpers."""

import math
from copy import deepcopy
from itertools import islice

import torch
from torch.nn import functional as F

from .features import forward_with_feature_statistics
from .trainer import _model


def _outputs(logits):
    logits = list(logits)
    if not logits or any(
        not isinstance(value, torch.Tensor) or value.ndim != 2 or not all(value.shape) for value in logits
    ):
        raise ValueError("classification outputs must be nonempty [batch, classes] tensors")
    if any(value.shape != logits[0].shape for value in logits):
        raise ValueError("classification outputs must have matching batch/classes")
    return logits


def confidence_loss(teacher_logits, targets=None):
    """Mean teacher cross-entropy; absent targets use each teacher's argmax."""
    logits = _outputs(teacher_logits)
    return torch.stack(
        [F.cross_entropy(value, value.detach().argmax(1) if targets is None else targets) for value in logits]
    ).mean()


def diversity_loss(teacher_logits):
    """Negative entropy of mean class probabilities; minimization favors diversity."""
    probabilities = torch.stack([value.softmax(1) for value in _outputs(teacher_logits)]).mean(0).mean(0)
    return (probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()).sum()


def adversarial_disagreement(teacher_logits, student_logits, only_agree=True):
    """Negative KL per sample, optionally limited to classifier agreements.

    Gradients flow through BOTH logits to synthetic inputs. Unlike the KD loss,
    this must not detach teacher logits or run teacher forwards under no_grad.
    Teacher/student parameters should be frozen by the synthesis caller.
    """
    logits = _outputs([*_outputs(teacher_logits), student_logits])
    teachers, student = logits[:-1], logits[-1]
    losses = []
    for teacher in teachers:
        per_sample = F.kl_div(student.log_softmax(1), teacher.log_softmax(1), reduction="none", log_target=True).sum(1)
        if only_agree:
            per_sample = per_sample * student.argmax(1).eq(teacher.argmax(1))
        losses.append(-per_sample.mean())
    return torch.stack(losses).mean()


class ClassificationObjective:
    """Composable image/classification synthesis default.

    Confidence and batch diversity are enabled by default. Feature matching and
    adversarial disagreement are opt-in. feature_adapter(model, inputs) returns
    (raw_output, scalar_feature_loss); its default uses BatchNorm input statistics.
    Supply an adapter for other trackers/layouts. output_adapter extracts aligned
    logits from model outputs. No class count or image dimensions are fixed.
    """

    def __init__(
        self,
        confidence_weight=1.0,
        diversity_weight=1.0,
        feature_weight=0.0,
        adversarial_weight=0.0,
        output_adapter=None,
        feature_adapter=None,
    ):
        weights = (confidence_weight, diversity_weight, feature_weight, adversarial_weight)
        if any(not math.isfinite(value) or value < 0 for value in weights) or not any(weights):
            raise ValueError("objective weights must be finite/nonnegative, with at least one positive")
        self.confidence_weight = confidence_weight
        self.diversity_weight = diversity_weight
        self.feature_weight = feature_weight
        self.adversarial_weight = adversarial_weight
        self.output_adapter = output_adapter
        self.feature_adapter = feature_adapter or forward_with_feature_statistics

    def __call__(self, inputs, teachers, student=None, targets=None):
        outputs, features = [], []
        for teacher in teachers:
            if self.feature_weight:
                output, feature = self.feature_adapter(teacher, inputs)
                features.append(feature)
            else:
                output = teacher(inputs)
            outputs.append(self.output_adapter(output) if self.output_adapter else output)
        loss = inputs.new_zeros(())
        if self.confidence_weight:
            loss = loss + self.confidence_weight * confidence_loss(outputs, targets)
        if self.diversity_weight:
            loss = loss + self.diversity_weight * diversity_loss(outputs)
        if self.feature_weight:
            loss = loss + self.feature_weight * torch.stack(features).mean()
        if self.adversarial_weight:
            if student is None:
                raise ValueError("adversarial synthesis requires a student model")
            output = student(inputs)
            output = self.output_adapter(output) if self.output_adapter else output
            loss = loss + self.adversarial_weight * adversarial_disagreement(outputs, output)
        return loss


def _probabilities(values):
    values = torch.as_tensor(values, device="cpu").detach().float()
    total = values.sum()
    if values.ndim != 1 or not len(values) or not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("class probabilities must be a finite nonnegative vector")
    if not torch.isfinite(total) or total <= 0:
        raise ValueError("class probabilities must have positive finite mass")
    return values / total


def sample_by_class(inputs, targets, batch_size, generator=None, *, probabilities):
    """Pool sampler matching requested class probabilities, with uniform fallback.

    Use functools.partial(sample_by_class, probabilities=...) as a pool sampler.
    Targets must be one-dimensional int64 class IDs. Probability assigned to an
    absent class falls back uniformly over the whole pool, matching the original
    research sampler's marginal distribution. RNG draw order is not preserved.
    """
    probabilities = _probabilities(probabilities)
    if (
        targets is None
        or targets.ndim != 1
        or targets.dtype != torch.long
        or len(targets) != len(inputs)
        or not len(targets)
        or (targets < 0).any()
        or (targets >= len(probabilities)).any()
    ):
        raise ValueError("class sampling requires valid int64 target IDs for every pooled input")
    counts = torch.bincount(targets, minlength=len(probabilities))
    weights = probabilities[targets] / counts[targets] + probabilities[counts == 0].sum() / len(targets)
    return torch.multinomial(weights, batch_size, replacement=True, generator=generator)


def mean_class_probabilities(
    model, batches, temperature=1.0, device="cpu", batch_adapter=None, output_adapter=None, max_batches=None
):
    """Average classifier probabilities over supplied probe inputs, on a copy.

    model accepts a module/factory; batches accepts a tensor or input iterable.
    The caller supplies probe data/randomness. This is a probe-dependent proxy,
    not a guarantee of recovering the original training class proportions.
    The result is a CPU vector with no fixed class count. Original model state,
    mode, gradients and device are untouched. Use FP32 model outputs; FP16/AMP
    support is outside this helper's scope. Arithmetic, including accumulation,
    uses the model output dtype without automatic casting or overflow recovery.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if max_batches is not None and (
        isinstance(max_batches, bool) or not isinstance(max_batches, int) or max_batches < 1
    ):
        raise ValueError("max_batches must be a positive integer or None")
    model = deepcopy(_model(model)).to(device).eval().requires_grad_(False)
    batches = [batches] if isinstance(batches, torch.Tensor) else batches
    batches = batches if max_batches is None else islice(batches, max_batches)
    total, count = None, 0
    with torch.no_grad():
        for batch in batches:
            inputs = batch_adapter(batch) if batch_adapter else batch
            output = model(inputs.detach().to(device))
            output = output_adapter(output) if output_adapter else output
            output = _outputs([output])[0]
            contribution = (output / temperature).softmax(1).sum(0).cpu()
            if total is not None and total.shape != contribution.shape:
                raise ValueError("probe batches produced different class counts")
            total = contribution if total is None else total + contribution
            count += len(output)
    if not count:
        raise ValueError("no probe inputs were supplied")
    return total / count
