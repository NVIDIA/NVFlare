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

"""Differentiable feature-statistics matching without a required tracker class."""

from contextlib import ExitStack

import torch
from torch import nn


def feature_statistics_loss(current, reference):
    """Match sequences of (mean, variance) tensors using summed L2 distances.

    Reference statistics are detached. Current statistics must remain attached
    to synthetic inputs. Callers choose the layers, statistics and reduction
    axes; this function makes no image, BatchNorm or classifier assumption.
    """
    current, reference = list(current), list(reference)
    if not current or len(current) != len(reference):
        raise ValueError("current/reference statistics must be nonempty and have the same length")
    losses = []
    for (mean, variance), (ref_mean, ref_variance) in zip(current, reference):
        if mean.shape != ref_mean.shape or variance.shape != ref_variance.shape:
            raise ValueError("feature statistic shapes do not match")
        losses.append(
            (mean - ref_mean.detach().to(mean)).norm(2) + (variance - ref_variance.detach().to(variance)).norm(2)
        )
    return torch.stack(losses).sum()


def forward_with_feature_statistics(model, inputs, layers=None, reference=None):
    """Return (model_output, statistics_loss) with temporary input-feature hooks.

    Defaults select BatchNorm1d/2d/3d layers and their running statistics.
    Supply layers/reference for other tracker types. Each selected layer must
    receive one batched tensor whose channel dimension is 1. Means/variances
    reduce batch and spatial dimensions. For other layouts, supply your own
    feature adapter and use feature_statistics_loss directly.

    The model must already be in eval mode, with any teacher parameters frozen.
    Hooks are removed on success and failure; model modes/buffers are untouched.
    """
    layers = (
        [layer for layer in model.modules() if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
        if layers is None
        else list(layers)
    )
    if not layers:
        raise ValueError("feature statistics require selected layers or a custom feature adapter")
    if reference is None:
        reference = [(layer.running_mean, layer.running_var) for layer in layers]
    reference = list(reference)
    if len(reference) != len(layers) or any(
        not isinstance(value, torch.Tensor) for pair in reference for value in pair
    ):
        raise ValueError("each selected layer requires reference mean/variance tensors")
    current = [None] * len(layers)

    def capture(index, args):
        features = args[0]
        dims = (0, *range(2, features.ndim))
        current[index] = (features.mean(dim=dims), features.var(dim=dims, unbiased=False))

    with ExitStack() as cleanup:
        for index, layer in enumerate(layers):
            handle = layer.register_forward_pre_hook(lambda module, args, index=index: capture(index, args))
            cleanup.callback(handle.remove)
        output = model(inputs)
        if any(value is None for value in current):
            raise ValueError("a selected feature layer did not execute")
        loss = feature_statistics_loss(current, reference)
    return output, loss
