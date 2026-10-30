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

"""Policy-specific local training; FLARE owns federation and server optimization."""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from nvflare.app_opt.pt.fedproxloss import PTFedProxLoss


def seed_training(seed: int, deterministic: bool = False) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


class _SharedParameterView(torch.nn.Module):
    """Expose shared parameters to FLARE's FedProx loss without copying a VLA.

    Reference tensors move to the parameter device lazily, preserving the
    original per-parameter memory use rather than cloning a full GPU model.
    """

    def __init__(self, model, snapshot, reference=False):
        super().__init__()
        self.source = model
        self.snapshot = snapshot
        self.reference = reference

    def named_parameters(self, *args, **kwargs):
        for name, param in self.source.named_parameters():
            if param.requires_grad and name in self.snapshot:
                yield name, self.snapshot[name].to(param) if self.reference else param


def train_steps(backend, dataloader, optimizer, scheduler, global_state, steps, strategy, grad_clip_norm=None):
    """Preserve the original forward/backward/clip/step/schedule order and loss metric.

    FedProx uses live, shared parameters only. The snapshot is the received
    wire-precision global state, not an unquantized server master.
    """
    snapshot = {k: v.detach().clone() for k, v in global_state.items()} if strategy.name == "fedprox" else {}
    if snapshot:
        live = _SharedParameterView(backend.model, snapshot)
        reference = _SharedParameterView(backend.model, snapshot, reference=True)
        count = sum(p.numel() for _, p in live.named_parameters())
        # Unit multiplier obtains the native squared distance. Applying mu
        # after normalization retains the reference's exact arithmetic order.
        criterion = PTFedProxLoss(mu=2.0)
    loss_sum, samples = 0.0, 0
    for step in range(steps):
        try:
            batch = next(dataloader)
        except StopIteration as exc:
            raise ValueError(f"dataloader exhausted after {step} of {steps} planned steps") from exc
        loss = backend.compute_loss(batch)["loss"]
        if snapshot and strategy.mu and count:
            prox = criterion(live, reference)
            if strategy.extra.get("normalize", True):
                prox = prox / count
            loss = loss + (0.5 * strategy.mu * prox).to(loss.dtype)
        loss.backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
        loss_sum += float(loss.detach())
        samples += next((int(v.shape[0]) for v in batch.values() if isinstance(v, torch.Tensor) and v.ndim), 1)
    frames = getattr(backend, "num_train_frames", lambda: None)()
    return loss_sum / steps, frames if frames is not None else samples
