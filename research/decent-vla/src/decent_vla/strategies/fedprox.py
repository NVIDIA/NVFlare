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

"""FedProx — adds a proximal term to the local loss.

Reference: Li et al. 2018, "Federated Optimization in Heterogeneous Networks".

The proximal term:
    L_total = L_base(θ) + (μ/2) · (1/N) · Σ_k ‖θ_k − θ_global,k‖²

where N is the total number of trainable elements across all params, is
added before `.backward()`. We use the **mean** of squared deviations
rather than the **sum**.

Why mean and not sum: the canonical FedProx paper's `‖w − w_global‖²`
sums over all parameters and scales linearly with model size, calibrated
for ~1K-param toy models. On a 118M-param LoRA fine-tune of pi05/smolvla
the summed formulation is ~118M× too strong and effectively zeros out
the LoRA contribution (especially LoRA-B matrices that init to zero).
Normalizing by element count makes μ dimensionless and gives it
consistent meaning across architectures.

The strategy caches a copy of the round-start global state in
`ctx.scratch["global_snapshot"]` during `on_round_start`, then reads it
during `modify_loss`.

Sanity property used in tests: as μ → 0, FedProx reduces to FedAvg.
"""
from __future__ import annotations

from decent_vla.core.strategy import ClientContext, StrategyBase
from decent_vla.core.types import LossDict


class FedProx(StrategyBase):
    name = "fedprox"

    def __init__(self, cfg=None, mu: float | None = None, normalize: bool | None = None):
        self.cfg = cfg
        # Resolve μ from cfg if not passed explicitly.
        if mu is not None:
            self.mu = float(mu)
        elif cfg is not None and getattr(cfg, "mu", None) is not None:
            self.mu = float(cfg.mu)
        else:
            self.mu = 0.01
        if self.mu == 0.0:
            import logging

            logging.getLogger(__name__).warning(
                "FedProx instantiated with mu=0.0 — the proximal term is disabled "
                "and this run is identical to FedAvg. Set fl.strategy.mu > 0 (the "
                "StrategyConfig default is 0.0)."
            )
        # `normalize=True` (default) divides the squared-diff sum by element count;
        # this is what we want for any non-toy model. Set False only to reproduce
        # paper-canonical sum-formulation results.
        if normalize is not None:
            self.normalize = bool(normalize)
        elif cfg is not None and hasattr(cfg, "extra") and isinstance(cfg.extra, dict):
            self.normalize = bool(cfg.extra.get("normalize", True))
        else:
            self.normalize = True

    def on_round_start(self, ctx: ClientContext) -> None:
        # Snapshot the *trainable, shared* params at round start, on the
        # client's device, so modify_loss can index by name.
        ctx.scratch["global_snapshot"] = {
            k: v.detach().clone() for k, v in ctx.global_state.items()
        }

    def modify_loss(self, loss: LossDict, ctx: ClientContext) -> LossDict:
        if self.mu == 0.0:
            return loss
        snapshot = ctx.scratch.get("global_snapshot")
        if snapshot is None:
            return loss
        # Read the LIVE model parameters — NOT ``backend.get_state("shared")``,
        # which returns ``.detach().clone()`` tensors with no autograd edge.
        # The proximal term must flow gradient back to the trainable params, or
        # ``loss.backward()`` produces the plain FedAvg gradient and FedProx is a
        # silent no-op. The shared-scope keys are exactly the ``requires_grad``
        # parameter names (see lerobot/groot ``get_state``), so we match the
        # round-start snapshot against ``named_parameters`` directly.
        prox_sq_sum = None
        element_count = 0
        for name, theta in ctx.model.named_parameters():
            if not theta.requires_grad or name not in snapshot:
                continue
            theta_g = snapshot[name].to(device=theta.device, dtype=theta.dtype)
            term = (theta - theta_g).pow(2).sum()
            prox_sq_sum = term if prox_sq_sum is None else prox_sq_sum + term
            element_count += theta.numel()
        if element_count == 0 or prox_sq_sum is None:
            return loss
        prox_term = prox_sq_sum / element_count if self.normalize else prox_sq_sum
        added = 0.5 * self.mu * prox_term
        out = dict(loss)
        out["loss"] = loss["loss"] + added.to(loss["loss"].dtype)
        out["prox_term"] = added.detach()
        return out
