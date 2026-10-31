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

"""FedAdam — Adam-style server update over aggregated client deltas.

Reference: Reddi et al. 2020, "Adaptive Federated Optimization".

Per-parameter server state:
    m_t = β1 m_{t-1} + (1 - β1) Δ_t          (1st moment of aggregated delta)
    v_t = β2 v_{t-1} + (1 - β2) Δ_t²         (2nd moment)
    θ_{t+1} = θ_t + server_lr · m_t / (√v_t + ε)

where Δ_t is the weighted average of client deltas in round t. We override
only `aggregate`; everything else (delta computation, round assembly,
client loop) is inherited from `StrategyBase`.

Precision handling lives in the framework-wide get/set_state code paths,
not in the strategy.
"""
from __future__ import annotations

import torch

from decent_vla.core.strategy import ClientUpdate, StrategyBase


class FedAdam(StrategyBase):
    name = "fedadam"

    def __init__(self, cfg=None, server_lr=None, betas=None, eps=None):
        self.cfg = cfg
        self.server_lr = float(server_lr) if server_lr is not None else (
            float(cfg.server_lr) if cfg is not None else 1e-3
        )
        if betas is not None:
            self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        elif cfg is not None and getattr(cfg, "server_betas", None) is not None:
            self.beta1, self.beta2 = float(cfg.server_betas[0]), float(cfg.server_betas[1])
        else:
            self.beta1, self.beta2 = 0.9, 0.999
        self.eps = float(eps) if eps is not None else (
            float(cfg.server_eps) if cfg is not None else 1e-3
        )
        self._m: dict[str, torch.Tensor] = {}
        self._v: dict[str, torch.Tensor] = {}
        self._t: int = 0

    def state_dict(self) -> dict:
        """Persist moment buffers + step for resume (see StrategyBase.state_dict)."""
        return {
            "t": self._t,
            "m": {k: v.detach().cpu() for k, v in self._m.items()},
            "v": {k: v.detach().cpu() for k, v in self._v.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        if not state:
            return
        self._t = int(state.get("t", 0))
        self._m = {k: v.clone() for k, v in state.get("m", {}).items()}
        self._v = {k: v.clone() for k, v in state.get("v", {}).items()}

    def aggregate(
        self,
        client_updates: list[ClientUpdate],
        global_state: dict[str, torch.Tensor],
        consume: bool = False,
        inplace: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Pure by default (`consume=False`): caller-owned `client_updates` are
        left untouched. Pass `consume=True` only when you own the fresh,
        per-round updates and are about to discard them anyway — this streams
        the delta accumulation in-place and frees each client's tensors as
        soon as they're folded in.

        `inplace=True` also licenses mutating `global_state` (see
        StrategyBase.aggregate): the Adam step is applied straight to the
        caller's master tensors instead of a clone. Note FedAdam still
        materialises the weighted-mean `delta` plus its own `m`/`v` moments, so
        its host-RAM floor is ~3x the payload regardless — unlike FedAvg it
        cannot be made single-copy without a precision or offload change."""
        if not client_updates:
            return global_state

        # 1) Weighted-mean of client deltas (same as FedAvg's aggregate).
        total = sum(u.num_samples for u in client_updates)
        if total == 0:
            return global_state
        delta: dict[str, torch.Tensor] = {}
        for update in client_updates:
            w = update.num_samples / total
            for k, d in update.delta.items():
                if k not in global_state:
                    continue
                # Upcast to the fp32 master dtype/device first, *then* weight —
                # multiplying in the client dtype would round before the upcast.
                contrib = d.to(global_state[k].dtype).to(global_state[k].device)
                if k in delta:
                    delta[k].add_(contrib, alpha=w)
                else:
                    delta[k] = contrib.mul(w)
            if consume:
                update.delta.clear()  # free this client's tensors as soon as they're folded in

        # 2) Adam-style server update.
        self._t += 1
        new_state = (
            global_state if inplace
            else {k: v.detach().clone() for k, v in global_state.items()}
        )
        for k, d in delta.items():
            # Moments may have been restored from a checkpoint on a different
            # device/dtype (saved on cpu); align them to the aggregated delta.
            m_prev = self._m.get(k)
            m_prev = torch.zeros_like(d) if m_prev is None else m_prev.to(device=d.device, dtype=d.dtype)
            v_prev = self._v.get(k)
            v_prev = torch.zeros_like(d) if v_prev is None else v_prev.to(device=d.device, dtype=d.dtype)
            m = self.beta1 * m_prev + (1.0 - self.beta1) * d
            v = self.beta2 * v_prev + (1.0 - self.beta2) * d.pow(2)
            self._m[k] = m
            self._v[k] = v
            # Bias-corrected step.
            m_hat = m / (1.0 - self.beta1**self._t)
            v_hat = v / (1.0 - self.beta2**self._t)
            step = m_hat / (v_hat.sqrt() + self.eps)
            new_state[k].add_(step, alpha=self.server_lr)
        return new_state
