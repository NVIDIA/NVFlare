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

"""VLA adapters for FLARE's FedAvg and FedOpt workflows; no custom round loop."""

from __future__ import annotations

import json
import math

import numpy as np
import torch
from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.fedopt_ctl import FedOpt

from decent_vla.runtime import wire


def _wire_array(value, dtype):
    array = wire.cast(torch.as_tensor(value), dtype).numpy()
    return array.copy() if dtype == torch.float32 else array


class SharedModel(torch.nn.Module):
    """CPU parameter tree containing only federated tensors, with original names."""

    def __init__(self, state):
        super().__init__()
        for name, value in state.items():
            node = self
            parts = name.split(".")
            for part in parts[:-1]:
                if part not in node._modules:
                    node.add_module(part, torch.nn.Module())
                node = node._modules[part]
            node.register_parameter(parts[-1], torch.nn.Parameter(torch.as_tensor(value).float()))


class VLAWorkflow:
    """Configuration, wire casting and checkpoint hooks around the native loop."""

    def run(self):
        self.model = self.load_model()
        self.cfg = cfg = self.persistor.cfg
        from decent_vla.runtime import observability

        observability.configure_wandb(cfg)
        self.start_round = self.persistor.start_round
        self.num_rounds = max(0, cfg.fl.num_rounds - self.start_round)
        self.num_clients = cfg.data.num_clients
        self.memory_gc_rounds = cfg.fl.nvflare.memory_gc_rounds
        self.client_losses = {}
        self.client_samples = {}
        if isinstance(self, FedOpt):
            self.source_model = SharedModel(self.model.params)
            self.optimizer_args = {
                "path": "torch.optim.Adam",
                "args": {
                    "lr": cfg.fl.strategy.server_lr,
                    "betas": tuple(cfg.fl.strategy.server_betas),
                    "eps": cfg.fl.strategy.server_eps,
                },
            }
        if not self.num_rounds:
            from nvflare.app_common.abstract.model import make_model_learnable
            from nvflare.app_common.app_constant import AppConstants

            self.current_round = cfg.fl.num_rounds - 1
            self.fl_ctx.set_prop(AppConstants.CURRENT_ROUND, self.current_round, private=True, sticky=False)
            self.persistor.save(make_model_learnable(self.model.params, self.model.meta), self.fl_ctx)
            return
        super().run()

    def send_model(self, task_name="train", data=None, targets=None, **kwargs):
        # Keep the fp32 master untouched. Quantization belongs to the downlink only.
        dt, _ = wire.resolve(self.cfg.fl.wire_dtype)
        payload = FLModel(
            params={k: _wire_array(v, dt) for k, v in data.params.items()},
            current_round=data.current_round,
            total_rounds=self.cfg.fl.num_rounds,
            meta=dict(data.meta or {}),
        )
        self.client_losses, self.client_samples = {}, {}
        self.broadcast_model(
            task_name=task_name,
            data=payload,
            targets=targets,
            min_responses=self.cfg.fl.min_clients or len(targets),
            timeout=math.ceil(self.cfg.fl.round_timeout_s or 0),
            wait_time_after_min_received=self.cfg.fl.nvflare.wait_time_after_min_received,
            blocking=False,
            **kwargs,
        )

    def _aggregate_one_result(self, result):
        # Decode wire fp16 to fp32 BEFORE the native weighted sum: weighting a
        # fp16 delta by a dataset's frame count can overflow or round prematurely.
        result.params = {k: np.asarray(v, dtype=np.float32) for k, v in (result.params or {}).items()}
        cid = result.meta.get("client_name", "unknown")
        if result.metrics and "loss" in result.metrics:
            self.client_losses[cid] = result.metrics["loss"]
        self.client_samples[cid] = result.meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND)
        return super()._aggregate_one_result(result)

    def _get_aggregated_result(self):
        need = self.cfg.fl.min_clients or self.cfg.data.num_clients
        if self._received_count < need:
            raise RuntimeError(f"round {self.current_round}: {self._received_count}/{need} usable updates")
        return super()._get_aggregated_result()

    def update_model(self, global_model, aggr_result):
        if isinstance(self, FedOpt) and self.persistor.optimizer_state:
            self.optimizer.load_state_dict(self.persistor.optimizer_state)
            self.persistor.optimizer_state = None
        model = super().update_model(global_model, aggr_result)
        losses = dict(sorted(self.client_losses.items()))
        self.persistor.curve.append(
            {
                "round": self.current_round,
                "client_losses": losses,
                "client_samples": dict(sorted(self.client_samples.items())),
                "mean_client_loss": sum(losses.values()) / len(losses) if losses else None,
            }
        )
        from decent_vla.runtime import observability

        observability.wandb_round_log(
            self.cfg.name,
            self.current_round,
            losses,
            self.persistor.curve[-1]["mean_client_loss"],
            {"n_updates": len(losses)},
        )
        path = self.persistor.out / "training_curve.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.persistor.curve, indent=2))
        tmp.replace(path)
        return model

    def save_model(self, model):
        self.persistor.optimizer_state = self.optimizer.state_dict() if isinstance(self, FedOpt) else None
        # Native FedAvg invokes this hook each round. Persist at the original cadence.
        n = self.cfg.fl.save_every_n_rounds
        if (n and (self.current_round + 1) % n == 0) or self.current_round + 1 == self.cfg.fl.num_rounds:
            super().save_model(model)


class VLAFedAvg(VLAWorkflow, FedAvg):
    pass


class VLAFedAdam(VLAWorkflow, FedOpt):
    def __init__(self, **kwargs):
        super().__init__(
            source_model="unused",
            device="cpu",
            lr_scheduler_args={
                "path": "torch.optim.lr_scheduler.ConstantLR",
                "args": {"factor": 1.0, "total_iters": 1},
            },
            **kwargs,
        )
