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

"""VLA checkpoint adapter implementing FLARE's LearnablePersistor contract."""

from __future__ import annotations

import dataclasses
import gc
import json
from pathlib import Path

import torch
from nvflare.app_common.abstract.learnable_persistor import LearnablePersistor
from nvflare.app_common.abstract.model import make_model_learnable
from nvflare.app_common.app_constant import AppConstants

from decent_vla._helpers.builders import build_backend_factory, partition_for
from decent_vla._helpers.checkpoints import CheckpointSpec, latest_server_checkpoint
from decent_vla._helpers.client_checkpoint import resume_disabled
from decent_vla._helpers.scope_manifest import write_scope_manifest
from decent_vla._helpers.site import apply_site_paths, load_job_config
from decent_vla._helpers.training import seed_training


class VLAPersistor(LearnablePersistor):
    def __init__(self, config_path):
        super().__init__()
        self.config_path = config_path
        self.start_round = 0
        self.optimizer_state = None
        self.curve = []

    def load(self, fl_ctx):
        workspace = fl_ctx.get_engine().get_workspace()
        job_id = fl_ctx.get_job_id()
        self.cfg = cfg = load_job_config(Path(workspace.get_app_config_dir(job_id)) / self.config_path)
        apply_site_paths(cfg, workspace.get_run_dir(job_id))
        seed_training(cfg.seed, bool(cfg.backend.extra.get("deterministic", False)))
        self.out = Path(cfg.fl.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2, default=str))
        self.ckpts = self.out / "checkpoints"
        self.ckpts.mkdir(exist_ok=True)
        factory, cls = build_backend_factory(cfg, torch.device("cpu"))
        _, manifest = partition_for(cfg, cls)
        manifest.write(self.out / "partition_manifest.json")
        checkpoint = None if resume_disabled() else latest_server_checkpoint(self.ckpts)
        if checkpoint:
            blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.start_round = int(blob["round"]) + 1
            self.optimizer_state = blob.get("optimizer")
            if cfg.fl.strategy.name == "fedadam" and not self.optimizer_state:
                raise ValueError("FedAdam resume requires a native checkpoint containing optimizer state")
            state = blob["state"]
            curve = self.out / "training_curve.json"
            self.curve = (
                [m for m in json.loads(curve.read_text()) if m["round"] < self.start_round] if curve.exists() else []
            )
        else:
            backend = factory()
            write_scope_manifest(self.out, cfg.backend, cfg.fl, backend)
            state = backend.get_state("shared")
            del backend
            gc.collect()
        return make_model_learnable({k: v.detach().cpu().float().numpy() for k, v in state.items()}, {})

    def save(self, learnable, fl_ctx):
        r = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        state = {k: torch.as_tensor(v).detach().cpu().float() for k, v in learnable["weights"].items()}
        blob = {"state": state, "round": r, "optimizer": self.optimizer_state}
        path = self.ckpts / f"server_round_{r + 1:04d}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save(blob, tmp)
        tmp.replace(path)
        if r + 1 == self.cfg.fl.num_rounds:
            factory, _ = build_backend_factory(self.cfg, torch.device("cpu"))
            backend = factory()
            backend.set_state(state, scope="shared")
            backend.save_checkpoint(str(self.ckpts / f"round_{r + 1:04d}.pt"), CheckpointSpec())
            tmp = self.ckpts / "server_round_final.pt.tmp"
            torch.save(blob, tmp)
            tmp.replace(self.ckpts / "server_round_final.pt")
