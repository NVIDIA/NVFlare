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
#
# Authors: Anbang Liu, Junhan Zhao, and Ziyue Xu

"""Patient-count-weighted FedAvg for FABRIC's NVIDIA FLARE clients."""

from __future__ import annotations

from pathlib import Path

from fabric_common import (
    SITES,
    cpu_state,
    inside,
    load_core,
    load_setup,
    local_seed,
    read_json,
    state_digest,
    write_json,
)
from fabric_resume import atomic_save

from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator


class FabricFedAvg(ModelAggregator):
    def __init__(self, runtime_path: str):
        super().__init__()
        self.runtime_path = runtime_path
        self.runtime = read_json(Path(runtime_path))
        self.root = Path(self.runtime["root"]).resolve()
        self.fold_dir = inside(inside(self.root, "results"), self.runtime["fold_dir"])
        inside(self.fold_dir, runtime_path)
        self.setup = load_setup(
            self.root, self.runtime["experiment"], self.runtime.get("variant", "pooling"), self.runtime.get("top_k")
        )
        self.core = load_core()
        self.round_offset = self.runtime.get("resume_offset", 0)
        if type(self.round_offset) is not int or not 0 <= self.round_offset < self.setup["settings"]["rounds"]:
            raise ValueError("Invalid resume round offset")
        self.next_round = 0
        self.expected_digest = self.runtime.get("resume_state_sha256", self.runtime["initial_state_sha256"])
        self.updates = {}

    def reset_stats(self):
        if self.updates:
            raise RuntimeError("Refusing to discard unaggregated client updates")
        self.updates = {}

    def accept_model(self, model: FLModel):
        site = model.meta.get("client_name")  # Identity supplied by FLARE server.
        if site not in SITES or model.meta.get("site") != site or site in self.updates:
            raise ValueError(f"Unexpected or duplicate client contribution: {site}")
        if model.current_round != self.next_round or model.params_type != ParamsType.FULL:
            raise ValueError("Stale/wrong-round or non-FULL update")
        if model.meta.get("train_patients") != self.runtime["client_patients"][site]:
            raise ValueError("Patient aggregation weight changed")
        if model.meta.get("local_epochs") != self.setup["settings"]["local_epochs"]:
            raise ValueError("Local epoch count changed")
        if model.meta.get("seed") != local_seed(
            self.setup, self.runtime["fold"], self.next_round + self.round_offset, site
        ):
            raise ValueError("Local seed changed")
        if model.meta.get("input_state_sha256") != self.expected_digest:
            raise ValueError("Clients did not receive the expected identical global model")
        state = cpu_state(model.params)
        if list(state) != list(self.runtime["state_spec"]):
            raise ValueError("Parameter keys/order differ from the original initialized model")
        for key, specification in self.runtime["state_spec"].items():
            if list(state[key].shape) != specification["shape"] or str(state[key].dtype) != specification["dtype"]:
                raise ValueError(f"Parameter shape/dtype changed: {key}")
        if self.updates:
            first = next(iter(self.updates.values()))[0]
            if list(state) != list(first):
                raise ValueError("Model parameter keys/order differ across clients")
            for key in state:
                if state[key].shape != first[key].shape or state[key].dtype != first[key].dtype:
                    raise ValueError(f"Model parameter specification differs: {key}")
        self.updates[site] = (state, model.meta["local_log"])

    def aggregate_model(self) -> FLModel:
        if set(self.updates) != set(SITES):
            raise RuntimeError("All three clients must complete each round")
        states = [self.updates[site][0] for site in SITES]
        weights = [self.runtime["client_patients"][site] for site in SITES]
        # Normalize site patient counts before float32 accumulation in SITES
        # order, preserving nonfloating buffers.
        state = self.core.aggregation.fedavg_state_dict(states, weights)
        state = cpu_state(state)
        digest = state_digest(state)
        original_round = self.next_round + self.round_offset + 1
        logs = [self.updates[site][1] for site in SITES]
        if (self.fold_dir / "server_audit" / f"round_{original_round}.json").exists():
            raise FileExistsError(f"Round {original_round} was already aggregated")
        checkpoint = {
            "model_state_dict": state,
            "config": self.runtime["model_config"],
            "fold": self.runtime["fold"],
            "condition": self.setup["model_key"],
            "completed_rounds": original_round,
            "state_sha256": digest,
        }
        # Save atomically before committing the audit. This adds persistence
        # only; the state, arithmetic, client order and returned model are unchanged.
        atomic_save(checkpoint, self.fold_dir / "server_checkpoints" / f"round_{original_round}.pt")
        write_json(
            self.fold_dir / "server_audit" / f"round_{original_round}.json",
            {
                "fold": self.runtime["fold"],
                "round": original_round,
                "sites_in_aggregation_order": list(SITES),
                "patient_weights": weights,
                "input_state_sha256": self.expected_digest,
                "output_state_sha256": digest,
                "local_logs": logs,
            },
        )
        if original_round == self.setup["settings"]["rounds"]:
            destination = self.fold_dir / "global_model_round_final.pt"
            if destination.exists():
                raise FileExistsError(destination)
            atomic_save(checkpoint, destination)
        print(
            f"[FABRIC FedAvg] fold={self.runtime['fold']} round={original_round} "
            f"sites={list(SITES)} patient_weights={weights}",
            flush=True,
        )
        self.updates = {}
        result = FLModel(
            params=state,
            params_type=ParamsType.FULL,
            current_round=self.next_round,
            metrics={"train_loss": sum(float(item["loss"]) for item in logs) / len(logs)},
        )
        self.expected_digest = digest
        self.next_round += 1
        return result
