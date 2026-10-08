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

"""CPU demonstration settings and a FLARE adapter for FABRIC's FedAvg function."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import torch
from core.aggregation import fedavg_state_dict
from fabric_common import SITES, cpu_state, inside, read_json, state_digest, write_json

from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator


def load_demo_runtime(path: Path) -> dict:
    runtime = read_json(path)
    if runtime.get("kind") != "fabric_synthetic_demo":
        raise ValueError("This entry point accepts only generated synthetic demo inputs")
    directory = inside(inside(Path(runtime["root"]), ".demo"), runtime["demo_dir"])
    inside(directory, path)
    settings = runtime["settings"]
    if settings["device"] != "cpu" or settings["amp"] or settings["rounds"] != 2:
        raise ValueError("The demonstration requires CPU execution, no AMP, and two rounds")
    if list(runtime["client_counts"]) != list(SITES):
        raise ValueError("Unexpected demonstration client order")
    return runtime


def training_args(runtime: dict) -> SimpleNamespace:
    return SimpleNamespace(**runtime["settings"])


def round_seed(runtime: dict, round_index: int, site: str) -> int:
    return runtime["settings"]["seed"] + (round_index + 1) * 100 + SITES.index(site)


def read_demo_manifest(runtime: dict, name: str):
    import pandas as pd

    directory = Path(runtime["demo_dir"])
    frame = pd.read_csv(inside(directory / "data", f"{name}.csv"))
    if frame.empty or not frame["patient_id"].str.startswith("synthetic-").all():
        raise ValueError("Demo manifests must contain explicitly synthetic identifiers")
    if frame["patient_id"].duplicated().any() or set(frame["recurrence_label"]) != {0, 1}:
        raise ValueError("Demo cases must be unique and include both labels")
    frame["feature_paths"] = frame["feature_paths"].map(lambda value: str(inside(directory / "data", value)))
    for path in frame["feature_paths"]:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    return frame


class DemoFedAvg(ModelAggregator):
    """Reuse FABRIC's averaging arithmetic without relaxing study input checks."""

    def __init__(self, runtime_path: str):
        super().__init__()
        self.runtime_path = runtime_path
        self.runtime = load_demo_runtime(Path(runtime_path))
        self.directory = Path(self.runtime["demo_dir"])
        self.expected_digest = self.runtime["initial_state_sha256"]
        self.round_index = 0
        self.updates = {}

    def reset_stats(self):
        if self.updates:
            raise RuntimeError("Refusing to discard unaggregated demo updates")

    def accept_model(self, model: FLModel):
        site = model.meta.get("client_name")
        if site not in SITES or model.meta.get("site") != site or site in self.updates:
            raise ValueError("Unexpected or duplicate demonstration client")
        if model.current_round != self.round_index or model.params_type != ParamsType.FULL:
            raise ValueError("Expected a full update for the current round")
        if model.meta.get("train_patients") != self.runtime["client_counts"][site]:
            raise ValueError("Synthetic case count changed")
        if model.meta.get("seed") != round_seed(self.runtime, self.round_index, site):
            raise ValueError("Unexpected client seed")
        if model.meta.get("input_state_sha256") != self.expected_digest:
            raise ValueError("Clients must start from the same global model")
        state = cpu_state(model.params)
        expected = self.runtime["state_spec"]
        if list(state) != list(expected):
            raise ValueError("Model parameter keys or order changed")
        for key, value in state.items():
            if list(value.shape) != expected[key]["shape"] or str(value.dtype) != expected[key]["dtype"]:
                raise ValueError(f"Parameter shape or dtype changed: {key}")
        loss = float(model.metrics["train_loss"])
        if not math.isfinite(loss):
            raise ValueError("Training loss is not finite")
        self.updates[site] = (state, loss)

    def aggregate_model(self) -> FLModel:
        if set(self.updates) != set(SITES):
            raise RuntimeError("All three demonstration clients must finish")
        weights = [self.runtime["client_counts"][site] for site in SITES]
        state = cpu_state(fedavg_state_dict([self.updates[site][0] for site in SITES], weights))
        digest = state_digest(state)
        number = self.round_index + 1
        audit_path = self.directory / "server_audit" / f"round_{number}.json"
        if audit_path.exists():
            raise FileExistsError(audit_path)
        losses = [self.updates[site][1] for site in SITES]
        torch.save({"model": state}, self.directory / "aggregated_model.pt")
        write_json(
            audit_path,
            {
                "round": number,
                "clients": list(SITES),
                "synthetic_case_counts": weights,
                "training_losses": losses,
                "input_state_sha256": self.expected_digest,
                "output_state_sha256": digest,
            },
        )
        print(f"[FABRIC demo] aggregated round {number}/2 from all three clients", flush=True)
        result = FLModel(
            params=state,
            params_type=ParamsType.FULL,
            current_round=self.round_index,
            metrics={"train_loss": sum(losses) / len(losses)},
        )
        self.round_index += 1
        self.expected_digest = digest
        self.updates = {}
        return result
