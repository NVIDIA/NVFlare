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

"""Run two clients/two rounds on one GPU; use --export for a deployable job configuration."""

import argparse
from pathlib import Path

from hf_model import ROUNDS, DossierAuditor

from nvflare.app_common.filters.memorization_audit_filter import (
    AuditModelSnapshot,
    AuditSnapshotFilter,
    MemorizationAuditFilter,
)
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assets = str(Path(args.assets).resolve())
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    measurements = output / "measurements"
    measurements.mkdir()
    recipe = FedAvgRecipe(
        name="memorization-hf",
        min_clients=2,
        num_rounds=ROUNDS,
        model={"path": "hf_model.CausalLM", "args": {"assets_dir": assets}},
        initial_ckpt=str(Path(assets) / "initial.pt"),
        train_script="client.py",
        train_args=["--assets", assets, "--measurements", str(measurements)],
        key_metric="ce",
        key_metric_mode="min",
    )
    recipe.add_client_component(AuditModelSnapshot(), "audit_snapshot")
    recipe.add_client_component(DossierAuditor(assets), "memorization_auditor")
    recipe.add_client_input_filter(AuditSnapshotFilter("audit_snapshot"), tasks=["train"])
    recipe.add_client_output_filter(MemorizationAuditFilter("memorization_auditor", "audit_snapshot"), tasks=["train"])
    run = recipe.execute(
        SimEnv(num_clients=2, num_threads=2, gpu_config="0", workspace_root=str(output / "simulation"))
    )
    print("Artifacts:", run.get_result())


if __name__ == "__main__":
    main()
