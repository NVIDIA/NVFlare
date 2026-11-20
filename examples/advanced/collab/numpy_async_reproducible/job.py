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

"""Reproducible NumPy FedBuff simulation with simulator-owned plans."""

import argparse
from pathlib import Path

import numpy as np
from numpy_async_reproducible.model import NumpyModelOps
from numpy_async_reproducible.seed_policy import SeedPolicy

from nvflare.collab import FedBuffRecipe, collab
from nvflare.recipe import SimEnv


class NumpyModelLoader:
    def load_model(self):
        return {"weights": np.zeros(3, dtype=np.float32)}


class FinalWeightsObserver:
    def on_finish(self, model, version):
        run_dir = Path(collab.workspace.get_run_dir(collab.fl_ctx.get_job_id()))
        np.save(run_dir / "final_weights.npy", model["weights"])


class SeededNumpyClient:
    @collab.publish
    def train(self, assignment_ids, model_version, global_model, logical_assignments, train_seeds=None):
        site = collab.site_name
        identity = assignment_ids[site]
        logical_name = logical_assignments[site]
        # A plan supplies its explicit seed. Live runs use the same independent
        # seed stream without needing a simulator-specific client protocol.
        seed = (
            train_seeds[site]
            if train_seeds is not None
            else SeedPolicy(collab.get_app_prop("run_seed")).seed(
                "train", assignment_id=identity, logical_name=logical_name
            )
        )
        rng = SeedPolicy(seed).numpy_rng("local_update")
        updated = {key: value + rng.normal(size=value.shape).astype(value.dtype) for key, value in global_model.items()}
        return (
            updated,
            {},
            {
                "assignment_id": identity,
                "model_version": model_version,
                "logical_name": logical_name,
                "train_seed": seed,
            },
        )


def make_recipe(run_seed=10):
    recipe = FedBuffRecipe(
        name="collab_numpy_async_reproducible",
        client=SeededNumpyClient(),
        model_loader=NumpyModelLoader(),
        observer=FinalWeightsObserver(),
        model_ops=NumpyModelOps(),
        num_rounds=4,
        concurrency=2,
        buffer_size=1,
        min_open_slots=1,
        call_timeout=30,
    )
    recipe.set_client_prop("run_seed", run_seed)
    # Bundle the example package so relative imports also work in exported jobs.
    package = str(Path(__file__).resolve().parent)
    recipe.add_server_file(package)
    recipe.add_client_file(package)
    return recipe


def make_plan(run_seed=10, mode="fixed"):
    logical = ["alice", "bob", "bob", "alice"]
    policy = SeedPolicy(run_seed)
    execution = {
        "logical_sequence": logical,
        "train_seeds": [
            policy.seed("train", assignment_id=identity, logical_name=name) for identity, name in enumerate(logical)
        ],
    }
    if mode == "fixed":
        # After accepting bob's first assignment, bob is eligible again.
        execution["arrival_sequence"] = [1, 0, 2, 3]
    elif mode == "timed":
        constant = {"distribution": "constant", "value": 1}
        profile = {phase: constant for phase in ("download", "train", "upload")}
        execution["timing"] = {
            "starts": [0, 0, 4, 4],
            "seed": policy.seed("delays"),
            "profiles": {name: profile for name in sorted(set(logical))},
            "horizon": 8,
        }
    else:
        raise ValueError(f"unknown plan mode: {mode}")
    return {"format_version": 1, "async_execution": execution}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--run-seed", type=int, default=10)
    parser.add_argument("--plan-mode", choices=("fixed", "timed"), default="fixed")
    parser.add_argument("--workspace-root", default="/tmp/nvflare/collab")
    args = parser.parse_args()
    run = make_recipe(args.run_seed).execute(
        SimEnv(
            num_clients=args.num_clients,
            workspace_root=args.workspace_root,
            reprod=True,
            plan=make_plan(args.run_seed, args.plan_mode),
        )
    )
    print("Job status:", run.get_status())
    print("Results:", run.get_result())


if __name__ == "__main__":
    main()
