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

"""Seeded logical replay using the standard AsyncRecipe and SimEnv."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from async_research.disk_storage import DiskResultStore, DiskSnapshotStore
from async_research.replay import ReplayScheduler
from async_research.seed_policy import SeededTrainArgs, SeedPolicy

from nvflare.collab import AsyncRecipe, collab
from nvflare.collab.async_models.numpy import NumpyAsyncModelAdapter
from nvflare.recipe import SimEnv


class ResearchApplication:
    def __init__(self, run_config: dict):
        self.run_config = run_config
        self._run_dir = None

    def initialize(self):
        self._run_dir = Path(collab.workspace.get_run_dir(collab.fl_ctx.get_job_id()))
        config_file = self._run_dir / "run_config.json"
        if config_file.exists():
            raise FileExistsError(f"Research artifacts already exist in {self._run_dir}; use a fresh workspace")
        config_file.write_text(json.dumps(self.run_config, indent=2))
        (self._run_dir / "schedule.jsonl").write_text("")
        (self._run_dir / "published_models").mkdir(exist_ok=True)
        return {"weights": np.zeros(3, dtype=np.float32)}

    def on_model(self, model, version):
        np.savez(self._run_dir / "published_models" / f"{version:08d}.npz", **model)

    def on_event(self, event):
        # Append immediately; do not accumulate an in-memory experiment trace.
        with (self._run_dir / "schedule.jsonl").open("a") as stream:
            stream.write(json.dumps(asdict(event)) + "\n")

    def on_finish(self, model, version):
        np.save(self._run_dir / "final_weights.npy", model["weights"])


class ResearchClient:
    @collab.publish
    def train(self, assignment_ids, model_version, global_model, logical_assignments, train_seeds):
        site = collab.site_name
        seed = train_seeds[site]
        rng = SeedPolicy(seed).numpy_rng("local_update")
        updated = {key: value + rng.normal(size=value.shape).astype(value.dtype) for key, value in global_model.items()}
        return (
            updated,
            {},
            {
                "assignment_id": assignment_ids[site],
                "model_version": model_version,
                "logical_name": logical_assignments[site],
                "train_seed": seed,
            },
        )


def make_recipe(run_seed=10):
    # Reserve three logical jobs even on one physical worker. Acceptance order
    # and assignment seeds are independent of physical completion timing.
    order = [2, 0, 1, 3]
    logical = ["alice", "bob", "alice", "carol"]
    recipe = AsyncRecipe(
        job_name="async_research",
        client=ResearchClient(),
        application=ResearchApplication(
            {
                "run_seed": run_seed,
                "K": 3,
                "B": 1,
                "O": 1,
                "arrival_sequence": order,
                "logical_sequence": logical,
            }
        ),
        model_adapter=NumpyAsyncModelAdapter(),
        num_updates=4,
        num_active_jobs=3,
        scheduler=ReplayScheduler(order, logical),
        snapshot_store=DiskSnapshotStore(),
        result_store=DiskResultStore(),
        train_kwargs=SeededTrainArgs(SeedPolicy(run_seed)),
        call_timeout=30,
    )
    # Bundle the example package so relative imports also work in exported jobs.
    package = str(Path(__file__).resolve().parent)
    recipe.add_server_file(package)
    recipe.add_client_file(package)
    return recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--run-seed", type=int, default=10)
    parser.add_argument("--workspace-root", default="/tmp/nvflare/collab")
    args = parser.parse_args()
    run = make_recipe(args.run_seed).execute(SimEnv(num_clients=args.num_clients, workspace_root=args.workspace_root))
    print("Job status:", run.get_status())
    print("Results:", run.get_result())


if __name__ == "__main__":
    main()
