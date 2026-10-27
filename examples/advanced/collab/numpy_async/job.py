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

"""Small NumPy consumer of AsyncRecipe. Run from this directory: python job.py."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from nvflare.collab import AsyncRecipe, collab
from nvflare.collab.async_models.numpy import NumpyAsyncModelAdapter
from nvflare.collab.async_scheduling import ReplayScheduler
from nvflare.collab.seed_policy import SeededTrainArgs, SeedPolicy
from nvflare.recipe import SimEnv


class ArrayApplication:
    def __init__(self):
        self._events = []

    def initialize(self):
        self._events = []
        return {"weights": np.zeros(3, dtype=np.float32)}

    def on_event(self, event):
        self._events.append(asdict(event))

    def on_finish(self, model, version):
        run_dir = Path(collab.workspace.get_run_dir(collab.fl_ctx.get_job_id()))
        np.save(run_dir / "final_weights.npy", model["weights"])
        (run_dir / "schedule.json").write_text(json.dumps(self._events, indent=2))


class ArrayClient:
    @collab.publish
    def train(self, assignment_ids, model_version, global_model, logical_assignments, train_seeds=None):
        metadata = {
            "assignment_id": assignment_ids[collab.site_name],
            "model_version": model_version,
            "logical_name": logical_assignments[collab.site_name],
        }
        if train_seeds is None:
            updated = {key: value + 1 for key, value in global_model.items()}
        else:
            seed = train_seeds[collab.site_name]
            metadata["train_seed"] = seed
            rng = SeedPolicy(seed).numpy_rng("local_update")
            updated = {
                key: value + rng.normal(size=value.shape).astype(value.dtype) for key, value in global_model.items()
            }
        return updated, {}, metadata


def make_recipe(num_clients=2, num_updates=4, buffer_size=2, min_open_slots=1, arrival_sequence=None, run_seed=None):
    return AsyncRecipe(
        job_name="numpy_async",
        application=ArrayApplication(),
        client=ArrayClient(),
        model_adapter=NumpyAsyncModelAdapter(),
        num_updates=num_updates,
        num_active_jobs=num_clients,
        buffer_size=buffer_size,
        min_open_slots=min_open_slots,
        min_clients=num_clients,
        call_timeout=30,
        scheduler=ReplayScheduler(arrival_sequence) if arrival_sequence is not None else None,
        train_kwargs=SeededTrainArgs(SeedPolicy(run_seed)) if run_seed is not None else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--num-updates", type=int, default=4)
    parser.add_argument("--buffer-size", type=int, default=2)
    parser.add_argument("--min-open-slots", type=int, default=1)
    parser.add_argument(
        "--arrival-sequence",
        type=int,
        nargs="+",
        default=None,
        help="Replay assignment IDs in this order (num-updates * buffer-size entries)",
    )
    parser.add_argument(
        "--run-seed", type=int, default=None, help="Use reproducible random local deltas instead of constant deltas"
    )
    args = parser.parse_args()
    recipe = make_recipe(
        args.num_clients, args.num_updates, args.buffer_size, args.min_open_slots, args.arrival_sequence, args.run_seed
    )
    run = recipe.execute(SimEnv(num_clients=args.num_clients))
    print("Job status:", run.get_status())
    print("Results:", run.get_result())


if __name__ == "__main__":
    main()
