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

"""Minimal asynchronous FL: update and refill one client at a time.

Run from this directory with: python job.py
"""

import argparse
import queue
import time
from pathlib import Path

import numpy as np

from nvflare.collab import CollabRecipe, collab
from nvflare.recipe import SimEnv


class Trainer:
    @collab.publish
    def train(self, weights, version):
        # Each site has its own synthetic linear-regression data and speed.
        rng = np.random.default_rng(collab.get_app_prop("seed"))
        x = rng.normal(size=(20, 2))
        y = x @ np.array([2.0, -1.0])
        time.sleep(collab.get_app_prop("delay"))
        for _ in range(5):
            weights = weights - 0.1 * x.T @ (x @ weights - y) / len(x)
        return weights, version


class AsyncServer:
    def __init__(self, num_updates=8):
        self.num_updates = num_updates

    @collab.main
    def run(self):
        if self.num_updates < len(collab.clients):
            raise ValueError("num_updates must be at least the number of clients")
        weights = np.zeros(2)
        updates = queue.Queue()

        def accept_result(group_call_context, result):
            # Callbacks can run concurrently; only the main loop updates weights.
            updates.put((collab.caller, result))

        # Call options go BEFORE .train(). This returns while clients train.
        calls = [collab.clients(blocking=False, timeout=30, process_resp_cb=accept_result).train(weights, 0)]
        submitted = len(collab.clients)
        print("Training dispatched; the server is free to process arriving updates.", flush=True)

        version = 0
        while version < self.num_updates:
            if collab.is_aborted:
                raise RuntimeError("Training aborted")
            # Failed calls do not invoke the success callback. Fail explicitly
            # instead of waiting forever for an update that will never arrive.
            for results in calls:
                if results.failures:
                    raise RuntimeError(f"Client training failed: {results.failures}")
            try:
                site, (local_weights, base_version) = updates.get(timeout=0.1)
            except queue.Empty:
                continue

            staleness = version - base_version
            weights = 0.5 * weights + 0.5 * local_weights
            version += 1
            print(
                f"Update {version}/{self.num_updates} from {site}: "
                f"base_version={base_version}, staleness={staleness}, weights={weights}",
                flush=True,
            )
            if submitted < self.num_updates:
                # Refill only the client that finished. Other clients continue
                # training on their original snapshots, with no round barrier.
                calls.append(
                    collab.get_clients([site])(blocking=False, timeout=30, process_resp_cb=accept_result).train(
                        weights.copy(), version
                    )
                )
                submitted += 1

        # Every submitted update has been accepted. Drain the call streams so
        # their callbacks have completed and their failure sets are final.
        for results in calls:
            for _ in results:
                pass
            if results.failures:
                raise RuntimeError(f"Client training failed: {results.failures}")
        output = Path(collab.workspace.get_run_dir(collab.fl_ctx.get_job_id())) / "final_model.npy"
        np.save(output, weights)
        print(f"Final model saved to {output}", flush=True)
        return weights


def make_recipe(args):
    if args.num_updates < 2:
        raise ValueError("num_updates must be >= 2")
    recipe = CollabRecipe(
        job_name="collab_simple_async",
        server=AsyncServer(num_updates=args.num_updates),
        client=Trainer(),
        min_clients=2,
    )
    recipe.set_per_site_config({"site-1": {"seed": 1, "delay": 0.1}, "site-2": {"seed": 2, "delay": 1.0}})
    return recipe


def main():
    parser = argparse.ArgumentParser(description="Simple asynchronous FL with nonblocking Collab calls")
    parser.add_argument("--num-updates", type=int, default=8, help="Total accepted updates across both clients (>= 2)")
    parser.add_argument("--workspace-root", default="/tmp/nvflare/collab/simple_async")
    args = parser.parse_args()
    if args.num_updates < 2:
        parser.error("--num-updates must be >= 2")

    recipe = make_recipe(args)
    run = recipe.execute(SimEnv(clients=recipe.configured_sites(), workspace_root=args.workspace_root))
    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())


if __name__ == "__main__":
    main()
