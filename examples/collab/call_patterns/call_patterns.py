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

"""Server-to-client call patterns.

The same trainer is driven by two server strategies that invoke clients in
different ways, selected with --pattern:

    seq     iterate over ``collab.clients`` and call them one at a time,
            aggregating at the end of the round
    cyclic  relay the model around the ring of clients (each client hands its
            result to the next)

The parallel group call (``collab.clients.train(...)`` fanning out to every
client at once) is shown in the hello_fedavg example.

Run:
    python -m collab.call_patterns.call_patterns --pattern seq
    python -m collab.call_patterns.call_patterns --pattern cyclic
"""

import argparse

from collab.call_patterns.avg_seq import NPFedAvgSequential
from collab.call_patterns.cyclic import NPCyclic
from collab.call_patterns.trainer import NPTrainer
from collab.call_patterns.widgets import MetricReceiver

from nvflare.collab import CollabRecipe, simple_logging
from nvflare.recipe import SimEnv

INITIAL_MODEL = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


def make_strategy(pattern: str, num_rounds: int):
    if pattern == "seq":
        return NPFedAvgSequential(initial_model=INITIAL_MODEL, num_rounds=num_rounds)
    if pattern == "cyclic":
        return NPCyclic(initial_model=INITIAL_MODEL, num_rounds=num_rounds)
    raise SystemExit(f"unknown pattern {pattern!r}")


def make_recipe(args):
    return CollabRecipe(
        job_name=f"collab_{args.pattern}",
        server=make_strategy(args.pattern, args.num_rounds),
        client=NPTrainer(delta=1.0),
        server_objects={"metric_receiver": MetricReceiver()},
        min_clients=args.num_clients,
        sync_task_timeout=60,
    )


def main():
    parser = argparse.ArgumentParser(description="Collab call patterns: seq | cyclic")
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--pattern", choices=("seq", "cyclic"), default="seq")
    parser.add_argument("--num-rounds", type=int, default=2)
    args = parser.parse_args()
    simple_logging()
    run = make_recipe(args).execute(SimEnv(num_clients=args.num_clients))
    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())


if __name__ == "__main__":
    main()
