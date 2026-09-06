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

"""Fast CPU-only end-to-end smoke run using real FedOptRecipe and SimEnv."""

import argparse
import os
import sys

import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
os.environ["PYTHONPATH"] = SRC_DIR + os.pathsep + os.environ.get("PYTHONPATH", "")

from adaptive_hetero.nvflare_aggregator import AdaptiveHeterogeneityAggregator  # noqa: E402
from nvflare.app_opt.pt.recipes.fedopt import FedOptRecipe  # noqa: E402
from nvflare.recipe import SimEnv  # noqa: E402


NUM_FEATURES = 6
NUM_CLASSES = 3


def main(args):
    torch.manual_seed(args.seed)
    model = torch.nn.Linear(NUM_FEATURES, NUM_CLASSES)
    aggregator = AdaptiveHeterogeneityAggregator(
        min_weight=0.05,
        max_weight=0.60,
        heterogeneity_threshold=0.08,
        heterogeneity_temperature=0.04,
    )
    client_script = os.path.join(os.path.dirname(__file__), "client.py")
    recipe = FedOptRecipe(
        name="adaptive-hetero-smoke",
        min_clients=args.n_clients,
        num_rounds=args.num_rounds,
        model=model,
        train_script=client_script,
        train_args=(
            f"--train_samples {args.train_samples} --valid_samples {args.valid_samples} "
            f"--local_epochs {args.local_epochs} --batch_size {args.batch_size} --seed {args.seed}"
        ),
        aggregator=aggregator,
        optimizer_args={"path": "torch.optim.SGD", "args": {"lr": 1.0}},
        device="cpu",
    )

    run = recipe.execute(SimEnv(num_clients=args.n_clients))
    status = str(run.get_status())
    result = run.get_result()
    print(f"ADAPTIVE_HETERO_SMOKE_STATUS={status}")
    print(f"ADAPTIVE_HETERO_SMOKE_RESULT={result}")
    if "COMPLETED" not in status.upper():
        raise RuntimeError(f"NVFlare smoke run did not complete successfully: {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=3)
    parser.add_argument("--num_rounds", type=int, default=2)
    parser.add_argument("--train_samples", type=int, default=180)
    parser.add_argument("--valid_samples", type=int, default=90)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260906)
    main(parser.parse_args())
