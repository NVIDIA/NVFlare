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

"""Fast CPU smoke run using NVIDIA's actual FedCERecipe and PTFedCEHelper."""

import argparse
import os

import torch

from nvflare.app_opt.pt.recipes.fedce import FedCERecipe
from nvflare.recipe import SimEnv

NUM_FEATURES = 6
NUM_CLASSES = 3


def main(args):
    torch.manual_seed(args.seed)
    model = torch.nn.Linear(NUM_FEATURES, NUM_CLASSES)
    client_script = os.path.join(os.path.dirname(__file__), "client.py")
    recipe = FedCERecipe(
        name="fedce-protocol-smoke",
        min_clients=args.n_clients,
        num_rounds=args.num_rounds,
        model=model,
        train_script=client_script,
        train_args=(
            f"--train_samples {args.train_samples} --valid_samples {args.valid_samples} "
            f"--local_epochs {args.local_epochs} --batch_size {args.batch_size} --seed {args.seed}"
        ),
        key_metric="accuracy",
        key_metric_mode="max",
    )

    run = recipe.execute(SimEnv(num_clients=args.n_clients))
    status = str(run.get_status())
    result = run.get_result()
    print(f"FEDCE_SMOKE_STATUS={status}")
    print(f"FEDCE_SMOKE_RESULT={result}")
    if "COMPLETED" not in status.upper():
        raise RuntimeError(f"FedCE smoke run did not complete successfully: {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=3)
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--train_samples", type=int, default=180)
    parser.add_argument("--valid_samples", type=int, default=90)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260906)
    main(parser.parse_args())
