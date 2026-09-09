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
"""Export the CLI tutorial's Hello PyTorch job with client log streaming."""

import argparse
import sys
from pathlib import Path

HELLO_PT_DIR = Path(__file__).resolve().parents[1] / "hello-world" / "hello-pt"
sys.path.insert(0, str(HELLO_PT_DIR))

from model import create_model  # noqa: E402

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", default="/tmp/nvflare/jobs/job_config")
    parser.add_argument("--num_rounds", type=int, default=2)
    args = parser.parse_args()

    recipe = FedAvgRecipe(
        name="hello-pt",
        min_clients=2,
        num_rounds=args.num_rounds,
        model=create_model(),
        train_script=str(HELLO_PT_DIR / "client.py"),
        train_args=["--dataset", "synthetic"],
    )
    recipe.enable_log_streaming()
    recipe.export(job_dir=args.job_dir)
    print(f"Job exported to {Path(args.job_dir) / 'hello-pt'}")


if __name__ == "__main__":
    main()
