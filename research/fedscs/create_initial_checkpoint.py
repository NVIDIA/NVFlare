# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Create a deterministic initial SimpleCNN checkpoint."""

import argparse
import os
import random

import numpy as np
import torch
from src.model import SimpleCNN


def set_seed(seed):
    """Set random seeds for deterministic model initialization."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    """Create and save the initial model checkpoint."""
    parser = argparse.ArgumentParser(description="Create an initial SimpleCNN checkpoint.")
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed used for model initialization.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output checkpoint.",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    model = SimpleCNN()

    output_path = os.path.abspath(args.output)
    output_dir = os.path.dirname(output_path)

    os.makedirs(output_dir, exist_ok=True)

    torch.save(model.state_dict(), output_path)

    print(f"Created initial checkpoint: {output_path}")
    print(f"Initialization seed: {args.seed}")


if __name__ == "__main__":
    main()
