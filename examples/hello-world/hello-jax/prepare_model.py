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

"""
Prepare an initial flattened JAX model checkpoint for hello-jax.
"""

import argparse

import numpy as np
from model import create_initial_params, flatten_params


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    initial_params = flatten_params(create_initial_params())
    np.save(args.output, initial_params)


if __name__ == "__main__":
    main()
