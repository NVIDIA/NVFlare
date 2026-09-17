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

import argparse
from pathlib import Path

import torchvision.datasets as datasets

DATASET_ROOT = "/tmp/nvflare/data"


def define_parser():
    parser = argparse.ArgumentParser(description="Download CIFAR-10 for the Hello Lightning example")
    parser.add_argument("--data_root", type=str, default=DATASET_ROOT)
    return parser.parse_args()


def prepare_data(data_root: Path):
    datasets.CIFAR10(root=data_root, train=True, download=True)
    datasets.CIFAR10(root=data_root, train=False, download=True)


def main():
    args = define_parser()
    data_root = Path(args.data_root).expanduser().resolve()
    prepare_data(data_root)
    print(f"Prepared CIFAR-10 under {data_root}")


if __name__ == "__main__":
    main()
