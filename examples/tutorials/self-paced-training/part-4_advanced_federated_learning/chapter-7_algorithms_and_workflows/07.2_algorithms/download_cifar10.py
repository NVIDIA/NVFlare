# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import torchvision.datasets as datasets

# default dataset path
CIFAR10_ROOT = "/tmp/nvflare/data/cifar10"


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=CIFAR10_ROOT, nargs="?")
    args = parser.parse_args()
    return args


def main(args):
    datasets.CIFAR10(root=args.dataset_path, train=True, download=True)
    datasets.CIFAR10(root=args.dataset_path, train=False, download=True)


if __name__ == "__main__":
    main(define_parser())
