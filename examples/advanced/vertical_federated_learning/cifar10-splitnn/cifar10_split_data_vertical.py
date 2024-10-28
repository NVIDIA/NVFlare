# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import logging
import sys

from splitnn.cifar10_vertical_data_splitter import Cifar10VerticalDataSplitter

from nvflare.apis.fl_context import FLContext

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
import argparse

from nvflare.apis.fl_constant import ReservedKey


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dir", type=str, default="/tmp/cifar10_vert_splits", help="output folder")
    parser.add_argument("--overlap", type=int, default=10_000, help="number of overlapping samples")
    args = parser.parse_args()

    splitter = Cifar10VerticalDataSplitter(split_dir=args.split_dir, overlap=args.overlap)

    # set up a dummy context for logging
    fl_ctx = FLContext()
    fl_ctx.set_prop(ReservedKey.IDENTITY_NAME, "local")
    fl_ctx.set_prop(ReservedKey.RUN_NUM, "_")

    splitter.split(fl_ctx)  # will download to CIFAR10_ROOT defined in
    # Cifar10DataSplitter


if __name__ == "__main__":
    main()
