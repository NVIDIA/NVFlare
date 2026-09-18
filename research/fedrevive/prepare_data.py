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

"""Prepare the CIFAR-10 split used by the FedRevive Figure 2 experiment."""

import argparse

from data import DELAY_SCHEDULES, prepare_cifar10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="/tmp/fedrevive/cifar10")
    parser.add_argument("--download-root", default="/tmp/cifar10")
    parser.add_argument("--num-logical-clients", type=int, default=1000)
    parser.add_argument("--client-data-size", type=int, default=350)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--setup-seed", type=int, default=10)
    parser.add_argument("--delay-schedule", choices=DELAY_SCHEDULES, default="default")
    args = parser.parse_args()
    result = prepare_cifar10(
        output_root=args.output_root,
        download_root=args.download_root,
        num_logical_clients=args.num_logical_clients,
        client_data_size=args.client_data_size,
        alpha=args.alpha,
        setup_seed=args.setup_seed,
        delay_schedule=args.delay_schedule,
    )
    print(f"Prepared FedRevive CIFAR-10 data at {result}")


if __name__ == "__main__":
    main()
