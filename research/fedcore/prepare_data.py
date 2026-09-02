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

"""Prepare deterministic MNIST image-plus-context data for FedCoRe."""

import argparse
from pathlib import Path

from src.data import MNISTDataConfig, generate_mnist_data
from src.validation import positive_int, probability


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare MNIST image-plus-context data for FedCoRe.")
    parser.add_argument("--output-dir", default="", help="Defaults to data/<scenario>.")
    parser.add_argument("--dataset-root", default="~/.cache/nvflare/fedcore")
    parser.add_argument("--scenario", choices=["recoverable", "uninformative"], default="recoverable")
    parser.add_argument("--train-samples-per-site", type=positive_int, default=96)
    parser.add_argument("--val-samples-per-site", type=positive_int, default=64)
    parser.add_argument("--test-samples-per-site", type=positive_int, default=64)
    parser.add_argument("--proxy-strength", type=probability, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--image-size", type=positive_int, default=224)
    return parser


def _resolve_output_dir(args) -> Path:
    return Path(args.output_dir) if args.output_dir else Path("data") / args.scenario


def main() -> None:
    args = define_parser().parse_args()
    if args.scenario == "uninformative" and args.proxy_strength is not None:
        raise ValueError("--proxy-strength is fixed at 0.5 for the uninformative scenario.")
    proxy_strength = args.proxy_strength if args.proxy_strength is not None else 0.9
    output_dir = _resolve_output_dir(args)
    summary = generate_mnist_data(
        MNISTDataConfig(
            output_dir=output_dir,
            dataset_root=Path(args.dataset_root),
            scenario=args.scenario,
            train_samples_per_site=args.train_samples_per_site,
            val_samples_per_site=args.val_samples_per_site,
            test_samples_per_site=args.test_samples_per_site,
            proxy_strength=proxy_strength,
            seed=args.seed,
            image_size=args.image_size,
        )
    )
    print(f"Prepared {summary['total_examples']} MNIST examples under {output_dir.resolve()}")
    for site, site_summary in summary["sites"].items():
        train = site_summary["splits"]["train"]
        print(
            f"  {site}: train={train['examples']} image_available={train['image_available']} "
            f"image_missing={train['image_missing']} sensor_matches={train['sensor_matches_label']}"
        )


if __name__ == "__main__":
    main()
