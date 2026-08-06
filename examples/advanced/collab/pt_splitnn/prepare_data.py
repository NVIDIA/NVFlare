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

"""Download CIFAR-10 and prepare aligned SplitNN sample indices."""

import argparse
import json
from pathlib import Path

import numpy as np
from data import INTERSECTION_FILE, MANIFEST_FILE
from torchvision import datasets

DEFAULT_DATA_ROOT = "/tmp/nvflare/datasets/cifar10"


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare CIFAR-10 data for the Collab SplitNN example")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="CIFAR-10 cache and prepared-data directory")
    parser.add_argument("--overlap", type=int, default=10_000, help="Aligned training samples held by both sites")
    parser.add_argument("--seed", type=int, default=0, help="Intersection sampling seed")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing SplitNN preparation metadata")
    return parser


def prepare_data(args) -> Path:
    if args.overlap < 1:
        raise ValueError("--overlap must be >= 1")

    data_root = Path(args.data_root).expanduser().resolve()
    manifest_path = data_root / MANIFEST_FILE
    intersection_path = data_root / INTERSECTION_FILE
    if (manifest_path.exists() or intersection_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"Prepared SplitNN data already exists under '{data_root}'. Use --overwrite to regenerate it."
        )

    data_root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading/verifying CIFAR-10 under {data_root} ...")
    train_dataset = datasets.CIFAR10(root=str(data_root), train=True, download=True)
    test_dataset = datasets.CIFAR10(root=str(data_root), train=False, download=True)
    if args.overlap > len(train_dataset):
        raise ValueError(f"--overlap ({args.overlap}) exceeds the CIFAR-10 training size ({len(train_dataset)})")

    rng = np.random.RandomState(args.seed)
    intersection = np.sort(rng.choice(len(train_dataset), size=args.overlap, replace=False)).astype(np.int64)
    np.save(intersection_path, intersection)
    manifest = {
        "format_version": 1,
        "dataset": "CIFAR10",
        "overlap": args.overlap,
        "seed": args.seed,
        "train_size": len(train_dataset),
        "test_size": len(test_dataset),
    }
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")

    print(f"Prepared {args.overlap} aligned samples under {data_root}")
    print(f"Next: python job.py --data-root {data_root}")
    return data_root


def main():
    prepare_data(define_parser().parse_args())


if __name__ == "__main__":
    main()
