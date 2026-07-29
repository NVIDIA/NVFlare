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

"""Download CIFAR-10 once and prepare deterministic logical-client splits.

Run from the ``examples/advanced/collab/pt_async_cifar10`` directory before
starting the Collab recipe:

    python prepare_data.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from data import MANIFEST_FILE, SPLITS_DIR, make_client_indices
from torchvision import datasets

DEFAULT_DATA_ROOT = "/tmp/nvflare/datasets/cifar10"


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare CIFAR-10 splits for the Collab async example")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="CIFAR-10 cache and prepared-split directory")
    parser.add_argument("--num-clients", type=int, default=10, help="Number of logical client splits")
    parser.add_argument("--subset-size", type=int, default=350, help="Training examples sampled per logical client")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5, help="Dirichlet class-skew parameter")
    parser.add_argument("--val-split", type=float, default=0.15, help="Fraction reserved from logical-client training")
    parser.add_argument(
        "--kd-split", type=float, default=0.10, help="Additional fraction held out of logical-client training"
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic partition seed")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing manifest and site split files in the selected data root",
    )
    return parser


def prepare_data(args) -> Path:
    if args.num_clients < 1:
        raise ValueError("--num-clients must be >= 1")
    if args.subset_size < 1:
        raise ValueError("--subset-size must be >= 1")
    if args.dirichlet_alpha <= 0:
        raise ValueError("--dirichlet-alpha must be > 0")
    if args.val_split < 0 or args.kd_split < 0 or args.val_split + args.kd_split >= 1:
        raise ValueError("--val-split and --kd-split must be non-negative and sum to less than 1")

    data_root = Path(args.data_root).expanduser().resolve()
    manifest_path = data_root / MANIFEST_FILE
    splits_dir = data_root / SPLITS_DIR
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Prepared data already exists at '{manifest_path}'. Use --overwrite to regenerate its split files."
        )

    data_root.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading/verifying CIFAR-10 under {data_root} ...")
    train_dataset = datasets.CIFAR10(root=str(data_root), train=True, download=True)
    datasets.CIFAR10(root=str(data_root), train=False, download=True)

    total_size = len(train_dataset)
    val_size = int(args.val_split * total_size)
    kd_size = int(args.kd_split * total_size)
    train_size = total_size - val_size - kd_size
    if args.subset_size > train_size:
        raise ValueError(f"--subset-size ({args.subset_size}) exceeds the available training pool ({train_size})")

    generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(total_size, generator=generator).numpy()
    train_pool = permutation[:train_size].astype(np.int64)
    labels = np.asarray(train_dataset.targets)
    class_indices = {class_id: train_pool[labels[train_pool] == class_id] for class_id in range(10)}

    print(
        f"Preparing {args.num_clients} logical clients with {args.subset_size} examples each "
        f"(Dirichlet alpha={args.dirichlet_alpha}) ..."
    )
    expected_files = set()
    for client_index in range(1, args.num_clients + 1):
        site_name = f"site-{client_index}"
        indices = make_client_indices(
            class_indices=class_indices,
            train_pool=train_pool,
            subset_size=args.subset_size,
            dirichlet_alpha=args.dirichlet_alpha,
            seed=args.seed + client_index,
        )
        split_file = splits_dir / f"{site_name}.npy"
        np.save(split_file, indices)
        expected_files.add(split_file.name)

    if args.overwrite:
        for stale_file in splits_dir.glob("site-*.npy"):
            if stale_file.name not in expected_files:
                stale_file.unlink()

    manifest = {
        "format_version": 1,
        "dataset": "CIFAR10",
        "num_clients": args.num_clients,
        "subset_size": args.subset_size,
        "dirichlet_alpha": args.dirichlet_alpha,
        "val_split": args.val_split,
        "kd_split": args.kd_split,
        "split_seed": args.seed,
        "train_pool_size": train_size,
        "test_size": 10000,
    }
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")

    print(f"Prepared data written to {data_root}")
    print(f"Next: python job.py --data-root {data_root}")
    return data_root


def main():
    prepare_data(define_parser().parse_args())


if __name__ == "__main__":
    main()
