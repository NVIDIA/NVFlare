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

"""Download CIFAR-10 once and create disjoint Dirichlet client partitions."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from data import DEFAULT_DATA_ROOT, MANIFEST_FILE, SPLITS_DIR
from torchvision import datasets

MIN_SAMPLES_PER_CLIENT = 10


def partition_data(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_samples: int = MIN_SAMPLES_PER_CLIENT,
    max_attempts: int = 1000,
) -> list[np.ndarray]:
    """Match the Dirichlet partitioning used by the standard CIFAR-10 examples."""

    rng = np.random.RandomState(seed)
    num_examples = len(labels)
    for _attempt in range(max_attempts):
        partitions = [[] for _ in range(num_clients)]
        for class_id in np.unique(labels):
            class_indices = np.flatnonzero(labels == class_id)
            rng.shuffle(class_indices)
            probabilities = rng.dirichlet(np.full(num_clients, alpha))
            probabilities = np.asarray(
                [
                    probability * (len(indices) < num_examples / num_clients)
                    for probability, indices in zip(probabilities, partitions)
                ]
            )
            probabilities /= probabilities.sum()
            split_points = (np.cumsum(probabilities) * len(class_indices)).astype(int)[:-1]
            partitions = [
                indices + split.tolist() for indices, split in zip(partitions, np.split(class_indices, split_points))
            ]

        if min(map(len, partitions)) >= min_samples:
            result = []
            for indices in partitions:
                indices = np.asarray(indices, dtype=np.int64)
                rng.shuffle(indices)
                result.append(indices)
            return result

    raise RuntimeError(
        f"Could not create a partition with at least {min_samples} examples per client after {max_attempts} attempts"
    )


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet data-heterogeneity parameter")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def prepare_data(args) -> Path:
    if args.num_clients < 2:
        raise ValueError("--num-clients must be at least 2")
    if args.alpha <= 0:
        raise ValueError("--alpha must be greater than 0")

    data_root = Path(args.data_root).expanduser().resolve()
    manifest_path = data_root / MANIFEST_FILE
    splits_dir = data_root / SPLITS_DIR
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"{manifest_path} already exists; pass --overwrite to regenerate the partitions")

    data_root.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading/verifying CIFAR-10 under {data_root} ...")
    train_dataset = datasets.CIFAR10(root=str(data_root), train=True, download=True)
    test_dataset = datasets.CIFAR10(root=str(data_root), train=False, download=True)

    minimum_required = args.num_clients * MIN_SAMPLES_PER_CLIENT
    if minimum_required > len(train_dataset):
        raise ValueError(
            f"--num-clients={args.num_clients} requires at least {minimum_required} training examples "
            f"to keep {MIN_SAMPLES_PER_CLIENT} per client, but CIFAR-10 has {len(train_dataset)}"
        )

    labels = np.asarray(train_dataset.targets)
    partitions = partition_data(
        labels,
        args.num_clients,
        args.alpha,
        args.seed,
        min_samples=MIN_SAMPLES_PER_CLIENT,
    )
    with tempfile.TemporaryDirectory(dir=data_root, prefix=".prepare-") as staging_root_name:
        staging_root = Path(staging_root_name)
        staging_splits_dir = staging_root / SPLITS_DIR
        staging_splits_dir.mkdir()

        expected_files = set()
        site_counts = {}
        class_counts = {}
        for client_index, indices in enumerate(partitions, start=1):
            site_name = f"site-{client_index}"
            filename = f"{site_name}.npy"
            np.save(staging_splits_dir / filename, indices)
            expected_files.add(filename)
            site_counts[site_name] = len(indices)
            classes, counts = np.unique(labels[indices], return_counts=True)
            class_counts[site_name] = {str(class_id): int(count) for class_id, count in zip(classes, counts)}

        manifest = {
            "format_version": 1,
            "dataset": "CIFAR10",
            "num_clients": args.num_clients,
            "dirichlet_alpha": args.alpha,
            "split_seed": args.seed,
            "train_size": len(train_dataset),
            "test_size": len(test_dataset),
            "site_counts": site_counts,
            "class_counts": class_counts,
        }
        staged_manifest_path = staging_root / MANIFEST_FILE
        with staged_manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")

        # The manifest is the validity marker for the complete split generation.
        # Remove the old marker before changing any split and publish the new one last.
        manifest_path.unlink(missing_ok=True)
        for filename in sorted(expected_files):
            os.replace(staging_splits_dir / filename, splits_dir / filename)
        for stale_file in splits_dir.glob("site-*.npy"):
            if stale_file.name not in expected_files:
                stale_file.unlink()
        os.replace(staged_manifest_path, manifest_path)

    print(f"Prepared {args.num_clients} disjoint client partitions under {data_root}")
    print(f"Next: python fedavg.py --data-root {data_root} --num-clients {args.num_clients}")
    return data_root


def main():
    prepare_data(define_parser().parse_args())


if __name__ == "__main__":
    main()
