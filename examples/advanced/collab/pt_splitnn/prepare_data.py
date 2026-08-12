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

"""Prepare the vertical CIFAR-10 split and aligned PSI artifacts for SplitNN."""

import argparse
import json
from pathlib import Path

import numpy as np
from local_psi import Cifar10LocalPSI
from torchvision import datasets

from nvflare.app_common.psi.recipes.dh_psi import DhPSIRecipe
from nvflare.recipe import SimEnv

DEFAULT_DATASET_ROOT = "/tmp/cifar10"
DEFAULT_SPLIT_DIR = "/tmp/cifar10_vert_splits"
DEFAULT_PSI_WORKSPACE = "/tmp/nvflare/cifar10_psi"
SITE_NAMES = ("site-1", "site-2")


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare CIFAR-10 and PSI artifacts for Collab SplitNN")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, help="CIFAR-10 download/cache directory")
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR, help="Output directory for per-site sample IDs")
    parser.add_argument(
        "--psi-workspace",
        default=DEFAULT_PSI_WORKSPACE,
        help="Simulation workspace that will contain the per-site intersection files",
    )
    parser.add_argument("--overlap", type=int, default=10_000, help="Number of samples shared by both sites")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the vertical split")
    return parser


def _prepare_vertical_split(dataset_root: Path, split_dir: Path, overlap: int, seed: int) -> None:
    print(f"Downloading/verifying CIFAR-10 under {dataset_root} ...")
    train_dataset = datasets.CIFAR10(root=str(dataset_root), train=True, download=True)
    datasets.CIFAR10(root=str(dataset_root), train=False, download=True)

    num_samples = len(train_dataset)
    if not 0 < overlap <= num_samples:
        raise ValueError(f"--overlap must be between 1 and {num_samples}, got {overlap}")

    rng = np.random.RandomState(seed)
    sample_indices = np.arange(num_samples)
    overlap_indices = rng.choice(sample_indices, size=overlap, replace=False)
    remaining_indices = np.setdiff1d(sample_indices, overlap_indices, assume_unique=True)

    # Site 2's non-overlapping IDs are offset so PSI reveals only overlap_indices.
    site_indices = {
        "overlap": overlap_indices,
        "site-1": np.concatenate((overlap_indices, remaining_indices)),
        "site-2": np.concatenate((overlap_indices, remaining_indices + num_samples)),
    }
    rng.shuffle(site_indices["site-1"])
    rng.shuffle(site_indices["site-2"])

    split_dir.mkdir(parents=True, exist_ok=True)
    for name, indices in site_indices.items():
        np.save(split_dir / f"{name}.npy", indices)

    labels = np.asarray(train_dataset.targets)
    classes, counts = np.unique(labels[overlap_indices], return_counts=True)
    class_summary = {int(label): int(count) for label, count in zip(classes, counts)}
    with (split_dir / "summary.txt").open("w", encoding="utf-8") as stream:
        stream.write("Class counts for overlap: \n")
        json.dump({"overlap": class_summary}, stream)

    print(f"Prepared vertical split written to {split_dir}")


def _run_psi(split_dir: Path, psi_workspace: Path) -> None:
    recipe = DhPSIRecipe(
        name=psi_workspace.name,
        min_clients=len(SITE_NAMES),
        local_psi=Cifar10LocalPSI(split_dir=str(split_dir)),
    )
    env = SimEnv(clients=list(SITE_NAMES), workspace_root=str(psi_workspace.parent), log_config="ERROR")
    run = recipe.execute(env)
    print(f"PSI artifacts written to {run.get_result()}")


def prepare_data(args) -> None:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    split_dir = Path(args.split_dir).expanduser().resolve()
    psi_workspace = Path(args.psi_workspace).expanduser().resolve()
    if psi_workspace.parent == psi_workspace:
        raise ValueError("--psi-workspace must include a job directory name")

    _prepare_vertical_split(dataset_root, split_dir, args.overlap, args.seed)
    _run_psi(split_dir, psi_workspace)
    print(f"Next: python job.py --dataset-root {dataset_root} --psi-workspace {psi_workspace}")


def main() -> None:
    prepare_data(define_parser().parse_args())


if __name__ == "__main__":
    main()
