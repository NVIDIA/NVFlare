#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import json
import os
import time

import numpy as np
from data.cifar10_data_utils import get_site_class_summary, load_cifar10_data


def partition_data(num_sites, alpha, seed):
    np.random.seed(seed)
    train_label = load_cifar10_data()

    min_size = 0
    k_classes = 10
    total_n = train_label.shape[0]
    site_idx = {}

    while min_size < 10:
        idx_batch = [[] for _ in range(num_sites)]
        for k in range(k_classes):
            idx_k = np.where(train_label == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_sites))
            proportions = np.array([p * (len(idx_j) < total_n / num_sites) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min(len(idx_j) for idx_j in idx_batch)

    for j in range(num_sites):
        np.random.shuffle(idx_batch[j])
        site_idx[j] = idx_batch[j]

    class_sum = get_site_class_summary(train_label, site_idx)
    return site_idx, class_sum


def get_split_dir(split_dir_prefix, num_sites, alpha, seed):
    return f"{split_dir_prefix}_{num_sites}sites_alpha{alpha:.2f}_seed{seed}"


def _site_file_name(split_dir, site_num):
    return os.path.join(split_dir, f"site-{site_num}.npy")


def _split_complete(split_dir, num_sites):
    expected_files = [os.path.join(split_dir, "summary.txt")]
    expected_files.extend(_site_file_name(split_dir, site_num) for site_num in range(1, num_sites + 1))
    return all(os.path.isfile(path) and os.path.getsize(path) > 0 for path in expected_files)


def _write_split(split_dir, num_sites, alpha, seed):
    site_idx, class_sum = partition_data(num_sites, alpha, seed)

    os.makedirs(split_dir, exist_ok=True)

    sum_file_name = os.path.join(split_dir, "summary.txt")
    with open(sum_file_name, "w", encoding="utf-8") as sum_file:
        sum_file.write(f"Number of clients: {num_sites}\n")
        sum_file.write(f"Dirichlet sampling parameter: {alpha}\n")
        sum_file.write(f"Seed: {seed}\n")
        sum_file.write("Class counts for each client:\n")
        sum_file.write(json.dumps(class_sum, indent=2))

    for site in range(num_sites):
        np.save(_site_file_name(split_dir, site + 1), np.array(site_idx[site]))


def split_and_save(
    split_dir_prefix,
    num_sites,
    alpha,
    seed=0,
    reuse=True,
    lock_timeout_seconds=600,
    lock_poll_seconds=0.2,
):
    if alpha < 0.0:
        raise ValueError(f"Alpha should be >= 0.0 but was {alpha}")

    split_dir = get_split_dir(split_dir_prefix, num_sites, alpha, seed)
    if reuse and _split_complete(split_dir, num_sites):
        print(f"Reusing CIFAR-10 split: {split_dir}")
        return split_dir

    parent_dir = os.path.dirname(split_dir)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    lock_dir = f"{split_dir}.lock"
    start_time = time.time()
    lock_acquired = False

    while not lock_acquired:
        try:
            os.mkdir(lock_dir)
            lock_acquired = True
        except FileExistsError:
            if reuse and _split_complete(split_dir, num_sites):
                print(f"Reusing CIFAR-10 split: {split_dir}")
                return split_dir
            if time.time() - start_time > lock_timeout_seconds:
                raise TimeoutError(f"Timed out waiting for CIFAR-10 split lock: {lock_dir}")
            time.sleep(lock_poll_seconds)

    try:
        if reuse and _split_complete(split_dir, num_sites):
            print(f"Reusing CIFAR-10 split: {split_dir}")
            return split_dir
        _write_split(split_dir, num_sites, alpha, seed)
    finally:
        os.rmdir(lock_dir)

    return split_dir


def main():
    parser = argparse.ArgumentParser(description="Split CIFAR-10 dataset into multiple sites using Dirichlet sampling.")
    parser.add_argument("--split_dir_prefix", type=str, required=True)
    parser.add_argument("--num_sites", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate the split even if it already exists.",
    )
    args = parser.parse_args()
    split_and_save(
        split_dir_prefix=args.split_dir_prefix,
        num_sites=args.num_sites,
        alpha=args.alpha,
        seed=args.seed,
        reuse=not args.force,
    )


if __name__ == "__main__":
    main()
