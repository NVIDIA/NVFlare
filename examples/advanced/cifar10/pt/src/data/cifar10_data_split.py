#!/usr/bin/env python3

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

# This Dirichlet sampling strategy for creating a heterogeneous partition is adopted
# from FedMA (https://github.com/IBM/FedMA).

# MIT License

# Copyright (c) 2020 International Business Machines

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Executable script to partition CIFAR-10 dataset into multiple sites using Dirichlet sampling.
"""

import argparse
import json
import os

import numpy as np
from data.cifar10_data_utils import get_site_class_summary, load_cifar10_data


def partition_data(num_sites, alpha, seed):
    """
    Partition CIFAR-10 data using Dirichlet sampling.

    Args:
        num_sites: Number of sites to partition data into
        alpha: Dirichlet distribution parameter (controls heterogeneity)
        seed: Random seed for reproducibility

    Returns:
        site_idx: Dictionary mapping site index to list of data indices
        class_sum: Dictionary with class distribution summary for each site
    """
    np.random.seed(seed)

    train_label = load_cifar10_data()

    min_size = 0
    K = 10  # Number of classes in CIFAR-10
    N = train_label.shape[0]
    site_idx = {}

    # Split data using Dirichlet sampling
    while min_size < 10:
        idx_batch = [[] for _ in range(num_sites)]
        # For each class in the dataset
        for k in range(K):
            idx_k = np.where(train_label == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_sites))
            # Balance
            proportions = np.array([p * (len(idx_j) < N / num_sites) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    # Shuffle
    for j in range(num_sites):
        np.random.shuffle(idx_batch[j])
        site_idx[j] = idx_batch[j]

    # Collect class summary
    class_sum = get_site_class_summary(train_label, site_idx)

    return site_idx, class_sum


def split_and_save(split_dir_prefix, num_sites, alpha, seed=0):
    """
    Split CIFAR-10 data and save to disk.

    Args:
        split_dir_prefix: Directory prefix to save split data
        num_sites: Number of sites to partition data into
        alpha: Dirichlet distribution parameter
        seed: Random seed for reproducibility
    """

    split_dir = f"{split_dir_prefix}_{num_sites}sites_alpha{alpha:.2f}_seed{seed}"

    if alpha < 0.0:
        raise ValueError(f"Alpha should be larger or equal 0.0 but was {alpha}!")

    print(f"Partitioning CIFAR-10 dataset into {num_sites} sites with Dirichlet sampling under alpha {alpha}")

    # Partition the data
    site_idx, class_sum = partition_data(num_sites, alpha, seed)

    # Create output directory if it doesn't exist
    if not os.path.isdir(split_dir):
        os.makedirs(split_dir)
        print(f"Created directory: {split_dir}")

    # Write summary file
    sum_file_name = os.path.join(split_dir, "summary.txt")
    with open(sum_file_name, "w") as sum_file:
        sum_file.write(f"Number of clients: {num_sites} \n")
        sum_file.write(f"Dirichlet sampling parameter: {alpha} \n")
        sum_file.write("Class counts for each client: \n")
        sum_file.write(json.dumps(class_sum, indent=2))
    print(f"Saved summary to: {sum_file_name}")

    # Save site data files
    site_file_path = os.path.join(split_dir, "site-")
    for site in range(num_sites):
        site_file_name = site_file_path + str(site + 1) + ".npy"
        np.save(site_file_name, np.array(site_idx[site]))
        print(f"Saved site {site + 1} data ({len(site_idx[site])} samples) to: {site_file_name}")

    print("\nData splitting completed successfully!")
    print("\nClass distribution summary:")
    for site, classes in class_sum.items():
        total_samples = sum(classes.values())
        print(f"  Site {site + 1}: {total_samples} samples - {classes}")

    print(f"Split data saved to: {split_dir}")
    return split_dir


def main():
    parser = argparse.ArgumentParser(
        description="Split CIFAR-10 dataset into multiple sites using Dirichlet sampling for federated learning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--split_dir_prefix",
        type=str,
        required=True,
        help="Path with prefix to directory where split data will be saved",
    )

    parser.add_argument("--num_sites", type=int, default=8, help="Number of sites to partition data into")

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dirichlet distribution parameter (controls data heterogeneity: "
        "lower values create more heterogeneous distributions)",
    )

    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Run the data splitting
    split_and_save(split_dir_prefix=args.split_dir_prefix, num_sites=args.num_sites, alpha=args.alpha, seed=args.seed)


if __name__ == "__main__":
    main()
