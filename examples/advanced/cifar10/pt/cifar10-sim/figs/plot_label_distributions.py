#!/usr/bin/env python3
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

"""
Script to visualize label distributions across clients for different alpha values.
Generates stacked bar charts showing how the Dirichlet distribution parameter 
controls data heterogeneity in federated learning.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from data.cifar10_data_split import partition_data
from data.cifar10_data_utils import load_cifar10_data


def compute_label_distribution(num_sites, alpha, seed=0):
    """
    Compute label distribution for each client given alpha parameter.

    Args:
        num_sites: Number of federated learning clients
        alpha: Dirichlet distribution parameter
        seed: Random seed for reproducibility

    Returns:
        distribution_matrix: Array of shape (num_sites, 10) with label counts
        class_summary: Dictionary with class distribution per site
    """
    print(f"Computing label distribution for alpha={alpha}")

    # Partition data using the same method as the training job
    site_idx, class_sum = partition_data(num_sites, alpha, seed)

    # Create distribution matrix
    train_label = load_cifar10_data()
    distribution_matrix = np.zeros((num_sites, 10), dtype=int)

    for site in range(num_sites):
        site_labels = train_label[site_idx[site]]
        for label in range(10):
            distribution_matrix[site, label] = np.sum(site_labels == label)

    return distribution_matrix, class_sum


def plot_label_distribution_bars(distribution_matrix, alpha, ax):
    """
    Plot stacked bar chart showing label distribution across clients.

    Args:
        distribution_matrix: Array of shape (num_sites, 10) with label counts
        alpha: Alpha value used for this distribution
        ax: Matplotlib axis to plot on
    """
    num_sites = distribution_matrix.shape[0]
    clients = np.arange(1, num_sites + 1)

    # Create color palette (using matplotlib's tab10 colormap)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Create stacked bar chart
    bottom = np.zeros(num_sites)
    for label_idx in range(10):
        ax.bar(
            clients,
            distribution_matrix[:, label_idx],
            bottom=bottom,
            label=f"Class {label_idx}",
            color=colors[label_idx],
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += distribution_matrix[:, label_idx]

    ax.set_xlabel("Client ID", fontsize=10)
    ax.set_ylabel("Number of Samples", fontsize=10)
    ax.set_title(f"Alpha = {alpha}", fontsize=12, fontweight="bold")
    ax.set_xticks(clients)
    ax.set_xticklabels(clients)
    ax.grid(axis="y", alpha=0.3, linestyle="--")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize label distributions for different alpha values in federated learning data splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--alpha_values",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 1.0],
        help="List of alpha values to visualize (controls data heterogeneity)",
    )

    parser.add_argument("--n_clients", type=int, default=8, help="Number of federated learning clients")

    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")

    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save output plots")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating label distribution visualizations for alpha values: {args.alpha_values}")
    print(f"Number of clients: {args.n_clients}")
    print(f"Random seed: {args.seed}")

    # Compute distributions for all alpha values
    distributions = []
    class_summaries = []
    for alpha in args.alpha_values:
        dist_matrix, class_sum = compute_label_distribution(args.n_clients, alpha, args.seed)
        distributions.append(dist_matrix)
        class_summaries.append(class_sum)

        # Print summary
        print(f"\nAlpha = {alpha}:")
        for site in range(args.n_clients):
            total = dist_matrix[site].sum()
            print(f"  Client {site + 1}: {total} samples - {class_sum[site]}")

    # Create stacked bar chart visualization
    num_alphas = len(args.alpha_values)

    # Calculate grid dimensions (prefer 2 columns for readability)
    if num_alphas == 1:
        n_rows, n_cols = 1, 1
    elif num_alphas == 2:
        n_rows, n_cols = 1, 2
    elif num_alphas == 3:
        n_rows, n_cols = 1, 3
    else:
        n_cols = 2
        n_rows = (num_alphas + 1) // 2

    fig_width = 8 * n_cols
    fig_height = 5 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    # Handle single subplot case (axes is not an array)
    if num_alphas == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    fig.suptitle(
        "Label Distribution Across Clients",
        fontsize=14,
        fontweight="bold",
    )

    for idx, (alpha, dist_matrix) in enumerate(zip(args.alpha_values, distributions)):
        plot_label_distribution_bars(dist_matrix, alpha, axes[idx])

    # Hide unused subplots if any
    for idx in range(num_alphas, len(axes)):
        axes[idx].set_visible(False)

    # Add legend to the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, -0.02), ncol=10, fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    output_path = os.path.join(args.output_dir, "figs/label_distributions.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved bar chart visualization to: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
