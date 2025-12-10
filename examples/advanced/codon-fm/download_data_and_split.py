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
Download and split the dataset for federated learning.
"""


import argparse
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def download_dataset(data_path, te_dataset_url):
    """Download the dataset if it doesn't exist."""
    # Ensure parent directory exists
    Path(os.path.dirname(data_path)).mkdir(parents=True, exist_ok=True)

    # Download if missing
    if not os.path.exists(data_path):
        print(f"Downloading TE dataset to {data_path} ...")
        urllib.request.urlretrieve(te_dataset_url, data_path)
        print("Download complete.")
    else:
        print(f"Found existing dataset at {data_path}.")

    return data_path


def split_data_federated(
    data_path, num_clients=3, output_dir="/data/federated_data", split_method="uniform", test_size=0.2, random_seed=42
):
    """
    Split data into multiple parts for federated learning.

    Args:
        data_path: Path to the input CSV file
        num_clients: Number of federated clients (data splits)
        output_dir: Directory to save the split data
        split_method: Method for splitting data
            - "uniform": Split data uniformly across clients
            - "random": Randomly distribute samples across clients
            - "stratified": Stratified split (if target column exists)
        test_size: Proportion of data to use for testing (0.0-1.0)
        random_seed: Random seed for reproducibility
    """
    print(f"\n{'=' * 60}")
    print(f"Splitting data for {num_clients} federated clients")
    print(f"Method: {split_method}, Test size: {test_size}")
    print(f"{'=' * 60}\n")

    # Load the dataset
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path, sep="\t")
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Try to find a suitable target column for stratified split
    target_col = None
    for col in ["label", "target", "class", "y"]:
        if col in df.columns:
            target_col = col
            break

    # Split into train and test sets first
    if test_size > 0:
        if target_col and split_method == "stratified":
            try:
                train_df, test_df = train_test_split(
                    df, test_size=test_size, random_state=random_seed, stratify=df[target_col]
                )
                print(f"Performed stratified train-test split on '{target_col}'")
            except (ValueError, KeyError) as e:
                train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_seed)
                print(f"Stratified split failed ({e}), using random split")
        else:
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_seed)
            print("Performed random train-test split")

        # Save global test set
        test_path = os.path.join(output_dir, "test_data.csv")
        test_df.to_csv(test_path, sep="\t", index=False)
        print(f"Saved global test set: {test_path} ({len(test_df)} samples)")
    else:
        train_df = df
        print("No test split performed (test_size=0)")

    # Split training data across clients
    print(f"\nSplitting {len(train_df)} training samples across {num_clients} clients...")

    if split_method == "uniform":
        # Uniform split - each client gets approximately equal data
        indices = np.array_split(np.arange(len(train_df)), num_clients)
        client_dfs = [train_df.iloc[idx].reset_index(drop=True) for idx in indices]

    elif split_method == "random":
        # Random distribution
        np.random.seed(random_seed)
        shuffled_df = train_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        indices = np.array_split(np.arange(len(shuffled_df)), num_clients)
        client_dfs = [shuffled_df.iloc[idx].reset_index(drop=True) for idx in indices]

    elif split_method == "stratified":
        # Stratified split across clients
        if target_col:
            client_dfs = []
            remaining_df = train_df.copy()
            for i in range(num_clients - 1):
                client_size = 1.0 / (num_clients - i)
                try:
                    client_df, remaining_df = train_test_split(
                        remaining_df,
                        test_size=(1 - client_size),
                        random_state=random_seed + i,
                        stratify=remaining_df[target_col],
                    )
                except (ValueError, KeyError) as e:
                    client_df, remaining_df = train_test_split(
                        remaining_df, test_size=(1 - client_size), random_state=random_seed + i
                    )
                client_dfs.append(client_df)
            client_dfs.append(remaining_df)
        else:
            print("Warning: No target column found for stratified split, using uniform split")
            indices = np.array_split(np.arange(len(train_df)), num_clients)
            client_dfs = [train_df.iloc[idx].reset_index(drop=True) for idx in indices]
    else:
        raise ValueError(f"Unknown split_method: {split_method}")

    # Save client data splits
    client_paths = []
    for i, client_df in enumerate(client_dfs):
        client_id = f"site-{i + 1}"  # site-1, site-2, site-3, etc. as per NVFlare site naming convention
        client_dir = os.path.join(output_dir, client_id)
        Path(client_dir).mkdir(parents=True, exist_ok=True)

        client_path = os.path.join(client_dir, "train_data.csv")
        client_df.to_csv(client_path, sep="\t", index=False)
        client_paths.append(client_path)
        print(f"  {client_id}: {client_path} ({len(client_df)} samples)")

    # Generate summary statistics
    print(f"\n{'=' * 60}")
    print("Summary Statistics:")
    print(f"{'=' * 60}")
    print(f"Total samples: {len(df)}")
    if test_size > 0:
        print(f"Test samples: {len(test_df)} ({100 * test_size:.1f}%)")
        print(f"Train samples: {len(train_df)} ({100 * (1 - test_size):.1f}%)")
    print(f"Clients: {num_clients}")
    for i, client_df in enumerate(client_dfs):
        pct = 100 * len(client_df) / len(train_df)
        print(f"  Client {i + 1}: {len(client_df)} samples ({pct:.1f}%)")
    print(f"{'=' * 60}\n")

    # Save metadata
    metadata = {
        "num_clients": num_clients,
        "split_method": split_method,
        "test_size": test_size,
        "random_seed": random_seed,
        "total_samples": len(df),
        "train_samples": len(train_df),
        "test_samples": len(test_df) if test_size > 0 else 0,
        "client_samples": [len(client_df) for client_df in client_dfs],
    }

    metadata_path = os.path.join(output_dir, "split_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {metadata_path}")

    return client_paths


def main():
    parser = argparse.ArgumentParser(description="Download and split data for federated learning")
    parser.add_argument(
        "--data-path",
        type=str,
        default="/data/federated_data/data_with_human_TE_cellline_all_NA_plain.csv",
        help="Path to store/load the dataset",
    )
    parser.add_argument("--num-clients", type=int, default=3, help="Number of federated clients (default: 3)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/federated_data",
        help="Directory to save split data (default: /data/federated_data)",
    )
    parser.add_argument(
        "--split-method",
        type=str,
        choices=["uniform", "random", "stratified"],
        default="uniform",
        help="Method for splitting data across clients (default: uniform)",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of data for testing (default: 0.2)")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--no-split", action="store_true", help="Only download data, don't split it")

    args = parser.parse_args()

    # Source URL for the TE dataset
    te_dataset_url = "https://raw.githubusercontent.com/CenikLab/TE_classic_ML/refs/heads/main/data/data_with_human_TE_cellline_all_NA_plain.csv"

    # Download dataset
    download_dataset(args.data_path, te_dataset_url)

    # Split data for federated learning (unless --no-split is specified)
    if not args.no_split:
        split_data_federated(
            data_path=args.data_path,
            num_clients=args.num_clients,
            output_dir=args.output_dir,
            split_method=args.split_method,
            test_size=args.test_size,
            random_seed=args.random_seed,
        )
    else:
        print("\nSkipping data split (--no-split flag provided)")


if __name__ == "__main__":
    main()
