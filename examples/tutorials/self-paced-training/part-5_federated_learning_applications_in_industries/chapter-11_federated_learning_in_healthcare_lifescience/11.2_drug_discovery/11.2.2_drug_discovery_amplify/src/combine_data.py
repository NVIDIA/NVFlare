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
Data preparation script for sequence classification.
This script reads multiple CSV files from a directory and combines the 'heavy' and 'light' feature columns
by concatenating them with a '|' separator. All processed data is combined into a single output CSV file.
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data_dirichlet(df, num_clients, alpha, random_seed=42):
    """
    Split data into multiple clients using Dirichlet distribution based on total data length.

    Args:
        df (pd.DataFrame): Input dataframe to split
        num_clients (int): Number of clients to split data into
        alpha (float): Concentration parameter for Dirichlet distribution
        random_seed (int): Random seed for reproducibility

    Returns:
        list: List of dataframes, one for each client
    """
    np.random.seed(random_seed)

    # Get total number of samples
    n_samples = len(df)

    # Generate proportions using Dirichlet distribution
    proportions = np.random.dirichlet([alpha] * num_clients)
    client_sizes = (proportions * n_samples).astype(int)

    # Adjust sizes to ensure all samples are distributed
    remaining = n_samples - sum(client_sizes)
    client_sizes[np.random.choice(num_clients, remaining)] += 1

    # Split and assign data to clients
    client_dfs = []
    start_idx = 0
    for size in client_sizes:
        if size > 0:
            client_dfs.append(df.iloc[start_idx : start_idx + size])
        else:
            client_dfs.append(pd.DataFrame(columns=df.columns))
        start_idx += size

    return client_dfs


def plot_client_distribution(client_dfs, output_dir, alpha):
    """
    Plot the distribution of samples across clients using a bar plot.

    Args:
        client_dfs (list): List of dataframes, one for each client
        output_dir (str): Directory to save the plot
    """
    # Get number of samples for each client
    client_sizes = [len(df) for df in client_dfs]
    client_ids = [f"Client {i + 1}" for i in range(len(client_dfs))]

    # Create bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(client_ids, client_sizes)

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f"{int(height)}", ha="center", va="bottom")

    plt.title("Number of Samples per Client (alpha={alpha})")
    plt.xlabel("Client ID")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_dir, f"client_distribution_alpha_{alpha}.png")
    plt.savefig(plot_path)
    print(f"Client distribution plot saved to {plot_path}")
    plt.close()


def prepare_data(
    input_dir,
    output_dir,
    heavy_col="heavy",
    light_col="light",
    combined_col="combined",
    test_ratio=0.2,
    random_seed=42,
    num_clients=None,
    alpha=None,
):
    """
    Read multiple CSV files from a directory, combine the 'heavy' and 'light' columns with a '|' separator,
    split into training and test sets, and save the processed data.

    Args:
        input_dir (str): Path to the directory containing input CSV files
        output_dir (str): Path to the directory where output CSV files will be saved
        heavy_col (str): Name of the heavy column
        light_col (str): Name of the light column
        combined_col (str): Name for the new combined column
        test_ratio (float): Ratio of data to use for testing (default: 0.2)
        random_seed (int): Random seed for reproducibility (default: 42)
        num_clients (int): Number of clients to split data into (default: None)
        alpha (float): Alpha parameter for Dirichlet distribution (default: None)
    """
    # Check if input directory exists
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {input_dir}")

    print(f"Found {len(csv_files)} CSV files in {input_dir}")

    # Process each CSV file and combine the results
    all_data = []
    n_total = 0

    for i, input_file in enumerate(csv_files):
        print(f"Processing {input_file} ({i + 1}/{len(csv_files)})...")

        # Read the CSV file
        df = pd.read_csv(input_file)

        # Check if required columns exist
        if heavy_col not in df.columns:
            print(f"Warning: Column '{heavy_col}' not found in {input_file}. Skipping this file.")
            continue
        if light_col not in df.columns:
            print(f"Warning: Column '{light_col}' not found in {input_file}. Skipping this file.")
            continue

        # Combine the columns with a '|' separator
        df[combined_col] = df[heavy_col].astype(str) + "|" + df[light_col].astype(str)

        print(f"Added {len(df)} rows...")
        n_total += len(df)

        # Add to the combined dataset
        all_data.append(df)

    if not all_data:
        raise ValueError("No valid data found in any of the CSV files")

    # Combine all processed data
    combined_df = pd.concat(all_data, ignore_index=True)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Split into training and test sets
    print(f"Splitting data into training ({1 - test_ratio:.0%}) and test ({test_ratio:.0%}) sets...")
    train_df, test_df = train_test_split(combined_df, test_size=test_ratio, random_state=random_seed)

    # Save test data
    test_output_file = os.path.join(output_dir, "test_data.csv")
    print(f"Saving test data to {test_output_file}...")
    test_df.to_csv(test_output_file, index=False)
    print(f"Test data saved: {len(test_df)} rows")

    # If num_clients and alpha are provided, split training data using Dirichlet distribution
    if num_clients is not None and alpha is not None:
        print(f"Splitting training data into {num_clients} clients using Dirichlet distribution (alpha={alpha})...")
        client_dfs = split_data_dirichlet(train_df, num_clients, alpha, random_seed)

        # Plot client distribution
        plot_client_distribution(client_dfs, output_dir, alpha)

        # Save each client's data
        for i, client_df in enumerate(client_dfs):
            client_output_file = os.path.join(output_dir, f"client{i + 1}_train_data.csv")
            print(f"Saving client {i + 1} data to {client_output_file}...")
            client_df.to_csv(client_output_file, index=False)
            print(f"Client {i + 1} data saved: {len(client_df)} rows")
    else:
        # Save training data to a single file
        train_output_file = os.path.join(output_dir, "train_data.csv")
        print(f"Saving training data to {train_output_file}...")
        train_df.to_csv(train_output_file, index=False)
        print(f"Training data preparation completed. Output saved {len(train_df)} rows to {train_output_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for sequence classification")
    parser.add_argument("--input_dir", "-i", required=True, help="Path to the directory containing input CSV files")
    parser.add_argument(
        "--output_dir", "-o", required=True, help="Path to the directory where output CSV files will be saved"
    )
    parser.add_argument(
        "--test_ratio", "-t", type=float, default=0.2, help="Ratio of data to use for testing (default: 0.2)"
    )
    parser.add_argument(
        "--random_seed", "-r", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--num_clients", "-n", type=int, default=None, help="Number of clients to split the data into (default: None)"
    )
    parser.add_argument(
        "--alpha", "-a", type=float, default=None, help="Alpha for the Dirichlet distribution (default: None)"
    )
    parser.add_argument(
        "--combined_col",
        "-c",
        type=str,
        default="combined",
        help="Name for the new combined column (default: combined)",
    )

    args = parser.parse_args()

    # Prepare data
    prepare_data(
        args.input_dir,
        args.output_dir,
        combined_col=args.combined_col,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed,
        num_clients=args.num_clients,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
