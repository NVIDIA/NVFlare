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

import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_data(
    input_dir, output_dir, heavy_col="heavy", light_col="light", combined_col="combined", test_ratio=0.2, random_seed=42
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
        "--combined_col",
        "-c",
        type=str,
        default="combined",
        help="Name for the new combined column (default: combined)",
    )

    args = parser.parse_args()

    # Prepare data using a uniform split
    prepare_data(
        args.input_dir,
        args.output_dir,
        combined_col=args.combined_col,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
