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

import os
from pathlib import Path
from typing import List, Optional

import pandas as pd


def load_csv_data_from_path(
    data_path: str | list[str],
    data_features: Optional[List[str]] = None,
    na_values=None,
) -> pd.DataFrame:
    """
    Load CSV data from a single file or directory containing multiple CSV files.

    This function handles both single file and directory loading, with comprehensive
    validation to ensure data consistency across multiple CSV files.

    Args:
        data_path (str): Path to a CSV file or directory containing CSV files
        data_features (Optional[List[str]]): List of column names to extract from the CSV files.
            If None, all columns will be loaded.
        na_values (Optional[str]): Values to treat as NaN. Default is None.

    Returns:
        pd.DataFrame: Concatenated DataFrame containing all loaded data

    Raises:
        FileNotFoundError: If the specified path doesn't exist
        ValueError: If no valid CSV files are found or feature consistency issues
    """
    client_name = "client"

    # Check if path is a file or directory
    if not isinstance(data_path, list) and os.path.isfile(data_path):
        # Single file case
        csv_files = [data_path]
        print(f"Loading data from single file: {data_path}")
    elif not isinstance(data_path, list) and os.path.isdir(data_path):
        # Directory case - find all CSV files
        csv_files = sorted(Path(data_path).glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in directory: {data_path}")
        print(f"Loading data from directory: {data_path}")
    elif isinstance(data_path, list):
        csv_files = data_path
        print(f"Loading data from list of files: {csv_files}")
        for file in csv_files:
            if not os.path.isfile(file):
                raise FileNotFoundError(f"File not found: {file}")
            print(f"  - {file}")
    else:
        raise FileNotFoundError(f"Path not found: {data_path}")

    print(f"Found {len(csv_files)} CSV file(s) to process on client {client_name}")

    # Load and validate all CSV files
    dataframes = []
    expected_columns = None

    for csv_file in csv_files:
        try:
            # Load the CSV file
            df = pd.read_csv(csv_file, usecols=data_features, na_values=na_values)

            print(f"Loaded {len(df)} rows from {csv_file}")

            # Validate feature consistency across files
            if expected_columns is None:
                # First file - set the expected column structure
                expected_columns = list(df.columns)
                print(f"Expected columns from first file: {expected_columns}")
            else:
                # Subsequent files - check for consistency
                current_columns = list(df.columns)
                if current_columns != expected_columns:
                    raise ValueError(
                        f"Column mismatch in {csv_file}. " f"Expected: {expected_columns}, Got: {current_columns}"
                    )

                # Check for missing features (only if data_features is specified)
                if data_features is not None:
                    missing_features = [col for col in data_features if col not in df.columns]
                    if missing_features:
                        raise ValueError(f"Missing features in {csv_file}: {missing_features}")

            # Check for empty dataframes
            if len(df) == 0:
                print(f"WARNING: File {csv_file} is empty, skipping")
                continue

            dataframes.append(df)

        except Exception as e:
            print(f"WARNING: Could not load {csv_file}: {e}")
            continue

    if not dataframes:
        raise ValueError("No valid CSV files could be loaded")

    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Successfully concatenated {len(dataframes)} files into {len(combined_df)} total rows")

    # Final validation
    if len(combined_df) == 0:
        raise ValueError("Combined dataset is empty")

    # Log summary statistics
    print(f"Final dataset shape: {combined_df.shape} on client {client_name}")
    print(f"Columns: {list(combined_df.columns)}")

    return combined_df


def validate_data_features(df: pd.DataFrame, data_features: Optional[List[str]]) -> None:
    """
    Validate that all required features exist in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to validate
        data_features (Optional[List[str]]): List of required feature names. If None, validation is skipped.

    Raises:
        ValueError: If any required features are missing
    """
    # this is more for type checking and to avoid silent issues if data_features is accidentally None
    assert data_features is not None, "Data features list must be provided for validation"

    missing_features = [col for col in data_features if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    # Check for completely empty features
    empty_features = []
    for feature in data_features:
        if df[feature].isna().all():
            empty_features.append(feature)

    if empty_features:
        raise ValueError(f"Features with all NaN values: {empty_features}")


def print_directory_tree(path, prefix="", max_depth=3, current_depth=0, dirs_only=False, endswith=None):
    """
    Print directory tree structure.

    Args:
        path: Path to the directory to display
        prefix: Prefix string for tree formatting (used internally for recursion)
        max_depth: Maximum depth to traverse (default: 3)
        current_depth: Current depth in the recursion (used internally)
        dirs_only: If True, only show directories
        endswith: If provided, only show directories and files ending with this string (e.g., '.csv', '.txt')
    """
    if current_depth >= max_depth:
        return

    if not os.path.exists(path):
        print(f"{prefix}[Path does not exist: {path}]")
        return

    try:
        items = sorted(os.listdir(path))
        if dirs_only:
            items = [item for item in items if os.path.isdir(os.path.join(path, item))]
        elif endswith:
            items = [item for item in items if os.path.isdir(os.path.join(path, item)) or item.endswith(endswith)]
        for i, item in enumerate(items):
            item_path = os.path.join(path, item)
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item}")

            if os.path.isdir(item_path):
                next_prefix = prefix + ("    " if is_last else "│   ")
                print_directory_tree(
                    item_path,
                    next_prefix,
                    max_depth,
                    current_depth + 1,
                    dirs_only,
                    endswith,
                )
    except PermissionError:
        print(f"{prefix}[Permission denied]")
