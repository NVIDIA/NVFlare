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
Data preparation utilities for Feature Election example.

Creates synthetic high-dimensional datasets for testing federated
feature selection across multiple clients.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import json
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def create_synthetic_dataset(
    n_samples: int = 1000,
    n_features: int = 100,
    n_informative: int = 20,
    n_redundant: int = 30,
    n_repeated: int = 10,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create a synthetic high-dimensional dataset for testing.

    Args:
        n_samples: Number of samples
        n_features: Total number of features
        n_informative: Number of informative features
        n_redundant: Number of redundant features
        n_repeated: Number of repeated features
        random_state: Random seed

    Returns:
        Tuple of (DataFrame with features and target, feature names)
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        n_clusters_per_class=2,
        flip_y=0.01,
        random_state=random_state,
    )

    feature_names = [f"feature_{i:03d}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    logger.info(
        f"Created synthetic dataset: {n_samples} samples, {n_features} features "
        f"({n_informative} informative, {n_redundant} redundant, {n_repeated} repeated)"
    )
    return df, feature_names


def split_data_for_clients(
    df: pd.DataFrame,
    num_clients: int,
    strategy: str = "stratified",
    random_state: int = 42,
) -> List[pd.DataFrame]:
    """
    Split dataset across multiple clients.

    Args:
        df: Full dataset with 'target' column
        num_clients: Number of clients to split data for
        strategy: Splitting strategy ('stratified', 'random', 'non_iid')
        random_state: Random seed

    Returns:
        List of DataFrames, one per client
    """
    if strategy == "stratified":
        return _split_stratified(df, num_clients, random_state)
    elif strategy == "random":
        return _split_random(df, num_clients, random_state)
    elif strategy == "non_iid":
        return _split_non_iid(df, num_clients, random_state)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _split_stratified(df: pd.DataFrame, num_clients: int, random_state: int) -> List[pd.DataFrame]:
    """Stratified split maintaining class distribution across clients."""
    y = df["target"].values

    # Use iterative stratified splitting to maintain class distribution
    from sklearn.model_selection import StratifiedKFold

    # If we can't use StratifiedKFold (fewer samples than clients), fall back to simple split
    if len(df) < num_clients:
        logger.warning(f"Not enough samples ({len(df)}) for {num_clients} clients. Using simple split.")
        return _split_random(df, num_clients, random_state)

    # For small client counts, use direct stratified splitting
    if num_clients == 2:
        indices = np.arange(len(df))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.5, random_state=random_state, stratify=y
        )
        return [df.iloc[train_idx].copy(), df.iloc[test_idx].copy()]

    # For more clients, use iterative approach
    client_dfs = []
    remaining_df = df.copy()
    remaining_indices = np.arange(len(df))

    for i in range(num_clients - 1):
        # Calculate target size for this client
        samples_remaining = len(remaining_df)
        clients_remaining = num_clients - i
        target_size = samples_remaining // clients_remaining
        test_size = max(0.01, min(0.99, target_size / samples_remaining))

        try:
            # Try stratified split
            train_idx, client_idx = train_test_split(
                np.arange(len(remaining_df)),
                test_size=test_size,
                random_state=random_state + i,
                stratify=remaining_df["target"].values
            )
            client_dfs.append(remaining_df.iloc[client_idx].copy())
            remaining_df = remaining_df.iloc[train_idx].reset_index(drop=True)
        except ValueError:
            # If stratification fails, use random split
            logger.warning(f"Stratification failed for client {i}, using random split")
            indices = np.arange(len(remaining_df))
            np.random.seed(random_state + i)
            np.random.shuffle(indices)
            split_point = int(len(indices) * test_size)
            client_idx = indices[:split_point]
            train_idx = indices[split_point:]
            client_dfs.append(remaining_df.iloc[client_idx].copy())
            remaining_df = remaining_df.iloc[train_idx].reset_index(drop=True)

    # Add remaining data as last client
    client_dfs.append(remaining_df)

    return client_dfs


def _split_random(df: pd.DataFrame, num_clients: int, random_state: int) -> List[pd.DataFrame]:
    """Random split without stratification."""
    np.random.seed(random_state)
    indices = np.arange(len(df))
    np.random.shuffle(indices)

    client_dfs = []
    samples_per_client = len(df) // num_clients

    for i in range(num_clients):
        start = i * samples_per_client
        end = start + samples_per_client if i < num_clients - 1 else len(df)
        client_dfs.append(df.iloc[indices[start:end]].copy())

    return client_dfs


def _split_non_iid(
    df: pd.DataFrame,
    num_clients: int,
    random_state: int,
    alpha: float = 0.5,
) -> List[pd.DataFrame]:
    """
    Non-IID split using Dirichlet distribution.

    Creates heterogeneous data distributions across clients,
    simulating real-world federated scenarios.
    """
    y = df["target"].values

    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)

    num_classes = len(np.unique(y))
    np.random.seed(random_state)

    # Dirichlet distribution for label assignment
    label_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)

    client_indices = [[] for _ in range(num_clients)]

    for k in range(num_classes):
        idx_k = np.where(y == k)[0]
        np.random.shuffle(idx_k)

        # Split indices according to Dirichlet proportions
        proportions = (label_distribution[k] * len(idx_k)).astype(int)
        proportions[-1] = len(idx_k) - proportions[:-1].sum()  # Ensure all assigned

        start = 0
        for i, prop in enumerate(proportions):
            client_indices[i].extend(idx_k[start: start + prop])
            start += prop

    return [df.iloc[indices].copy() for indices in client_indices]


def load_client_data(
    client_id: int,
    num_clients: int,
    data_root: Optional[str] = None,
    split_strategy: str = "stratified",
    test_size: float = 0.2,
    random_state: int = 42,
    n_samples: int = 1000,
    n_features: int = 100,
    n_informative: int = 20,
    n_redundant: int = 30,
    n_repeated: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load or generate data for a specific client.

    Args:
        client_id: Client identifier (0 to num_clients-1)
        num_clients: Total number of clients
        data_root: Optional path to pre-generated data
        split_strategy: Data splitting strategy
        test_size: Fraction of data for validation
        random_state: Random seed
        n_samples: Total samples (will be divided among clients)
        n_features: Number of features
        n_informative: Number of informative features
        n_redundant: Number of redundant features
        n_repeated: Number of repeated features

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, feature_names)
    """
    # Check for pre-generated data
    if data_root:
        data_path = Path(data_root) / f"client_{client_id}.csv"
        if data_path.exists():
            logger.info(f"Loading pre-generated data from {data_path}")
            df = pd.read_csv(data_path)
            feature_names = [c for c in df.columns if c != "target"]
            X = df.drop(columns=["target"]).values
            y = df["target"].values

            # Check if stratification is possible (all classes must have at least 2 samples)
            unique, counts = np.unique(y, return_counts=True)
            can_stratify = np.all(counts >= 2)

            if can_stratify:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=test_size, random_state=random_state + client_id, stratify=y
                )
            else:
                logger.warning(
                    f"Client {client_id}: Cannot stratify pre-generated data (some classes have <2 samples). "
                    f"Using random split instead. Class distribution: {dict(zip(unique, counts))}"
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=test_size, random_state=random_state + client_id
                )
            return X_train, y_train, X_val, y_val, feature_names

    # Generate synthetic data
    total_samples = n_samples * num_clients
    df, feature_names = create_synthetic_dataset(
        n_samples=total_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        random_state=random_state,
    )

    # Split among clients
    client_dfs = split_data_for_clients(df, num_clients, split_strategy, random_state)
    client_df = client_dfs[client_id]

    # Separate features and target
    X = client_df.drop(columns=["target"]).values
    y = client_df["target"].values

    # Train/validation split
    # Check if stratification is possible (all classes must have at least 2 samples)
    unique, counts = np.unique(y, return_counts=True)
    can_stratify = np.all(counts >= 2)

    if can_stratify:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state + client_id, stratify=y
        )
    else:
        logger.warning(
            f"Client {client_id}: Cannot stratify (some classes have <2 samples). "
            f"Using random split instead. Class distribution: {dict(zip(unique, counts))}"
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state + client_id
        )

    logger.info(f"Client {client_id}: {len(X_train)} train samples, {len(X_val)} val samples")

    return X_train, y_train, X_val, y_val, feature_names


def prepare_data_for_all_clients(
    output_dir: str,
    num_clients: int = 3,
    split_strategy: str = "stratified",
    random_state: int = 42,
    **dataset_kwargs,
) -> None:
    """
    Pre-generate and save data for all clients.

    Args:
        output_dir: Directory to save client data files
        num_clients: Number of clients
        split_strategy: Data splitting strategy
        random_state: Random seed
        **dataset_kwargs: Additional arguments for create_synthetic_dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate full dataset
    total_samples = dataset_kwargs.pop("n_samples", 1000) * num_clients
    df, feature_names = create_synthetic_dataset(
        n_samples=total_samples,
        random_state=random_state,
        **dataset_kwargs,
    )

    # Split and save
    client_dfs = split_data_for_clients(df, num_clients, split_strategy, random_state)

    for i, client_df in enumerate(client_dfs):
        filepath = output_path / f"client_{i}.csv"
        client_df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(client_df)} samples to {filepath}")

    # Save metadata
    metadata = {
        "num_clients": num_clients,
        "split_strategy": split_strategy,
        "random_state": random_state,
        "feature_names": feature_names,
        "total_samples": total_samples,
    }

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Data preparation complete. Files saved to {output_path}")


if __name__ == "__main__":
    # Example: Generate data for 3 clients
    logging.basicConfig(level=logging.INFO)

    prepare_data_for_all_clients(
        output_dir="./data",
        num_clients=3,
        split_strategy="stratified",
        n_samples=1000,
        n_features=100,
        n_informative=20,
        n_redundant=30,
        n_repeated=10,
    )
