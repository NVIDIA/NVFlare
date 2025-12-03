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
Feature Election Library for NVIDIA FLARE
High-level API for federated feature selection and training workflow.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class FeatureElection:
    """
    High-level interface for Feature Election in NVIDIA FLARE.
    Simplifies integration with tabular datasets for federated feature selection.

    This class provides:
    - Easy data preparation and splitting
    - Local simulation for testing
    - Result management and persistence

    """

    def __init__(
            self,
            freedom_degree: float = 0.5,
            fs_method: str = "lasso",
            aggregation_mode: str = "weighted",
            auto_tune: bool = False,
            tuning_rounds: int = 5,
    ):
        if not 0 <= freedom_degree <= 1:
            raise ValueError("freedom_degree must be between 0 and 1")
        if aggregation_mode not in ["weighted", "uniform"]:
            raise ValueError("aggregation_mode must be 'weighted' or 'uniform'")

        self.freedom_degree = freedom_degree
        self.fs_method = fs_method
        self.aggregation_mode = aggregation_mode
        self.auto_tune = auto_tune
        self.tuning_rounds = tuning_rounds

        # Storage for results
        self.global_mask = None
        self.selected_feature_names = None
        self.election_stats = {}

    def create_flare_job(
            self,
            job_name: str = "feature_election",
            output_dir: str = "jobs/feature_election",
            min_clients: int = 2,
            num_rounds: int = 5,
            client_sites: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Generate FLARE job configuration.
        """
        job_path = Path(output_dir) / job_name
        job_path.mkdir(parents=True, exist_ok=True)
        (job_path / "app" / "config").mkdir(parents=True, exist_ok=True)
        (job_path / "app" / "custom").mkdir(parents=True, exist_ok=True)

        # Server config
        server_config = {
            "format_version": 2,
            "workflows": [
                {
                    "id": "feature_election_workflow",
                    "path": "nvflare.app_opt.feature_election.controller.FeatureElectionController",
                    "args": {
                        "freedom_degree": self.freedom_degree,
                        "aggregation_mode": self.aggregation_mode,
                        "min_clients": min_clients,
                        "num_rounds": num_rounds,
                        "task_name": "feature_election",
                        "auto_tune": self.auto_tune,
                        "tuning_rounds": self.tuning_rounds,
                    },
                }
            ],
            "components": [],
        }

        # Client config
        client_config = {
            "format_version": 2,
            "executors": [
                {
                    "tasks": ["feature_election"],
                    "executor": {
                        "path": "nvflare.app_opt.feature_election.executor.FeatureElectionExecutor",
                        "args": {
                            "fs_method": self.fs_method,
                            "eval_metric": "f1",
                            "quick_eval": True,
                            "task_name": "feature_election",
                        },
                    },
                }
            ],
            "task_result_filters": [],
            "task_data_filters": [],
        }

        if client_sites is None:
            client_sites = [f"site-{i + 1}" for i in range(min_clients)]

        meta_config = {
            "name": job_name,
            "resource_spec": {site: {"num_of_gpus": 0, "mem_per_gpu_in_GiB": 0} for site in client_sites},
            "min_clients": min_clients,
            "mandatory_clients": [],
            "deploy_map": {"app": ["@ALL"]},
            "task_data_filters": [],
            "task_result_filters": [],
        }

        # Write files
        paths = {
            "server_config": job_path / "app" / "config" / "config_fed_server.json",
            "client_config": job_path / "app" / "config" / "config_fed_client.json",
            "meta": job_path / "meta.json",
            "readme": job_path / "README.md",
        }

        with open(paths["server_config"], "w") as f: json.dump(server_config, f, indent=2)
        with open(paths["client_config"], "w") as f: json.dump(client_config, f, indent=2)
        with open(paths["meta"], "w") as f: json.dump(meta_config, f, indent=2)

        # Create README
        with open(paths["readme"], "w") as f:
            f.write(f"# {job_name}\n\nFeature Election job (Auto-tune: {self.auto_tune})")

        logger.info(f"FLARE job configuration created in {job_path}")
        return {k: str(v) for k, v in paths.items()}

    def prepare_data_splits(
            self,
            df: pd.DataFrame,
            target_col: str,
            num_clients: int = 3,
            split_strategy: str = "stratified",
            split_ratios: Optional[List[float]] = None,
            random_state: int = 42,
    ) -> List[Tuple[pd.DataFrame, pd.Series]]:
        """Prepare data splits for federated clients."""
        X = df.drop(columns=[target_col])
        y = df[target_col]

        if split_ratios is None:
            if num_clients == 2:
                split_ratios = [0.6, 0.4]
            elif num_clients == 3:
                split_ratios = [0.5, 0.3, 0.2]
            else:
                split_ratios = [1.0 / num_clients] * num_clients

        if abs(sum(split_ratios) - 1.0) > 0.001:
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratios)}")

        client_data = []
        indices = np.arange(len(df))

        if split_strategy == "stratified":
            remaining_X, remaining_y, remaining_indices = X, y, indices
            for i in range(num_clients - 1):
                size = split_ratios[i] / sum(split_ratios[i:])
                c_idx, r_idx = train_test_split(
                    remaining_indices, test_size=1 - size, stratify=remaining_y, random_state=random_state + i
                )
                client_data.append((X.iloc[c_idx], y.iloc[c_idx]))
                remaining_indices = r_idx
                remaining_y = y.iloc[remaining_indices]
            client_data.append((X.iloc[remaining_indices], y.iloc[remaining_indices]))

        elif split_strategy == "random":
            np.random.seed(random_state)
            np.random.shuffle(indices)
            start = 0
            for ratio in split_ratios:
                end = start + int(len(indices) * ratio)
                c_idx = indices[start:end]
                client_data.append((X.iloc[c_idx], y.iloc[c_idx]))
                start = end

        elif split_strategy == "dirichlet":
            # Non-IID split logic
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            n_classes = len(le.classes_)
            np.random.seed(random_state)
            label_distribution = np.random.dirichlet([0.5] * num_clients, n_classes)

            client_indices = [[] for _ in range(num_clients)]
            for k in range(n_classes):
                idx_k = np.where(y_encoded == k)[0]
                np.random.shuffle(idx_k)
                proportions = (label_distribution[k] * len(idx_k)).astype(int)[:-1]
                splits = np.split(idx_k, np.cumsum(proportions))
                for i in range(num_clients):
                    if i < len(splits): client_indices[i].extend(splits[i])

            for indices_i in client_indices:
                client_data.append((X.iloc[indices_i], y.iloc[indices_i]))

        else:
            # Fallback for sequential or other
            start = 0
            for ratio in split_ratios:
                end = start + int(len(indices) * ratio)
                c_idx = indices[start:end]
                client_data.append((X.iloc[c_idx], y.iloc[c_idx]))
                start = end

        return client_data

    def simulate_election(
            self,
            client_data: List[Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]],
            feature_names: Optional[List[str]] = None,
    ) -> Dict:
        """Simulate election locally."""
        # Local import to avoid circular dependency
        from .controller import FeatureElectionController
        from .executor import FeatureElectionExecutor

        controller = FeatureElectionController(
            freedom_degree=self.freedom_degree,
            aggregation_mode=self.aggregation_mode,
            min_clients=len(client_data),
            auto_tune=self.auto_tune,
            tuning_rounds=self.tuning_rounds
        )

        client_selections = {}
        for i, (X, y) in enumerate(client_data):
            X_np = X.values if isinstance(X, pd.DataFrame) else X
            y_np = y.values if isinstance(y, pd.Series) else y
            if feature_names is None and isinstance(X, pd.DataFrame):
                feature_names = X.columns.tolist()

            executor = FeatureElectionExecutor(fs_method=self.fs_method, eval_metric="f1")
            executor.set_data(X_np, y_np, feature_names=feature_names)

            # Local Selection
            selected_mask, feature_scores = executor._perform_feature_selection()
            initial_score = executor.evaluate_model(X_np, y_np, X_np, y_np)

            # Apply mask to evaluate
            X_sel = X_np[:, selected_mask]
            fs_score = executor.evaluate_model(X_sel, y_np, X_sel, y_np)

            client_selections[f"client_{i}"] = {
                "selected_features": selected_mask,
                "feature_scores": feature_scores,
                "num_samples": len(X_np),
                "initial_score": initial_score,
                "fs_score": fs_score,
            }

        # Simulate Controller Aggregation
        self.global_mask = controller._aggregate_selections(client_selections)

        # Build Stats
        masks = np.array([sel["selected_features"] for sel in client_selections.values()])
        self.election_stats = {
            "num_clients": len(client_data),
            "num_features_original": len(self.global_mask),
            "num_features_selected": int(np.sum(self.global_mask)),
            "reduction_ratio": 1 - (np.sum(self.global_mask) / len(self.global_mask)),
            "freedom_degree": self.freedom_degree,
            "fs_method": self.fs_method,  # <--- FIXED: Added this missing key
            "auto_tune": self.auto_tune,
            "intersection_features": int(np.sum(np.all(masks, axis=0))),
            "union_features": int(np.sum(np.any(masks, axis=0))),
            "client_stats": client_selections
        }

        if feature_names is not None:
            if len(feature_names) != len(self.global_mask):
                raise ValueError(
                    f"Feature names length ({len(feature_names)}) doesn't match global mask length ({len(self.global_mask)})")
            self.selected_feature_names = [name for i, name in enumerate(feature_names) if self.global_mask[i]]

        return self.election_stats

    def apply_mask(
            self, X: Union[pd.DataFrame, np.ndarray], feature_names: Optional[List[str]] = None
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Apply global feature mask to new data."""
        if self.global_mask is None:
            raise ValueError("No global mask available. Run simulate_election() first.")

        if isinstance(X, pd.DataFrame):
            if self.selected_feature_names:
                return X[self.selected_feature_names]
            return X.iloc[:, self.global_mask]
        return X[:, self.global_mask]

    def save_results(self, filepath: str):
        """Save results to JSON."""
        results = {
            "freedom_degree": self.freedom_degree,
            "fs_method": self.fs_method,
            "aggregation_mode": self.aggregation_mode,
            "auto_tune": self.auto_tune,
            "global_mask": self.global_mask.tolist() if self.global_mask is not None else None,
            "selected_feature_names": self.selected_feature_names,
            "election_stats": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in self.election_stats.items()
                if k != "client_stats"  # Simplified saving for brevity
            },
        }
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

    def load_results(self, filepath: str):
        """Load results from JSON."""
        with open(filepath, "r") as f:
            results = json.load(f)

        self.freedom_degree = results.get("freedom_degree", 0.5)
        self.fs_method = results.get("fs_method", "lasso")
        self.aggregation_mode = results.get("aggregation_mode", "weighted")
        self.auto_tune = results.get("auto_tune", False)

        if results.get("global_mask"):
            self.global_mask = np.array(results["global_mask"])

        self.selected_feature_names = results.get("selected_feature_names")
        self.election_stats = results.get("election_stats", {})


# --- HELPER FUNCTIONS ---

def quick_election(
        df: pd.DataFrame,
        target_col: str,
        num_clients: int = 3,
        freedom_degree: float = 0.5,
        fs_method: str = "lasso",
        split_strategy: str = "stratified",
        **kwargs,
) -> Tuple[np.ndarray, Dict]:
    """
    Quick Feature Election for tabular data (one-line solution).
    """
    # Initialize Feature Election
    fe = FeatureElection(freedom_degree=freedom_degree, fs_method=fs_method, **kwargs)

    # Prepare client data
    client_data = fe.prepare_data_splits(df, target_col, num_clients, split_strategy=split_strategy)

    # Run election
    stats = fe.simulate_election(client_data)

    return fe.global_mask, stats


def load_election_results(filepath: str) -> Dict:
    """
    Load election results from a JSON file.
    """
    with open(filepath, "r") as f:
        results = json.load(f)
    return results