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
        eval_metric: str = "f1",
        wait_time_after_min_received: int = 10,
        fs_params: Optional[Dict] = None,
    ):
        if not 0 <= freedom_degree <= 1:
            raise ValueError("freedom_degree must be between 0 and 1")
        if aggregation_mode not in ["weighted", "uniform"]:
            raise ValueError("aggregation_mode must be 'weighted' or 'uniform'")
        if eval_metric not in ["f1", "accuracy"]:
            raise ValueError("eval_metric must be 'f1' or 'accuracy'")

        self.freedom_degree = freedom_degree
        self.fs_method = fs_method
        self.aggregation_mode = aggregation_mode
        self.auto_tune = auto_tune
        self.tuning_rounds = tuning_rounds
        self.eval_metric = eval_metric
        self.wait_time_after_min_received = wait_time_after_min_received
        # FS hyperparameters (e.g. {"alpha": 0.1} for Lasso) forwarded to the
        # executor; None means the executor uses its own defaults.
        self.fs_params = fs_params or {}

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
                        "wait_time_after_min_received": self.wait_time_after_min_received,
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
                            "eval_metric": self.eval_metric,
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

        with open(paths["server_config"], "w") as f:
            json.dump(server_config, f, indent=2)
        with open(paths["client_config"], "w") as f:
            json.dump(client_config, f, indent=2)
        with open(paths["meta"], "w") as f:
            json.dump(meta_config, f, indent=2)

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
        dirichlet_alpha: float = 0.5,
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
        if len(split_ratios) != num_clients:
            raise ValueError(f"len(split_ratios) ({len(split_ratios)}) must equal num_clients ({num_clients})")

        # Accept "non_iid" as an alias for "dirichlet" so callers who learn the
        # prepare_data.py / job.py CLI convention ("non_iid") get the same result
        # without a confusing ValueError.
        if split_strategy == "non_iid":
            split_strategy = "dirichlet"

        client_data = []
        indices = np.arange(len(df))

        if split_strategy == "stratified":
            remaining_y, remaining_indices = y, indices
            for i in range(num_clients - 1):
                size = split_ratios[i] / sum(split_ratios[i:])
                try:
                    c_idx, r_idx = train_test_split(
                        remaining_indices, test_size=1 - size, stratify=remaining_y, random_state=random_state + i
                    )
                except ValueError as e:
                    # Only fall back to non-stratified splitting for sklearn's own
                    # stratification errors (e.g. a class with fewer than 2 samples).
                    # Any other ValueError — such as test_size=0 from a zero ratio —
                    # should propagate so the caller gets a meaningful error message.
                    if "stratif" not in str(e).lower() and "least populated" not in str(e).lower():
                        raise
                    c_idx, r_idx = train_test_split(
                        remaining_indices, test_size=1 - size, random_state=random_state + i
                    )
                client_data.append((X.iloc[c_idx], y.iloc[c_idx]))
                remaining_indices = r_idx
                remaining_y = y.iloc[remaining_indices]
            client_data.append((X.iloc[remaining_indices], y.iloc[remaining_indices]))

        elif split_strategy == "random":
            np.random.seed(random_state)
            np.random.shuffle(indices)
            start = 0
            for i, ratio in enumerate(split_ratios):
                if i == len(split_ratios) - 1:
                    c_idx = indices[start:]  # last client gets all remaining
                else:
                    end = start + int(len(indices) * ratio)
                    c_idx = indices[start:end]
                    start = end
                if len(c_idx) == 0:
                    raise ValueError(
                        f"Client {i} received 0 samples from random split. "
                        "Increase the dataset size or adjust split_ratios."
                    )
                client_data.append((X.iloc[c_idx], y.iloc[c_idx]))

        elif split_strategy == "dirichlet":
            # Non-IID split logic
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            n_classes = len(le.classes_)
            np.random.seed(random_state)
            label_distribution = np.random.dirichlet([dirichlet_alpha] * num_clients, n_classes)

            client_indices = [[] for _ in range(num_clients)]
            for k in range(n_classes):
                idx_k = np.where(y_encoded == k)[0]
                np.random.shuffle(idx_k)
                proportions = (label_distribution[k] * len(idx_k)).astype(int)
                # Assign any rounding remainder to the last client so that every
                # sample in this class is distributed exactly once.  This matches
                # the convention used in prepare_data.py and the random/sequential
                # split strategies above (last client takes indices[start:]).
                total_assigned = proportions[:-1].sum()
                proportions[-1] = max(0, len(idx_k) - total_assigned)

                start = 0
                for i, prop in enumerate(proportions):
                    client_indices[i].extend(idx_k[start : start + prop])
                    start += prop

            for i, indices_i in enumerate(client_indices):
                if len(indices_i) == 0:
                    raise ValueError(
                        f"Client {i} received 0 samples from Dirichlet split (alpha={dirichlet_alpha}). "
                        "Increase the dataset size or reduce the number of clients."
                    )
                client_data.append((X.iloc[indices_i], y.iloc[indices_i]))

        else:
            # Fallback for sequential or other
            start = 0
            for i, ratio in enumerate(split_ratios):
                if i == len(split_ratios) - 1:
                    c_idx = indices[start:]  # last client gets all remaining
                else:
                    end = start + int(len(indices) * ratio)
                    c_idx = indices[start:end]
                    start = end
                if len(c_idx) == 0:
                    raise ValueError(
                        f"Client {i} received 0 samples from sequential split. "
                        "Increase the dataset size or adjust split_ratios."
                    )
                client_data.append((X.iloc[c_idx], y.iloc[c_idx]))

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
            tuning_rounds=self.tuning_rounds,
            wait_time_after_min_received=self.wait_time_after_min_received,
        )

        # Accumulate feature names from the first DataFrame client encountered;
        # validate all subsequent DataFrame clients against that reference.
        # Using a separate local variable avoids in-loop reassignment of the
        # parameter, making the capture-once intent explicit.
        resolved_feature_names = feature_names
        n_features_ref = None  # established from the first client; all others must match
        client_selections = {}
        executors = []
        for i, (X, y) in enumerate(client_data):
            X_np = X.values if isinstance(X, pd.DataFrame) else X
            y_np = y.values if isinstance(y, pd.Series) else y

            # Validate feature count consistency across all clients (DataFrame and ndarray).
            if n_features_ref is None:
                n_features_ref = X_np.shape[1]
            elif X_np.shape[1] != n_features_ref:
                raise ValueError(
                    f"Client {i} has {X_np.shape[1]} features but client 0 has {n_features_ref}. "
                    "All clients must have the same number of features."
                )

            if isinstance(X, pd.DataFrame):
                client_cols = X.columns.tolist()
                if resolved_feature_names is None:
                    resolved_feature_names = client_cols
                elif client_cols != resolved_feature_names:
                    raise ValueError(
                        f"Client {i} has different column labels than client 0. "
                        f"Expected {len(resolved_feature_names)} columns ({resolved_feature_names[:3]}...), "
                        f"got {len(client_cols)} ({client_cols[:3]}...). "
                        "All DataFrame clients must have identical feature columns."
                    )

            # Split into train/val so tuning scores are not evaluated on training data.
            # Attempt stratified split so minority classes appear in both halves (mirrors
            # _safe_train_test_split in prepare_data.py); fall back to random if any class
            # has fewer than 2 samples (e.g. after a Dirichlet split).
            try:
                X_train_sim, X_val_sim, y_train_sim, y_val_sim = train_test_split(
                    X_np, y_np, test_size=0.2, random_state=42 + i, stratify=y_np
                )
            except ValueError:
                X_train_sim, X_val_sim, y_train_sim, y_val_sim = train_test_split(
                    X_np, y_np, test_size=0.2, random_state=42 + i
                )

            executor = FeatureElectionExecutor(
                fs_method=self.fs_method, eval_metric=self.eval_metric, fs_params=self.fs_params
            )
            executor.set_data(
                X_train_sim, y_train_sim, X_val=X_val_sim, y_val=y_val_sim, feature_names=resolved_feature_names
            )
            executors.append(executor)

            # Local Selection

            try:
                selected_mask, feature_scores = executor.perform_feature_selection()
            except (TypeError, ValueError) as e:
                raise RuntimeError(f"Feature selection returned unexpected format: {e}")

            if not np.any(selected_mask):
                logger.warning(
                    f"Client {i}: feature selection rejected all features "
                    f"(fs_method='{self.fs_method}', fs_params={self.fs_params}). "
                    "The global mask may be all-False. Consider lowering the regularisation "
                    "strength (e.g. reduce 'alpha' for Lasso/ElasticNet)."
                )

            initial_score = executor.evaluate_model(X_train_sim, y_train_sim, X_val_sim, y_val_sim)

            # Apply mask to evaluate on held-out val set
            X_sel_tr = X_train_sim[:, selected_mask]
            X_sel_val = X_val_sim[:, selected_mask]
            fs_score = executor.evaluate_model(X_sel_tr, y_train_sim, X_sel_val, y_val_sim)

            client_selections[f"client_{i}"] = {
                "selected_features": selected_mask.tolist(),
                "feature_scores": feature_scores.tolist(),
                # Use the train-split size so this matches what the real executor
                # reports via _handle_feature_selection (len(self.X_train)), keeping
                # mask-aggregation weights and tuning-score weights consistent.
                "num_samples": len(X_train_sim),
                "initial_score": float(initial_score),
                "fs_score": float(fs_score),
            }

        # Simulate Controller Aggregation with optional auto-tuning.
        # Use controller.tuning_rounds (not self.tuning_rounds) so that edge-case
        # normalisation performed by FeatureElectionController.__init__ (e.g. the
        # tuning_rounds=1 no-op guard) is respected and simulation stays in sync
        # with what the real FL deployment would do.
        if self.auto_tune and controller.tuning_rounds > 0:
            logger.info(f"Starting local auto-tuning ({controller.tuning_rounds} rounds)...")

            for t in range(controller.tuning_rounds):
                # Generate mask at current freedom_degree
                candidate_mask = controller.aggregate_selections(client_selections)

                # Evaluate across all clients
                if np.sum(candidate_mask) == 0:
                    score = 0.0
                else:
                    # Use the same weighting as the real controller so the tuning
                    # objective is consistent with the actual aggregation_mode.
                    weighted_score, total_weight = 0.0, 0.0
                    for i_exec, exec_i in enumerate(executors):
                        if exec_i.X_train.shape[1] != len(candidate_mask):
                            raise ValueError(
                                f"Executor {i_exec} X_train has {exec_i.X_train.shape[1]} features "
                                f"but candidate_mask has {len(candidate_mask)} entries — "
                                "X_train must remain full-width throughout the tuning loop"
                            )
                        X_masked = exec_i.X_train[:, candidate_mask]
                        X_val_masked = exec_i.X_val[:, candidate_mask]
                        s = exec_i.evaluate_model(X_masked, exec_i.y_train, X_val_masked, exec_i.y_val)
                        n = len(exec_i.X_train) if self.aggregation_mode == "weighted" else 1
                        weighted_score += s * n
                        total_weight += n
                    score = weighted_score / total_weight if total_weight > 0 else 0.0

                logger.info(
                    f"Tuning Round {t + 1}/{controller.tuning_rounds}: "
                    f"FD={controller.freedom_degree:.4f} -> Score={score:.4f}"
                )
                # Append to controller's own history so _calculate_next_fd has the right state
                controller.tuning_history.append((controller.freedom_degree, score))

                if t < controller.tuning_rounds - 1:
                    controller.freedom_degree = controller._calculate_next_fd(first_step=(t == 0))

            # Select best FD
            if controller.tuning_history:
                best_fd, best_score = max(controller.tuning_history, key=lambda x: x[1])
                controller.freedom_degree = best_fd
                self.freedom_degree = best_fd
                logger.info(f"Tuning Complete. Optimal Freedom Degree: {best_fd:.4f} (Score: {best_score:.4f})")

        self.global_mask = controller.aggregate_selections(client_selections)

        # Build Stats
        masks = np.array([sel["selected_features"] for sel in client_selections.values()])
        self.election_stats = {
            "num_clients": len(client_data),
            "num_features_original": len(self.global_mask),
            "num_features_selected": int(np.sum(self.global_mask)),
            "reduction_ratio": float(1 - (np.sum(self.global_mask) / len(self.global_mask))),
            "freedom_degree": float(self.freedom_degree),
            "fs_method": self.fs_method,
            "auto_tune": self.auto_tune,
            "tuning_history": (
                [(float(fd), float(s)) for fd, s in controller.tuning_history]
                if self.auto_tune and controller.tuning_rounds > 0
                else []
            ),
            "intersection_features": int(np.sum(np.all(masks, axis=0))),
            "union_features": int(np.sum(np.any(masks, axis=0))),
            "client_stats": client_selections,
        }

        if resolved_feature_names is not None:
            if len(resolved_feature_names) != len(self.global_mask):
                raise ValueError(
                    f"Feature names length ({len(resolved_feature_names)}) doesn't match global mask length ({len(self.global_mask)})"
                )
            self.selected_feature_names = [name for i, name in enumerate(resolved_feature_names) if self.global_mask[i]]

        return self.election_stats

    def apply_mask(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Apply global feature mask to new data."""
        if self.global_mask is None:
            raise ValueError("No global mask available. Run simulate_election() first.")

        if isinstance(X, pd.DataFrame):
            if self.selected_feature_names:
                return X[self.selected_feature_names]
            # Convert boolean mask to integer indices for iloc
            selected_indices = np.where(self.global_mask)[0]
            return X.iloc[:, selected_indices]
        return X[:, self.global_mask]

    def save_results(self, filepath: str):
        """Save results to JSON."""
        results = {
            "freedom_degree": float(self.freedom_degree),
            "fs_method": self.fs_method,
            "aggregation_mode": self.aggregation_mode,
            "auto_tune": self.auto_tune,
            "eval_metric": self.eval_metric,
            "global_mask": self.global_mask.tolist() if self.global_mask is not None else None,
            "selected_feature_names": self.selected_feature_names,
            "election_stats": {
                k: (
                    v.tolist()
                    if isinstance(v, np.ndarray)
                    else int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v
                )
                for k, v in self.election_stats.items()
                if k
                != "client_stats"  # client_stats may contain arbitrary numpy types; excluded from persistence intentionally
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
        self.eval_metric = results.get("eval_metric", "f1")

        if results.get("global_mask") is not None:
            self.global_mask = np.array(results["global_mask"])

        self.selected_feature_names = results.get("selected_feature_names")
        self.election_stats = results.get("election_stats", {})


# --- HELPER FUNCTIONS ---


_FEATURE_ELECTION_INIT_PARAMS = {
    "aggregation_mode",
    "auto_tune",
    "tuning_rounds",
    "eval_metric",
    "wait_time_after_min_received",
    "fs_params",
}

_PREPARE_DATA_PARAMS = {
    "split_ratios",
    "random_state",
    "dirichlet_alpha",
}


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

    ``**kwargs`` are routed to either :class:`FeatureElection` or
    :meth:`FeatureElection.prepare_data_splits` based on the parameter name.
    Recognised split parameters: ``split_ratios``, ``random_state``,
    ``dirichlet_alpha``.  All other kwargs are forwarded to
    :class:`FeatureElection` (e.g. ``aggregation_mode``, ``auto_tune``,
    ``fs_params``).
    """
    init_kwargs = {k: v for k, v in kwargs.items() if k not in _PREPARE_DATA_PARAMS}
    split_kwargs = {k: v for k, v in kwargs.items() if k in _PREPARE_DATA_PARAMS}

    unknown = set(kwargs) - _FEATURE_ELECTION_INIT_PARAMS - _PREPARE_DATA_PARAMS
    if unknown:
        raise TypeError(f"quick_election() got unexpected keyword argument(s): {sorted(unknown)}")

    fe = FeatureElection(freedom_degree=freedom_degree, fs_method=fs_method, **init_kwargs)
    client_data = fe.prepare_data_splits(df, target_col, num_clients, split_strategy=split_strategy, **split_kwargs)
    stats = fe.simulate_election(client_data)
    return fe.global_mask, stats


def load_election_results(filepath: str) -> Dict:
    """
    Load election results from a JSON file.
    """
    with open(filepath, "r") as f:
        results = json.load(f)
    return results
