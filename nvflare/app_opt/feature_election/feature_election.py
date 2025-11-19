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
High-level API for federated feature selection on tabular datasets
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class FeatureElection:
    """
    High-level interface for Feature Election in NVIDIA FLARE.
    Simplifies integration with tabular datasets for federated feature selection.
    
    This class provides:
    - Easy data preparation and splitting
    - Local simulation for testing
    - FLARE job configuration generation
    - Result management and persistence
    
    Example:
        >>> fe = FeatureElection(freedom_degree=0.5, fs_method='lasso')
        >>> client_data = fe.prepare_data_splits(df, 'target', num_clients=4)
        >>> stats = fe.simulate_election(client_data)
        >>> selected_features = fe.selected_feature_names
    """
    
    def __init__(
        self,
        freedom_degree: float = 0.5,
        fs_method: str = "lasso",
        aggregation_mode: str = "weighted",
        auto_tune: bool = False
    ):
        """
        Initialize Feature Election
        
        Args:
            freedom_degree: Controls feature selection strategy (0=intersection, 1=union).
                          If auto_tune=True, this serves as initial value.
            fs_method: Feature selection method. Options:
                      'lasso', 'elastic_net', 'random_forest', 'mutual_info',
                      'chi2', 'f_classif', 'rfe', 'pyimpetus'
            aggregation_mode: How to aggregate client contributions:
                             'weighted' - weight by sample count (recommended)
                             'uniform' - equal weight for all clients
            auto_tune: Whether to automatically optimize freedom_degree
        
        Raises:
            ValueError: If parameters are invalid
        """
        if not 0 <= freedom_degree <= 1:
            raise ValueError("freedom_degree must be between 0 and 1")
        if aggregation_mode not in ['weighted', 'uniform']:
            raise ValueError("aggregation_mode must be 'weighted' or 'uniform'")
            
        self.freedom_degree = freedom_degree
        self.fs_method = fs_method
        self.aggregation_mode = aggregation_mode
        self.auto_tune = auto_tune
        
        # Storage for results
        self.global_mask = None
        self.selected_feature_names = None
        self.election_stats = {}
        
    def create_flare_job(
        self,
        job_name: str = "feature_election",
        output_dir: str = "jobs/feature_election",
        min_clients: int = 2,
        num_rounds: int = 1,
        client_sites: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Generate NVIDIA FLARE job configuration for Feature Election.
        Creates a complete job folder that can be submitted to FLARE.
        
        Args:
            job_name: Name of the FLARE job
            output_dir: Directory to save job configuration
            min_clients: Minimum number of clients required
            num_rounds: Number of election rounds (typically 1)
            client_sites: List of client site names (e.g., ['site-1', 'site-2'])
        
        Returns:
            Dictionary with paths to created configuration files:
            {'job_dir': str, 'server_config': str, 'client_config': str, 'meta': str}
        
        Example:
            >>> fe = FeatureElection(freedom_degree=0.5)
            >>> paths = fe.create_flare_job(
            ...     job_name="my_feature_selection",
            ...     output_dir="./jobs",
            ...     client_sites=['hospital_1', 'hospital_2', 'hospital_3']
            ... )
            >>> # Submit: nvflare job submit -j ./jobs/my_feature_selection
        """
        job_path = Path(output_dir) / job_name
        job_path.mkdir(parents=True, exist_ok=True)
        
        # Create app folders
        (job_path / "app" / "config").mkdir(parents=True, exist_ok=True)
        (job_path / "app" / "custom").mkdir(parents=True, exist_ok=True)
        
        # Server configuration (config_fed_server.json)
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
                    }
                }
            ],
            "components": []
        }
        
        # Client configuration (config_fed_client.json)
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
                            "task_name": "feature_election"
                        }
                    }
                }
            ],
            "task_result_filters": [],
            "task_data_filters": []
        }
        
        # Meta configuration (meta.json)
        if client_sites is None:
            client_sites = [f"site-{i+1}" for i in range(min_clients)]
            
        meta_config = {
            "name": job_name,
            "resource_spec": {
                "site-1": {
                    "num_of_gpus": 0,
                    "mem_per_gpu_in_GiB": 0
                }
            },
            "min_clients": min_clients,
            "mandatory_clients": [],
            "deploy_map": {
                "app": ["@ALL"]
            },
            "task_data_filters": [],
            "task_result_filters": []
        }
        
        # Save configurations
        server_config_path = job_path / "app" / "config" / "config_fed_server.json"
        client_config_path = job_path / "app" / "config" / "config_fed_client.json"
        meta_config_path = job_path / "meta.json"
        
        with open(server_config_path, 'w') as f:
            json.dump(server_config, f, indent=2)
            
        with open(client_config_path, 'w') as f:
            json.dump(client_config, f, indent=2)
        
        with open(meta_config_path, 'w') as f:
            json.dump(meta_config, f, indent=2)
        
        # Create README
        readme_path = job_path / "README.md"
        with open(readme_path, 'w') as f:
            f.write(f"""# {job_name}

Feature Election job for NVIDIA FLARE.

## Configuration

- **Freedom Degree**: {self.freedom_degree}
- **FS Method**: {self.fs_method}
- **Aggregation Mode**: {self.aggregation_mode}
- **Min Clients**: {min_clients}

## Usage

1. Ensure clients have loaded their data using FeatureElectionExecutor.set_data()
2. Submit the job:
   ```bash
   nvflare job submit -j {job_path}
   ```
3. Monitor the job:
   ```bash
   nvflare job list
   ```
4. Retrieve results after completion

## Client Data Setup

On each client, use:

```python
from nvflare.app_opt.feature_election import FeatureElectionExecutor

executor = FeatureElectionExecutor(fs_method='{self.fs_method}')
X_train, y_train = load_your_data()  # Your data loading logic
executor.set_data(X_train, y_train, feature_names=feature_names)
```
""")
        
        logger.info(f"FLARE job configuration created in {job_path}")
        
        return {
            "job_dir": str(job_path),
            "server_config": str(server_config_path),
            "client_config": str(client_config_path),
            "meta": str(meta_config_path),
            "readme": str(readme_path)
        }
    
    def prepare_data_splits(
        self,
        df: pd.DataFrame,
        target_col: str,
        num_clients: int = 3,
        split_strategy: str = "stratified",
        split_ratios: Optional[List[float]] = None,
        random_state: int = 42
    ) -> List[Tuple[pd.DataFrame, pd.Series]]:
        """
        Prepare data splits for federated clients (simulation/testing).
        
        Args:
            df: Input DataFrame with features and target
            target_col: Name of target column
            num_clients: Number of clients to simulate
            split_strategy: Strategy for splitting data:
                          'stratified' - maintain class distribution (recommended)
                          'random' - random split
                          'sequential' - sequential split (ordered data)
                          'dirichlet' - non-IID split using Dirichlet distribution
            split_ratios: Custom split ratios (must sum to 1.0).
                         If None, uses uneven split to simulate realistic scenario
            random_state: Random seed for reproducibility
        
        Returns:
            List of (X, y) tuples for each client
        
        Example:
            >>> client_data = fe.prepare_data_splits(
            ...     df=my_dataframe,
            ...     target_col='diagnosis',
            ...     num_clients=5,
            ...     split_strategy='stratified'
            ... )
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        if split_ratios is None:
            # Default: uneven split to simulate realistic federated scenario
            if num_clients == 2:
                split_ratios = [0.6, 0.4]
            elif num_clients == 3:
                split_ratios = [0.5, 0.3, 0.2]
            elif num_clients == 4:
                split_ratios = [0.4, 0.3, 0.2, 0.1]
            else:
                # Equal splits for other cases
                split_ratios = [1.0 / num_clients] * num_clients
        
        if abs(sum(split_ratios) - 1.0) > 0.001:
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratios)}")
        
        client_data = []
        indices = np.arange(len(df))
        
        if split_strategy == "stratified":
            remaining_X = X
            remaining_y = y
            remaining_indices = indices
            
            for i in range(num_clients - 1):
                size = split_ratios[i] / sum(split_ratios[i:])
                
                client_indices, remaining_indices = train_test_split(
                    remaining_indices,
                    test_size=1-size,
                    stratify=remaining_y,
                    random_state=random_state + i
                )
                
                client_X = X.iloc[client_indices]
                client_y = y.iloc[client_indices]
                client_data.append((client_X, client_y))
                
                remaining_X = X.iloc[remaining_indices]
                remaining_y = y.iloc[remaining_indices]
            
            # Last client gets remaining data
            client_data.append((remaining_X, remaining_y))
            
        elif split_strategy == "random":
            np.random.seed(random_state)
            np.random.shuffle(indices)
            start = 0
            for ratio in split_ratios:
                end = start + int(len(indices) * ratio)
                client_indices = indices[start:end]
                client_X = X.iloc[client_indices]
                client_y = y.iloc[client_indices]
                client_data.append((client_X, client_y))
                start = end
                
        elif split_strategy == "sequential":
            start = 0
            for ratio in split_ratios:
                end = start + int(len(indices) * ratio)
                client_indices = indices[start:end]
                client_X = X.iloc[client_indices]
                client_y = y.iloc[client_indices]
                client_data.append((client_X, client_y))
                start = end
                
        elif split_strategy == "dirichlet":
            # Non-IID split using Dirichlet distribution
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            n_classes = len(le.classes_)
            
            # Generate Dirichlet distribution (alpha=0.5 creates non-IID)
            np.random.seed(random_state)
            label_distribution = np.random.dirichlet([0.5] * num_clients, n_classes)
            
            client_indices = [[] for _ in range(num_clients)]
            for k in range(n_classes):
                idx_k = np.where(y_encoded == k)[0]
                np.random.shuffle(idx_k)
                
                proportions = label_distribution[k]
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                
                client_splits = np.split(idx_k, proportions)
                for i in range(num_clients):
                    if i < len(client_splits):
                        client_indices[i].extend(client_splits[i])
            
            for indices_i in client_indices:
                client_X = X.iloc[indices_i]
                client_y = y.iloc[indices_i]
                client_data.append((client_X, client_y))
        else:
            raise ValueError(f"Unknown split strategy: {split_strategy}")
        
        logger.info(f"Data split into {num_clients} clients using '{split_strategy}' strategy")
        logger.info(f"Sample distribution: {[len(X) for X, _ in client_data]}")
        
        return client_data
    
    def simulate_election(
        self,
        client_data: List[Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]],
        feature_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Simulate Feature Election locally (for testing without FLARE deployment).
        This runs the complete election process in-memory for rapid prototyping.
        
        Args:
            client_data: List of (X, y) tuples for each client
            feature_names: Optional feature names (auto-detected from DataFrame)
            
        Returns:
            Dictionary with election statistics:
            - num_clients: Number of participating clients
            - num_features_original: Original feature count
            - num_features_selected: Selected feature count
            - reduction_ratio: Feature reduction ratio
            - freedom_degree: Used freedom degree (may differ if auto-tuned)
            - client_stats: Per-client statistics
            - intersection_features: Number of features in intersection
            - union_features: Number of features in union
        
        Example:
            >>> stats = fe.simulate_election(client_data)
            >>> print(f"Reduced from {stats['num_features_original']} to "
            ...       f"{stats['num_features_selected']} features")
        """
        # Import here to avoid circular dependency
        from .executor import FeatureElectionExecutor
        from .controller import FeatureElectionController
        
        # Initialize controller
        controller = FeatureElectionController(
            freedom_degree=self.freedom_degree,
            aggregation_mode=self.aggregation_mode,
            min_clients=len(client_data)
        )
        
        # Perform feature selection for each client
        client_selections = {}
        
        for i, (X, y) in enumerate(client_data):
            # Convert to numpy if needed
            if isinstance(X, pd.DataFrame):
                X_np = X.values
                if feature_names is None:
                    feature_names = X.columns.tolist()
            else:
                X_np = X
                
            if isinstance(y, pd.Series):
                y_np = y.values
            else:
                y_np = y
            
            # Create executor for this client
            executor = FeatureElectionExecutor(
                fs_method=self.fs_method,
                eval_metric="f1"
            )
            executor.set_data(X_np, y_np, feature_names=feature_names)
            
            # Perform feature selection
            selected_mask, feature_scores = executor._perform_feature_selection()
            
            # Evaluate
            initial_score = executor._evaluate_model(X_np, y_np, X_np, y_np)
            X_selected = X_np[:, selected_mask]
            fs_score = executor._evaluate_model(X_selected, y_np, X_selected, y_np)
            
            client_selections[f"client_{i}"] = {
                "selected_features": selected_mask,
                "feature_scores": feature_scores,
                "num_samples": len(X_np),
                "initial_score": initial_score,
                "fs_score": fs_score
            }
            
            logger.info(f"Client {i}: {np.sum(selected_mask)}/{len(selected_mask)} features, "
                       f"score: {initial_score:.3f} -> {fs_score:.3f}")
        
        # Auto-tune freedom degree if requested
        if self.auto_tune:
            best_fd, best_score = self._auto_tune_freedom_degree(client_selections)
            self.freedom_degree = best_fd
            controller.freedom_degree = best_fd
            logger.info(f"Auto-tuned freedom_degree: {best_fd:.2f} (score: {best_score:.3f})")
        
        # Aggregate selections
        self.global_mask = controller._aggregate_selections(client_selections)
        
        # Calculate intersection and union for stats
        masks = np.array([sel["selected_features"] for sel in client_selections.values()])
        intersection_mask = np.all(masks, axis=0)
        union_mask = np.any(masks, axis=0)
        
        # Store results
        self.election_stats = {
            "num_clients": len(client_data),
            "num_features_original": len(self.global_mask),
            "num_features_selected": int(np.sum(self.global_mask)),
            "reduction_ratio": 1 - (np.sum(self.global_mask) / len(self.global_mask)),
            "freedom_degree": self.freedom_degree,
            "aggregation_mode": self.aggregation_mode,
            "fs_method": self.fs_method,
            "intersection_features": int(np.sum(intersection_mask)),
            "union_features": int(np.sum(union_mask)),
            "client_stats": {
                name: {
                    "num_selected": int(np.sum(sel["selected_features"])),
                    "initial_score": float(sel["initial_score"]),
                    "fs_score": float(sel["fs_score"]),
                    "improvement": float(sel["fs_score"] - sel["initial_score"]),
                    "num_samples": sel["num_samples"]
                }
                for name, sel in client_selections.items()
            }
        }
        
        if feature_names is not None:
            self.selected_feature_names = [
                name for i, name in enumerate(feature_names)
                if self.global_mask[i]
            ]
        
        logger.info(f"Election completed: {self.election_stats['num_features_selected']}/"
                   f"{self.election_stats['num_features_original']} features selected")
        
        return self.election_stats
    
    def _auto_tune_freedom_degree(
        self,
        client_selections: Dict,
        candidate_freedoms: Optional[List[float]] = None
    ) -> Tuple[float, float]:
        """
        Auto-tune freedom degree using performance-based optimization.
        
        Args:
            client_selections: Dictionary of client selection data
            candidate_freedoms: List of freedom degrees to try
        
        Returns:
            Tuple of (best_freedom_degree, best_score)
        """
        from .controller import FeatureElectionController

        
        if candidate_freedoms is None:
            candidate_freedoms = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        best_fd = 0.5
        best_score = -float('inf')
        
        for fd in candidate_freedoms:
            controller = FeatureElectionController(
                freedom_degree=fd,
                aggregation_mode=self.aggregation_mode
            )
            
            # Get global mask for this fd
            global_mask = controller._aggregate_selections(client_selections)
            
            # Evaluate: balance between selection ratio and average score improvement
            num_selected = np.sum(global_mask)
            num_total = len(global_mask)
            
            if num_selected == 0:
                # Skip if no features selected
                continue
                
            selection_ratio = num_selected / num_total
            
            # Average score improvement across clients
            improvements = [
                sel["fs_score"] - sel["initial_score"]
                for sel in client_selections.values()
            ]
            avg_improvement = np.mean(improvements)
            
            # Combined score: balance performance improvement and dimensionality reduction
            # Prefer moderate reduction (30-70% of features kept)
            if 0.3 <= selection_ratio <= 0.7:
                reduction_bonus = 1.0
            else:
                reduction_bonus = 0.5
            
            combined_score = avg_improvement * reduction_bonus
            
            logger.debug(f"fd={fd:.2f}: selected={num_selected}/{num_total}, "
                        f"improvement={avg_improvement:.4f}, score={combined_score:.4f}")
            
            if combined_score > best_score:
                best_score = combined_score
                best_fd = fd
        
        return best_fd, best_score
    
    def apply_mask(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Apply the global feature mask to new data.
        
        Args:
            X: Input features (DataFrame or numpy array)
            feature_names: Feature names (for validation)
            
        Returns:
            Filtered features with only selected features
        
        Raises:
            ValueError: If no global mask is available
        
        Example:
            >>> X_selected = fe.apply_mask(X_test)
        """
        if self.global_mask is None:
            raise ValueError("No global mask available. Run simulate_election() first.")
        
        if isinstance(X, pd.DataFrame):
            if self.selected_feature_names is not None:
                return X[self.selected_feature_names]
            else:
                # Use boolean indexing
                return X.iloc[:, self.global_mask]
        else:
            return X[:, self.global_mask]
    
    def save_results(self, filepath: str):
        """
        Save election results to JSON file.
        
        Args:
            filepath: Path to save results
        
        Example:
            >>> fe.save_results("feature_election_results.json")
        """
        results = {
            "freedom_degree": self.freedom_degree,
            "fs_method": self.fs_method,
            "aggregation_mode": self.aggregation_mode,
            "auto_tune": self.auto_tune,
            "global_mask": self.global_mask.tolist() if self.global_mask is not None else None,
            "selected_feature_names": self.selected_feature_names,
            "election_stats": self.election_stats
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str):
        """
        Load election results from JSON file.
        
        Args:
            filepath: Path to load results from
        
        Example:
            >>> fe.load_results("feature_election_results.json")
        """
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        self.freedom_degree = results["freedom_degree"]
        self.fs_method = results["fs_method"]
        self.aggregation_mode = results["aggregation_mode"]
        self.auto_tune = results.get("auto_tune", False)
        self.global_mask = np.array(results["global_mask"]) if results["global_mask"] else None
        self.selected_feature_names = results["selected_feature_names"]
        self.election_stats = results["election_stats"]
        
        logger.info(f"Results loaded from {filepath}")


def quick_election(
    df: pd.DataFrame,
    target_col: str,
    num_clients: int = 3,
    freedom_degree: float = 0.5,
    fs_method: str = "lasso",
    auto_tune: bool = False,
    split_strategy: str = "stratified",
    **kwargs
) -> Tuple[np.ndarray, Dict]:
    """
    Quick Feature Election for tabular data (one-line solution).
    
    This is a convenience function that handles data splitting, election simulation,
    and returns results in a single call. Perfect for rapid prototyping and testing.
    
    Args:
        df: Input DataFrame with features and target
        target_col: Name of target column
        num_clients: Number of federated clients to simulate
        freedom_degree: Feature election parameter (0=intersection, 1=union)
        fs_method: Feature selection method ('lasso', 'elastic_net', 'random_forest', etc.)
        auto_tune: Whether to auto-tune freedom degree (recommended)
        split_strategy: Data splitting strategy ('stratified', 'random', 'dirichlet')
        **kwargs: Additional arguments passed to FeatureElection
        
    Returns:
        Tuple of (selected_feature_mask, election_stats)
        - selected_feature_mask: Boolean numpy array indicating selected features
        - election_stats: Dictionary with detailed election statistics
    
    Example:
        >>> import pandas as pd
        >>> from nvflare.app_opt.feature_election import quick_election
        >>> 
        >>> df = pd.read_csv("my_data.csv")
        >>> mask, stats = quick_election(
        ...     df=df,
        ...     target_col='target',
        ...     num_clients=4,
        ...     fs_method='lasso',
        ...     auto_tune=True
        ... )
        >>> print(f"Selected {stats['num_features_selected']} features")
        >>> selected_features = df.columns[:-1][mask]
    """
    # Initialize Feature Election
    fe = FeatureElection(
        freedom_degree=freedom_degree,
        fs_method=fs_method,
        auto_tune=auto_tune,
        **kwargs
    )
    
    # Prepare client data
    client_data = fe.prepare_data_splits(
        df, target_col, num_clients, split_strategy=split_strategy
    )
    
    # Run election
    stats = fe.simulate_election(client_data)
    
    return fe.global_mask, stats


def load_election_results(filepath: str) -> Dict:
    """
    Load election results from a JSON file.
    
    Args:
        filepath: Path to the results file
        
    Returns:
        Dictionary with election results
    
    Example:
        >>> results = load_election_results("feature_election_results.json")
        >>> selected_features = results['selected_feature_names']
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    return results
