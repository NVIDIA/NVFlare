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
Client-side script for Feature Election example.

This script demonstrates how to set up client data for the
FeatureElectionExecutor from nvflare.app_opt.feature_election.
"""

import logging
import re
from typing import Optional

from prepare_data import load_client_data

from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.feature_election.executor import FeatureElectionExecutor

logger = logging.getLogger(__name__)


def get_executor(
    client_id: int,
    num_clients: int,
    fs_method: str = "lasso",
    eval_metric: str = "f1",
    data_root: Optional[str] = None,
    split_strategy: str = "stratified",
    n_samples: int = 1000,
    n_features: int = 100,
    n_informative: int = 20,
    n_redundant: int = 30,
) -> FeatureElectionExecutor:
    """
    Create and configure a FeatureElectionExecutor with data.

    Args:
        client_id: Client identifier (0 to num_clients-1)
        num_clients: Total number of clients
        fs_method: Feature selection method
        eval_metric: Evaluation metric ('f1' or 'accuracy')
        data_root: Optional path to pre-generated data
        split_strategy: Data splitting strategy
        n_samples: Samples per client for synthetic data
        n_features: Number of features
        n_informative: Number of informative features
        n_redundant: Number of redundant features

    Returns:
        Configured FeatureElectionExecutor
    """
    # Create executor
    executor = FeatureElectionExecutor(
        fs_method=fs_method,
        eval_metric=eval_metric,
        task_name="feature_election",
    )

    # Load data for this client
    X_train, y_train, X_val, y_val, feature_names = load_client_data(
        client_id=client_id,
        num_clients=num_clients,
        data_root=data_root,
        split_strategy=split_strategy,
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
    )

    # Set data on executor
    executor.set_data(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names,
    )

    logger.info(
        f"Client {client_id} executor configured: "
        f"{X_train.shape[0]} train, {X_val.shape[0]} val, "
        f"{X_train.shape[1]} features, method={fs_method}"
    )

    return executor


class SyntheticDataExecutor(FeatureElectionExecutor):
    """
    FeatureElectionExecutor with built-in synthetic data loading.

    This executor automatically loads synthetic data based on
    client_id extracted from the FL context.

    Args:
        fs_method: Feature selection method
        eval_metric: Evaluation metric
        num_clients: Total number of clients in federation
        split_strategy: Data splitting strategy
        n_samples: Samples per client
        n_features: Number of features
        n_informative: Number of informative features
        n_redundant: Number of redundant features
        n_repeated: Number of repeated features
    """

    def __init__(
        self,
        fs_method: str = "lasso",
        eval_metric: str = "f1",
        num_clients: int = 3,
        split_strategy: str = "stratified",
        n_samples: int = 1000,
        n_features: int = 100,
        n_informative: int = 20,
        n_redundant: int = 30,
        n_repeated: int = 10,
        task_name: str = "feature_election",
    ):
        super().__init__(
            fs_method=fs_method,
            eval_metric=eval_metric,
            task_name=task_name,
        )
        self.num_clients = num_clients
        self.split_strategy = split_strategy
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_informative = n_informative
        self.n_redundant = n_redundant
        self.n_repeated = n_repeated
        self._data_loaded = False

    def _load_data_if_needed(self, fl_ctx: FLContext) -> None:
        """Load data based on client identity from FL context."""
        if self._data_loaded:
            return

        # Extract client ID from site name
        site_name = fl_ctx.get_identity_name()

        # Parse client_id from site name (e.g., "site-1" -> 0)
        try:
            if site_name.startswith("site-"):
                client_id = int(site_name.split("-")[1]) - 1
            else:
                # Try to extract any number from site name
                match = re.search(r"\d+", site_name)
                if match:
                    client_id = int(match.group()) - 1
                else:
                    client_id = 0
        except (ValueError, IndexError):
            client_id = 0
        X_train, y_train, X_val, y_val, feature_names = load_client_data(
            client_id=client_id,
            num_clients=self.num_clients,
            split_strategy=self.split_strategy,
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=self.n_informative,
            n_redundant=self.n_redundant,
            n_repeated=self.n_repeated,
        )

        self.set_data(X_train, y_train, X_val, y_val, feature_names)
        self._data_loaded = True

        logger.info(f"Loaded synthetic data for {site_name} (client_id={client_id})")

    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        """Override execute to ensure data is loaded before processing."""
        self._load_data_if_needed(fl_ctx)
        return super().execute(task_name, shareable, fl_ctx, abort_signal)
