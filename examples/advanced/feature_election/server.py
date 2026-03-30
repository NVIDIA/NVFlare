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
Server-side configuration for Feature Election example.

This script provides factory functions for creating the
FeatureElectionController from nvflare.app_opt.feature_election.
"""

import logging

from nvflare.app_opt.feature_election.controller import FeatureElectionController

logger = logging.getLogger(__name__)


def get_controller(
    freedom_degree: float = 0.5,
    aggregation_mode: str = "weighted",
    min_clients: int = 2,
    num_rounds: int = 5,
    auto_tune: bool = False,
    tuning_rounds: int = 4,
) -> FeatureElectionController:
    """
    Create and configure a FeatureElectionController.

    Args:
        freedom_degree: Controls feature inclusion (0=intersection, 1=union)
        aggregation_mode: How to weight client votes ('weighted' or 'uniform')
        min_clients: Minimum clients required to proceed
        num_rounds: Number of FL training rounds after feature selection
        auto_tune: Whether to automatically tune freedom_degree
        tuning_rounds: Number of rounds for auto-tuning

    Returns:
        Configured FeatureElectionController
    """
    controller = FeatureElectionController(
        freedom_degree=freedom_degree,
        aggregation_mode=aggregation_mode,
        min_clients=min_clients,
        num_rounds=num_rounds,
        task_name="feature_election",
        auto_tune=auto_tune,
        tuning_rounds=tuning_rounds,
    )

    logger.info(
        f"Controller configured: FD={freedom_degree}, "
        f"mode={aggregation_mode}, rounds={num_rounds}, "
        f"auto_tune={auto_tune}"
    )

    return controller


# Default configurations for common scenarios
CONFIGS = {
    "basic": {
        "freedom_degree": 0.5,
        "aggregation_mode": "weighted",
        "min_clients": 2,
        "num_rounds": 5,
        "auto_tune": False,
        "tuning_rounds": 0,
    },
    "auto_tune": {
        "freedom_degree": 0.6,
        "aggregation_mode": "weighted",
        "min_clients": 2,
        "num_rounds": 10,
        "auto_tune": True,
        "tuning_rounds": 4,
    },
    "conservative": {
        "freedom_degree": 0.2,
        "aggregation_mode": "weighted",
        "min_clients": 2,
        "num_rounds": 5,
        "auto_tune": False,
        "tuning_rounds": 0,
    },
    "aggressive": {
        "freedom_degree": 0.8,
        "aggregation_mode": "weighted",
        "min_clients": 2,
        "num_rounds": 5,
        "auto_tune": False,
        "tuning_rounds": 0,
    },
}


def get_controller_by_name(config_name: str = "basic") -> FeatureElectionController:
    """
    Get a controller with a predefined configuration.

    Args:
        config_name: One of 'basic', 'auto_tune', 'conservative', 'aggressive'

    Returns:
        Configured FeatureElectionController
    """
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(CONFIGS.keys())}")

    return get_controller(**CONFIGS[config_name])
