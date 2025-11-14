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
Feature Election for NVIDIA FLARE

A plug-and-play horizontal federated feature selection framework for tabular datasets.

This module provides:
- FeatureElection: High-level API for feature election
- FeatureElectionController: Server-side FLARE controller
- FeatureElectionExecutor: Client-side FLARE executor
- Helper functions for quick deployment

Example:
    Basic usage::

        from nvflare.app_opt.feature_election import quick_election
        import pandas as pd

        df = pd.read_csv("data.csv")
        selected_mask, stats = quick_election(
            df=df,
            target_col='target',
            num_clients=4,
            fs_method='lasso',
            auto_tune=True
        )

    FLARE deployment::

        from nvflare.app_opt.feature_election import FeatureElection

        fe = FeatureElection(freedom_degree=0.5, fs_method='lasso')
        config_paths = fe.create_flare_job(
            job_name="feature_selection",
            output_dir="./jobs"
        )
"""

from .feature_election import FeatureElection, quick_election, load_election_results
from .controller import FeatureElectionController
from .executor import FeatureElectionExecutor

__version__ = "0.0.9"
__author__ = "Ioannis Christofilogiannis"
__all__ = [
    "FeatureElection",
    "FeatureElectionController", 
    "FeatureElectionExecutor",
    "quick_election",
    "load_election_results"
]
