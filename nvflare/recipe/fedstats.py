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

from typing import Any, Dict, List

from nvflare.app_common.abstract.statistics_spec import Statistics
from nvflare.job_config.stats_job import StatsJob
from nvflare.recipe.spec import Recipe


class FedStatsRecipe(Recipe):
    """A recipe for federated statistics computation.

    FedStatsRecipe is a specialized recipe that facilitates the computation of
    statistics across multiple federated sites. It creates and configures a
    StatsJob with the specified parameters and sets up the necessary client
    connections for distributed statistics computation.

    This recipe computes various statistical measures (such as mean, variance,
    histograms, & quantiles) across data distributed across multiple sites while
    maintaining data privacy.

    Args:
        name (str): The name of the federated statistics job.
        stats_output_path (str): The file path where the computed statistics
            results will be saved.
        sites (List[str]): A list of site names/identifiers that will
            participate in the federated statistics computation.
        statistic_configs (Dict[str, Any]): Configuration dictionary specifying
            which statistics to compute and their parameters. The structure
            depends on the specific statistics generator being used.
        stats_generator (Statistics): An instance of a Statistics class that
            implements the actual statistics computation logic. This object
            must implement the Statistics interface.

    Example:
        >>> from nvflare.recipe.fedstats import FedStatsRecipe
        >>> from my_stats_generator import MyStatsGenerator
        >>>
        >>> config = {
        ...     "count": {},
        ...     "sum": {},
        ...     "mean": {},
        ...     "std": {}
        ... }
        >>>
        >>> recipe = FedStatsRecipe(
        ...     name="my_stats_job",
        ...     stats_output_path="path/to/output",
        ...     sites=["site1", "site2", "site3"],
        ...     statistic_configs=config,
        ...     stats_generator=MyStatsGenerator()
        ... )
    """

    def __init__(
        self,
        name: str,
        stats_output_path: str,
        sites: List[str],
        statistic_configs: Dict[str, Any],
        stats_generator: Statistics,
    ):
        job = StatsJob(
            job_name=name,
            statistic_configs=statistic_configs,
            stats_generator=stats_generator,
            output_path=stats_output_path,
        )

        job.setup_clients(sites)

        Recipe.__init__(self, job)
