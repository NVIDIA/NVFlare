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
    def __init__(
        self,
        name: str,
        stats_output_path: str,
        sites: List[str],
        statistic_configs: Dict[str, Any],
        stats_generator: Statistics,
    ):

        output_path = stats_output_path

        job = StatsJob(
            job_name=name,
            statistic_configs=statistic_configs,
            stats_generator=stats_generator,
            output_path=output_path,
        )

        job.setup_clients(sites)

        Recipe.__init__(self, job)
