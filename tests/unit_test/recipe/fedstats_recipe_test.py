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

from unittest.mock import Mock, patch

import pytest

from nvflare.fuel.utils.secret_utils import PotentialSecretWarning, UnsupportedSecretRefWarning
from nvflare.recipe import secret_ref
from nvflare.recipe.fedstats import FedStatsRecipe


def test_fedstats_warns_for_secret_like_statistic_config():
    with patch("nvflare.recipe.fedstats.StatsJob") as stats_job_cls:
        with pytest.warns(PotentialSecretWarning, match="statistic_configs"):
            recipe = FedStatsRecipe(
                name="stats",
                stats_output_path="stats.json",
                sites=["site-1"],
                statistic_configs={"password": "hunter22x"},
                stats_generator=Mock(),
            )

    stats_job_cls.return_value.setup_clients.assert_called_once_with(["site-1"])
    assert recipe.job is stats_job_cls.return_value


def test_fedstats_warns_for_unsupported_secret_ref():
    with patch("nvflare.recipe.fedstats.StatsJob"):
        with pytest.warns(UnsupportedSecretRefWarning, match="statistic_configs"):
            FedStatsRecipe(
                name="stats",
                stats_output_path="stats.json",
                sites=["site-1"],
                statistic_configs={"api_token": secret_ref("API_TOKEN")},
                stats_generator=Mock(),
            )
