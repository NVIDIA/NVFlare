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

import argparse

from stats import DFStatistics

from nvflare.recipe import SimEnv
from nvflare.recipe.fedstats import FedStatsRecipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_dir", default="/tmp/nvflare/df_stats/data")
    parser.add_argument("--workspace_dir", default="/tmp/nvflare/chapter2/stats_df")
    args = parser.parse_args()
    recipe = FedStatsRecipe(
        name="stats_df",
        sites=[f"site-{i}" for i in range(1, 3)],
        stats_output_path="statistics/adults_stats.json",
        statistic_configs={
            "count": {},
            "mean": {},
            "sum": {},
            "stddev": {},
            "histogram": {"*": {"bins": 20}, "Age": {"bins": 20, "range": [0, 100]}},
            "quantile": {"*": [0.1, 0.5, 0.9], "Age": [0.5, 0.9]},
        },
        stats_generator=DFStatistics(filename="data.csv", data_root_dir=args.data_root_dir),
    )
    run = recipe.execute(
        SimEnv(clients=[f"site-{i}" for i in range(1, 3)], workspace_root=args.workspace_dir, log_config="progress")
    )
    print("Results at:", run.get_result())


if __name__ == "__main__":
    main()
