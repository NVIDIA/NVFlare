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
import argparse

from client import ImageStatistics
from nvflare.recipe import SimEnv
from nvflare.recipe.fedstats import FedStatsRecipe


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_clients", type=int, default=3)
    parser.add_argument("-d", "--data_root_dir", type=str, nargs="?", default="/tmp/nvflare/image_stats/data")
    parser.add_argument("-o", "--stats_output_path", type=str, nargs="?", default="statistics/image_statistics.json")

    return parser.parse_args()


def main():
    args = define_parser()

    n_clients = args.n_clients
    data_root_dir = args.data_root_dir
    output_path = args.stats_output_path

    statistic_configs = {"count": {}, "histogram": {"*": {"bins": 20, "range": [0, 256]}}}
    # define local stats generator
    stats_generator = ImageStatistics(data_root_dir)

    sites = [f"site-{i + 1}" for i in range(n_clients)]
    recipe = FedStatsRecipe(
        name="stats_image",
        stats_output_path=output_path,
        sites=sites,
        statistic_configs=statistic_configs,
        stats_generator=stats_generator,
        min_count = 10
    )

    env = SimEnv(clients=sites, num_threads=n_clients)
    recipe.execute(env=env)


if __name__ == "__main__":
    main()
