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
import logging

from nvflare.fox.api.app import ServerApp
from nvflare.fox.api.utils import simple_logging
from nvflare.fox.examples.pt.pt_avg_stream import PTFedAvgStream, PTTrainer
from nvflare.fox.sys.recipe import FoxRecipe

JOB_ROOT_DIR = "/Users/yanc/NVFlare/sandbox/fox/prod_00/admin@nvidia.com/transfer"


def main():
    simple_logging(logging.DEBUG)

    server_app = ServerApp(
        strategy_name="fedavg_stream",
        strategy=PTFedAvgStream(
            initial_model={
                "x": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                "y": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                "z": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            },
            num_rounds=2,
        ),
    )

    client_app = PTTrainer(delta=1.0)

    recipe = FoxRecipe(
        job_name="pt_fedavg_stream",
        server_app=server_app,
        client_app=client_app,
    )
    recipe.export(JOB_ROOT_DIR)


if __name__ == "__main__":
    main()
