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

"""Run the real Flower smoke job in an independently owned interpreter."""

import sys

from nvflare.app_opt.flower.recipe import FlowerRecipe
from nvflare.recipe import SimEnv


def run(workspace, app):
    recipe = FlowerRecipe(
        name="flower-http-smoke",
        flower_content=app,
        min_clients=2,
        superlink_ready_timeout=30,
        configure_task_timeout=60,
        start_task_timeout=60,
    )
    env = SimEnv(num_clients=2, workspace_root=workspace)
    execution = None
    try:
        execution = recipe.execute(env)
        execution.get_result(timeout=180, clean_up=False)
    finally:
        if execution is None or not execution._stopped:
            if execution:
                execution.abort()
            env.stop(clean_up=False)


if __name__ == "__main__":
    run(*sys.argv[1:])
