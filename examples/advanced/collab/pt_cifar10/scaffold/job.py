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

"""Run the synchronous CIFAR-10 SCAFFOLD Collab example."""

from nvflare.collab import CollabRecipe, simple_logging
from nvflare.recipe import SimEnv

from .client import ScaffoldClient
from .server import ScaffoldServer

NUM_CLIENTS = 2
NUM_ROUNDS = 3


def make_recipe() -> CollabRecipe:
    return CollabRecipe(
        job_name="collab_cifar10_scaffold",
        server=ScaffoldServer(num_rounds=NUM_ROUNDS),
        client=ScaffoldClient(),
        min_clients=NUM_CLIENTS,
    )


def main():
    simple_logging()
    run = make_recipe().execute(SimEnv(num_clients=NUM_CLIENTS))
    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())


if __name__ == "__main__":
    main()
