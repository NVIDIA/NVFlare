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

from server import KM

from nvflare import FedJob
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.recipe.spec import Recipe


class KaplanMeierRecipe(Recipe):
    """Package the custom histogram protocol for simulation or export."""

    def __init__(self, num_clients=5, encryption=False):
        name = "km_he" if encryption else "km"
        data_root = "/tmp/nvflare/dataset/km_data"
        if encryption:
            from server_he import KM_HE

            context = "/tmp/nvflare/he_context/he_context_server.txt"
            controller = KM_HE(min_clients=num_clients, he_context_path=context)
            script = "client_he.py"
            train_args = f"--data_root {data_root} --he_context_path /tmp/nvflare/he_context/he_context_client.txt"
        else:
            controller = KM(min_clients=num_clients)
            script = "client.py"
            train_args = f"--data_root {data_root}"
        job = FedJob(name=name, min_clients=num_clients)
        job.to_server(controller)
        job.to_clients(ScriptRunner(script=script, script_args=train_args, framework="raw"), tasks=["train"])
        super().__init__(job)
