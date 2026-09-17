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

from typing import Optional

from server import KM
from server_he import KM_HE

from nvflare import FedJob
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.recipe.spec import Recipe


class KaplanMeierRecipe(Recipe):
    """Configure a federated Kaplan-Meier job with optional homomorphic encryption."""

    def __init__(
        self,
        *,
        num_clients: int,
        encryption: bool = False,
        data_root: str = "/tmp/nvflare/dataset/km_data",
        he_context_path_client: Optional[str] = "/tmp/nvflare/he_context/he_context_client.txt",
        he_context_path_server: Optional[str] = "/tmp/nvflare/he_context/he_context_server.txt",
    ):
        if encryption:
            if not he_context_path_client:
                raise ValueError("he_context_path_client must be provided when encryption=True")
            if not he_context_path_server:
                raise ValueError("he_context_path_server must be provided when encryption=True")
            job_name = "KM_HE"
            train_script = "client_he.py"
            script_args = f"--data_root {data_root} --he_context_path {he_context_path_client}"
            controller = KM_HE(min_clients=num_clients, he_context_path=he_context_path_server)
        else:
            job_name = "KM"
            train_script = "client.py"
            script_args = f"--data_root {data_root}"
            controller = KM(min_clients=num_clients)

        job = FedJob(name=job_name, min_clients=num_clients)
        job.to_server(controller)
        job.to_clients(
            ScriptRunner(
                script=train_script,
                script_args=script_args,
                framework="raw",
                launch_external_process=False,
            ),
            tasks=["train"],
        )
        super().__init__(job)
