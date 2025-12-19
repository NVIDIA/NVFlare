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

"""Recipe helpers for DH-based Private Set Intersection (PSI)."""

from nvflare.app_common.psi.dh_psi.dh_psi_controller import DhPSIController
from nvflare.app_common.psi.file_psi_writer import FilePSIWriter
from nvflare.app_common.psi.psi_executor import PSIExecutor
from nvflare.app_common.psi.psi_spec import PSI
from nvflare.app_opt.psi.dh_psi.dh_psi_task_handler import DhPSITaskHandler
from nvflare.job_config.api import FedJob
from nvflare.recipe.spec import Recipe


class DhPSIRecipe(Recipe):
    """Job recipe for running DH-PSI.

    This is a job composition helper (server workflow + client executors/components).
    """

    def __init__(
        self,
        *,
        name: str = "dh_psi",
        min_clients: int,
        local_psi: PSI,
        output_path: str = "psi/intersection.txt",
    ):
        """Constructor of DhPSIRecipe.

        Args:
            name: Name of the federated job.
            min_clients: Minimum number of clients required to start the PSI workflow.
            local_psi: PSI component implementation used on each client to compute local PSI artifacts.
            output_path: Local file path on each client where the PSI intersection result will be written.
        """
        # These IDs and task name must be consistent across PSI components.
        psi_task_name = "PSI"
        local_psi_id = "local_psi"
        psi_algo_id = "dh_psi"

        job = FedJob(name=name, min_clients=min_clients)

        # Server: PSI workflow controller.
        job.to_server(DhPSIController(), id="DhPSIController")

        # Client: executor + task handler + user PSI + writer.
        job.to_clients(
            PSIExecutor(psi_algo_id=psi_algo_id),
            id="Executor",
            tasks=[psi_task_name],
        )
        job.to_clients(
            DhPSITaskHandler(local_psi_id=local_psi_id),
            id=psi_algo_id,
        )
        job.to_clients(local_psi, id=local_psi_id)
        job.to_clients(FilePSIWriter(output_path=output_path), id=local_psi.psi_writer_id)

        super().__init__(job)
