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

"""Collab Simulation Environment using nvflare.collab.sim.Simulator."""

import os
from typing import Dict, Optional, Tuple, Union

from nvflare.collab.sim.publish_simulator import CollabSimulator
from nvflare.job_config.api import FedJob
from nvflare.recipe.spec import ExecEnv

WORKSPACE_ROOT = "/tmp/nvflare/collab_simulation"


class SimEnv(ExecEnv):
    """Simulation execution environment for Collab using the Collab Simulator.

    This environment runs federated learning jobs using the Collab simulation
    backend. Supports both in-process simulation and subprocess execution
    (e.g., for torchrun multi-GPU training).
    """

    def __init__(
        self,
        *,
        num_clients: Union[int, Tuple[int, int]] = 2,
        server: object = None,
        client: object = None,
        server_objects: Dict[str, object] = None,
        client_objects: Dict[str, object] = None,
        max_workers: int = 100,
        workspace_root: str = WORKSPACE_ROOT,
        # Subprocess execution options
        inprocess: bool = True,
        run_cmd: Optional[str] = None,
        subprocess_timeout: float = 300.0,
        extra: dict = None,
    ):
        """Initialize Collab simulation execution environment.

        Args:
            num_clients: Number of simulated clients, or tuple (min, max) for range.
            server: Server-side collab object.
            client: Client-side collab object.
            server_objects: Additional server-side collab objects.
            client_objects: Additional client-side collab objects.
            max_workers: Maximum number of worker threads.
            workspace_root: Root directory for simulation workspace.
            inprocess: If True, execute in-process. If False, use subprocess.
            run_cmd: Command prefix for subprocess (e.g., "torchrun --nproc_per_node=4").
            subprocess_timeout: Timeout for subprocess operations.
            extra: Extra env config info.
        """
        super().__init__(extra)

        self.num_clients = num_clients
        self.server = server
        self.client = client
        self.server_objects = server_objects
        self.client_objects = client_objects
        self.max_workers = max_workers
        self.workspace_root = workspace_root

        # Subprocess options
        self.inprocess = inprocess
        self.run_cmd = run_cmd
        self.subprocess_timeout = subprocess_timeout
        self.training_module = None  # Auto-detected from recipe, not user-facing

        self._simulator: Optional[CollabSimulator] = None

    def deploy(self, job: FedJob) -> str:
        """Deploy a FedJob using the Collab Simulator.

        Args:
            job: The FedJob to deploy.

        Returns:
            str: The job ID (job name).
        """
        experiment_name = job.name
        root_dir = os.path.join(self.workspace_root, experiment_name)
        os.makedirs(root_dir, exist_ok=True)

        print(f"\n{'='*60}")
        if self.inprocess:
            print("Collab SimEnv: Starting in-process simulation")
        else:
            print("Collab SimEnv: Starting subprocess simulation")
            if self.run_cmd:
                print(f"  Run command: {self.run_cmd}")
        print(f"{'='*60}")
        print(f"  → Creating simulator with {self.num_clients} clients...")

        self._simulator = CollabSimulator(
            root_dir=root_dir,
            experiment_name=experiment_name,
            server=self.server,
            client=self.client,
            server_objects=self.server_objects,
            client_objects=self.client_objects,
            max_workers=self.max_workers,
            num_clients=self.num_clients,
            # Subprocess options
            inprocess=self.inprocess,
            run_cmd=self.run_cmd,
            training_module=self.training_module,
            subprocess_timeout=self.subprocess_timeout,
        )

        print("  → Running simulation...")
        print(f"{'='*60}\n")

        # Run the simulation
        self._simulator.run()

        print(f"\n{'='*60}")
        print("  → Simulation completed")
        print(f"{'='*60}\n")

        return experiment_name

    def get_job_status(self, job_id: str) -> Optional[str]:
        """Get job status - not fully supported in Collab simulation environment."""
        print(
            f"Note: get_status returns None in Collab SimEnv. "
            f"Logs can be found at {os.path.join(self.workspace_root, job_id)}"
        )
        return None

    def abort_job(self, job_id: str) -> None:
        """Abort job - not supported in Collab simulation environment."""
        print("Abort is not supported in Collab simulation environment, it will always run to completion.")

    def get_job_result(self, job_id: str, timeout: float = 0.0) -> Optional[str]:
        """Get job result workspace path.

        Args:
            job_id: The job ID to get results for.
            timeout: The timeout for the job to complete (not used in simulation).

        Returns:
            str: The result workspace path.
        """
        if self.workspace_root is None:
            raise RuntimeError("Simulation workspace_root is None - SimEnv may not be properly initialized")
        return os.path.join(self.workspace_root, job_id)
