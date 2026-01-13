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
import os

from server import KM
from server_he import KM_HE

from nvflare import FedJob
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.recipe import ProdEnv, SimEnv
from nvflare.recipe.spec import Recipe


class KMRecipe(Recipe):
    """Recipe wrapper around the Kaplan-Meier job configuration.

    This provides a recipe-style API for easy job configuration and execution
    in both simulation and production environments.
    """

    def __init__(
        self,
        *,
        num_clients: int,
        encryption: bool = False,
        data_root: str = "/tmp/nvflare/dataset/km_data",
        he_context_path_client: str = "/tmp/nvflare/he_context/he_context_client.txt",
        he_context_path_server: str = "/tmp/nvflare/he_context/he_context_server.txt",
    ):
        self.num_clients = num_clients
        self.encryption = encryption
        self.data_root = data_root
        self.he_context_path_client = he_context_path_client
        self.he_context_path_server = he_context_path_server

        # Set job name and script based on encryption mode
        if self.encryption:
            job_name = "KM_HE"
            train_script = "client_he.py"
            script_args = f"--data_root {data_root} --he_context_path {he_context_path_client}"
            controller = KM_HE(min_clients=num_clients, he_context_path=he_context_path_server)
        else:
            job_name = "KM"
            train_script = "client.py"
            script_args = f"--data_root {data_root}"
            controller = KM(min_clients=num_clients)

        # Create the FedJob
        job = FedJob(name=job_name, min_clients=num_clients)

        # Add controller workflow to server
        job.to_server(controller)

        # Add ScriptRunner to all clients
        runner = ScriptRunner(
            script=train_script,
            script_args=script_args,
            framework="raw",
            launch_external_process=False,
        )
        job.to_clients(runner, tasks=["train"])

        super().__init__(job)


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace_dir",
        type=str,
        default="/tmp/nvflare/workspaces",
        help="Work directory for simulator runs, default to '/tmp/nvflare/workspaces'",
    )
    parser.add_argument(
        "--job_dir",
        type=str,
        default="/tmp/nvflare/jobs",
        help="Directory for job export, default to '/tmp/nvflare/jobs'",
    )
    parser.add_argument(
        "--encryption",
        action=argparse.BooleanOptionalAction,
        help="Whether to enable encryption, default to False",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=5,
        help="Number of clients to simulate, default to 5",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        help="Number of threads to use for FL simulation, default to the number of clients if not specified",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/tmp/nvflare/dataset/km_data",
        help="Root directory for KM data, default to '/tmp/nvflare/dataset/km_data'",
    )
    parser.add_argument(
        "--he_context_path",
        type=str,
        default="/tmp/nvflare/he_context/he_context_client.txt",
        help="Path to HE context file for simulation mode (client context), default to '/tmp/nvflare/he_context/he_context_client.txt'. In production mode, context files are auto-provisioned.",
    )
    parser.add_argument(
        "--startup_kit_location",
        type=str,
        default=None,
        help="Startup kit location for production mode, default to None (simulation mode)",
    )
    parser.add_argument(
        "--username",
        type=str,
        default="admin@nvidia.com",
        help="Username for production mode, default to 'admin@nvidia.com'",
    )
    return parser.parse_args()


def main():
    args = define_parser()

    num_clients = args.num_clients
    num_threads = args.num_threads if args.num_threads else num_clients

    # Use workspace directory directly (SimEnv will create job-specific subdirectories)
    workspace_dir = args.workspace_dir

    # Adjust HE context paths based on environment
    if args.startup_kit_location and args.encryption:
        # In production mode, use just the filename - NVFlare's SecurityContentService
        # will resolve the path relative to each participant's workspace
        he_context_path_client = "client_context.tenseal"
        he_context_path_server = "server_context.tenseal"
    elif args.encryption:
        # In simulation mode, use the manually prepared context files
        he_context_path_client = args.he_context_path
        # Derive server path from client path using proper path manipulation
        dir_path = os.path.dirname(he_context_path_client)
        he_context_path_server = os.path.join(dir_path, "he_context_server.txt")
    else:
        # No encryption - values won't be used
        he_context_path_client = None
        he_context_path_server = None

    # Create the recipe
    recipe = KMRecipe(
        num_clients=num_clients,
        encryption=args.encryption,
        data_root=args.data_root,
        he_context_path_client=he_context_path_client,
        he_context_path_server=he_context_path_server,
    )

    # Export job
    recipe.job.export_job(args.job_dir)

    # Run recipe
    if args.startup_kit_location:
        print("\n=== Production Mode ===")
        env = ProdEnv(startup_kit_location=args.startup_kit_location, username=args.username)
    else:
        print("\n=== Simulation Mode ===")
        env = SimEnv(num_clients=num_clients, num_threads=num_threads, workspace_root=workspace_dir)

    run = recipe.execute(env)
    print(f"Job Status: {run.get_status()}")


if __name__ == "__main__":
    main()
