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
        he_context_path: str = "/tmp/nvflare/he_context/he_context_client.txt",
    ):
        self.num_clients = num_clients
        self.encryption = encryption
        self.data_root = data_root
        self.he_context_path = he_context_path

        # Set job name and script based on encryption mode
        if self.encryption:
            job_name = "KM_HE"
            train_script = "client_he.py"
            script_args = f"--data_root {data_root} --he_context_path {he_context_path}"
            controller = KM_HE(min_clients=num_clients, he_context_path=he_context_path)
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
        help="Path to HE context file, default to '/tmp/nvflare/he_context/he_context_client.txt'",
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
    print("Starting Kaplan-Meier job...")
    args = define_parser()
    print("args:", args)

    num_clients = args.num_clients
    num_threads = args.num_threads if args.num_threads else num_clients

    # Determine job name for workspace directory
    job_name = "KM_HE" if args.encryption else "KM"
    workspace_dir = f"{args.workspace_dir}/{job_name}"

    # Create the recipe
    recipe = KMRecipe(
        num_clients=num_clients,
        encryption=args.encryption,
        data_root=args.data_root,
        he_context_path=args.he_context_path,
    )

    # Export job
    print("Exporting job to", args.job_dir)
    recipe.job.export_job(args.job_dir)

    # Run recipe
    if args.startup_kit_location:
        print("Running job in production mode...")
        print("startup_kit_location=", args.startup_kit_location)
        print("username=", args.username)
        env = ProdEnv(startup_kit_location=args.startup_kit_location, username=args.username)
    else:
        print("Running job in simulation mode...")
        print("workspace_dir=", workspace_dir)
        print("num_clients=", num_clients)
        print("num_threads=", num_threads)
        env = SimEnv(num_clients=num_clients, num_threads=num_threads, workspace_root=workspace_dir)

    run = recipe.execute(env)
    print("Job Status is:", run.get_status())

    # In production mode, job runs asynchronously on the FL system
    # Check status via admin console instead of waiting for result here
    if args.startup_kit_location:
        print("\nJob submitted successfully to the FL system!")
        print("To monitor job status, use the admin console:")
        print(f"  cd {args.startup_kit_location}")
        print("  ./startup/fl_admin.sh")
        print("  > check_status server")
        print("  > list_jobs")
        print(f"  > download_job {run.job_id}")
    else:
        # In simulation mode, we can get the result synchronously
        print("Job Result is:", run.get_result())


if __name__ == "__main__":
    main()
