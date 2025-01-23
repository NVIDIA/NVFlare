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

from src.kaplan_meier_wf import KM
from src.kaplan_meier_wf_he import KM_HE

from nvflare import FedJob
from nvflare.job_config.script_runner import ScriptRunner


def main():
    args = define_parser()
    # Default paths
    data_root = "/tmp/nvflare/dataset/km_data"
    he_context_path = "/tmp/nvflare/he_context/he_context_client.txt"

    # Set the script and config
    if args.encryption:
        job_name = "KM_HE"
        train_script = "src/kaplan_meier_train_he.py"
        script_args = f"--data_root {data_root} --he_context_path {he_context_path}"
    else:
        job_name = "KM"
        train_script = "src/kaplan_meier_train.py"
        script_args = f"--data_root {data_root}"

    # Set the number of clients and threads
    num_clients = args.num_clients
    if args.num_threads:
        num_threads = args.num_threads
    else:
        num_threads = num_clients

    # Set the output workspace and job directories
    workspace_dir = os.path.join(args.workspace_dir, job_name)
    job_dir = args.job_dir

    # Create the FedJob
    job = FedJob(name=job_name, min_clients=num_clients)

    # Define the KM controller workflow and send to server
    if args.encryption:
        controller = KM_HE(min_clients=num_clients, he_context_path=he_context_path)
    else:
        controller = KM(min_clients=num_clients)
    job.to_server(controller)

    # Define the ScriptRunner and send to all clients
    runner = ScriptRunner(
        script=train_script,
        script_args=script_args,
        params_exchange_format="raw",
        launch_external_process=False,
    )
    job.to_clients(runner, tasks=["train"])

    # Export the job
    print("job_dir=", job_dir)
    job.export_job(job_dir)

    # Run the job
    print("workspace_dir=", workspace_dir)
    print("num_threads=", num_threads)
    job.simulator_run(workspace_dir, n_clients=num_clients, threads=num_threads)


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace_dir",
        type=str,
        default="/tmp/nvflare/jobs/km/workdir",
        help="work directory, default to '/tmp/nvflare/jobs/km/workdir'",
    )
    parser.add_argument(
        "--job_dir",
        type=str,
        default="/tmp/nvflare/jobs/km/jobdir",
        help="directory for job export, default to '/tmp/nvflare/jobs/km/jobdir'",
    )
    parser.add_argument(
        "--encryption",
        action=argparse.BooleanOptionalAction,
        help="whether to enable encryption, default to False",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=5,
        help="number of clients to simulate, default to 5",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        help="number of threads to use for FL simulation, default to the number of clients if not specified",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
