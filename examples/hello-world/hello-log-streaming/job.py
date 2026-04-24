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
"""
Demonstrates live log streaming from clients to the server during a federated job.

JobLogStreamer is added to each client: it tails the job's log file and streams
new bytes to the server in real time as the job runs.

JobLogReceiver is added to the server: it accepts the incoming stream chunks,
writes them to a temporary file, and hands the result to the job manager for
storage when the stream closes.
"""
import argparse

from nvflare.app_common.logging.job_log_receiver import JobLogReceiver
from nvflare.app_common.logging.job_log_streamer import JobLogStreamer
from nvflare.app_common.np.recipes.fedavg import NumpyFedAvgRecipe
from nvflare.client.config import TransferType
from nvflare.recipe import SimEnv, add_experiment_tracking


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--update_type", type=str, default="full", choices=["full", "diff"])
    parser.add_argument("--launch_process", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--export_config", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--log_config",
        type=str,
        default=None,
        help="Log config mode ('concise', 'full', 'verbose'), filepath to a log config json file, or level (info, debug, error, etc.)",
    )

    return parser.parse_args()


def main():
    args = define_parser()

    n_clients = args.n_clients
    num_rounds = args.num_rounds
    launch_process = args.launch_process

    train_args = f"--update_type {args.update_type}"
    recipe = NumpyFedAvgRecipe(
        name="hello-log-streaming",
        min_clients=n_clients,
        num_rounds=num_rounds,
        model=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        train_script="client.py",
        train_args=train_args,
        launch_external_process=launch_process,
        params_transfer_type=TransferType.FULL if args.update_type == "full" else TransferType.DIFF,
    )
    add_experiment_tracking(recipe, tracking_type="tensorboard")

    # Stream live log from each client to the server while the job is running.
    recipe.job.to_clients(JobLogStreamer())

    # Receive the streamed log on the server and store it via the job manager.
    recipe.job.to_server(JobLogReceiver())

    if args.export_config:
        job_dir = "/tmp/nvflare/jobs/job_config"
        recipe.export(job_dir)
        print(f"Job config exported to {job_dir}")
    else:
        env = SimEnv(num_clients=n_clients, log_config=args.log_config)
        run = recipe.execute(env)
        print()
        print("Result can be found in :", run.get_result())
        print("Job Status is:", run.get_status())
        print()


if __name__ == "__main__":
    main()
