# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare import FedJob
from nvflare.app_common.workflows.etl_controller import ETLController
from nvflare.job_config.script_runner import ScriptRunner


def main():
    args = define_parser()

    site_names = args.sites
    work_dir = args.work_dir
    job_name = args.job_name
    task_script_path = args.task_script_path
    task_script_args = args.task_script_args

    job = FedJob(name=job_name)

    enrich_ctrl = ETLController(task_name="enrich")
    job.to(enrich_ctrl, "server", id="enrich")

    # Add clients
    for site_name in site_names:
        executor = ScriptRunner(script=task_script_path, script_args=task_script_args)
        job.to(executor, site_name, tasks=["enrich"])

    if work_dir:
        print(f"{work_dir=}")
        job.export_job(work_dir)

    if not args.config_only:
        job.simulator_run(work_dir)


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--sites",
        nargs="*",  # 0 or more values expected => creates a list
        type=str,
        default=[],  # default if nothing is provided
        help="Space separated site names",
    )
    parser.add_argument(
        "-n",
        "--job_name",
        type=str,
        nargs="?",
        default="credit_card_enrich_job",
        help="job name, default to xgb_job",
    )
    parser.add_argument(
        "-w",
        "--work_dir",
        type=str,
        nargs="?",
        default="/tmp/nvflare/jobs/xgb/workdir",
        help="work directory, default to '/tmp/nvflare/jobs/xgb/workdir'",
    )

    parser.add_argument(
        "-p",
        "--task_script_path",
        type=str,
        nargs="?",
        help="task script",
    )

    parser.add_argument(
        "-a",
        "--task_script_args",
        type=str,
        nargs="?",
        default="",
        help="",
    )

    parser.add_argument("-co", "--config_only", action="store_true", help="config only mode, will not run simulator")

    return parser.parse_args()


if __name__ == "__main__":
    main()
