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
import os

from src.simple_network import SimpleNetwork

from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.fuel_opt.statsd.statsd_reporter import StatsDReporter
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.metrics.job_metrics_collector import JobMetricsCollector


def define_parser(parser):

    parser.add_argument(
        "-j", "--job_configs_dir", type=str, default="/tmp/nvflare/jobs/job_config/", help="job configure folder"
    )
    return parser


def main(job_configs_dir):
    n_clients = 2
    num_rounds = 2
    train_script = "src/hello-pt_cifar10_fl.py"
    job_name = "hello-pt"

    job = FedAvgJob(name=job_name, n_clients=n_clients, num_rounds=num_rounds, initial_model=SimpleNetwork())

    # add server side monitoring components

    server_tags = {"site": "server", "env": "dev"}

    metrics_reporter = StatsDReporter(site="server", host="localhost", port=9125)
    metrics_collector = JobMetricsCollector(tags=server_tags, streaming_to_server=False)

    job.to_server(metrics_collector, id="server_job_metrics_collector")
    job.to_server(metrics_reporter, id="statsd_reporter")

    # Add clients
    for i in range(n_clients):
        executor = ScriptRunner(script=train_script, script_args="")
        client_site = f"site-{i + 1}"
        job.to(executor, client_site)

        # add client side monitoring components
        tags = {"site": client_site, "env": "dev"}

        metrics_collector = JobMetricsCollector(tags=tags)

        job.to(metrics_collector, target=client_site, id=f"{client_site}_job_metrics_collector")
        job.to(metrics_reporter, target=client_site, id="statsd_reporter")

    job_config_path = os.path.join(job_configs_dir, job_name)
    print(f"job config folder = {job_config_path}")

    job.export_job(job_configs_dir)
    # job.simulator_run("/tmp/nvflare/jobs/workdir", gpu="0")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="FedAvg Job Script")
    parser = define_parser(parser)
    args = parser.parse_args()

    main(args.job_configs_dir)
