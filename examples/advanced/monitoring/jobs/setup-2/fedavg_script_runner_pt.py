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

from src.simple_network import SimpleNetwork

from nvflare.app_common.widgets.convert_to_fed_event import ConvertToFedEvent
from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.fuel_opt.statsd.statsd_reporter import StatsDReporter
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.metrics.job_metrics_collector import JobMetricsCollector
from nvflare.metrics.metrics_keys import METRICS_EVENT_TYPE
from nvflare.metrics.remote_metrics_receiver import RemoteMetricsReceiver

if __name__ == "__main__":
    n_clients = 2
    num_rounds = 2
    train_script = "src/hello-pt_cifar10_fl.py"

    job = FedAvgJob(
        name="hello-pt_cifar10_fedavg", n_clients=n_clients, num_rounds=num_rounds, initial_model=SimpleNetwork()
    )
    metrics_reporter = StatsDReporter(site="server", host="localhost", port=9125)
    remote_metrics_collector = RemoteMetricsReceiver()

    server_tags = {"site": "server", "env": "dev"}
    server_job_metrics_collector = JobMetricsCollector(tags=server_tags, streaming_to_server=False)
    job.to_server(server_job_metrics_collector, id="server_job_metrics_collector")
    job.to_server(remote_metrics_collector, id="server_remote_metrics_collector")
    job.to_server(metrics_reporter, id="server_statsd_reporter")

    # Add clients
    event_convertor = ConvertToFedEvent([METRICS_EVENT_TYPE])

    for i in range(n_clients):
        executor = ScriptRunner(script=train_script, script_args="")
        client_site = f"site-{i + 1}"
        tags = {"site": client_site, "env": "dev"}
        job_metrics_collector = JobMetricsCollector(tags=tags, streaming_to_server=True)

        job.to(executor, client_site)
        job.to(job_metrics_collector, target=client_site, id=f"{client_site}_job_metrics_collector")
        job.to(event_convertor, target=client_site, id=f"{client_site}_event_convertor")

        # job.to(metrics_reporter, target=client_site, id=f"{client_site}_statsd_reporter")

    job.export_job("/tmp/nvflare/jobs/job_config")
    job.simulator_run("/tmp/nvflare/jobs/workdir", gpu="0")
