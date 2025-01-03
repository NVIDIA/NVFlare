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

from model.simple_network import SimpleNetwork

from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.fuel_opt.prometheus.statsd_reporter import StatsDReporter
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.metrics.system_metrics_collector import SysMetricsCollector
from nvflare.metrics.job_metrics_collector import JobMetricsCollector

if __name__ == "__main__":
    n_clients = 2
    num_rounds = 2
    train_script = "src/hello-pt_cifar10_fl.py"

    job = FedAvgJob(
        name="hello-pt_cifar10_fedavg", n_clients=n_clients, num_rounds=num_rounds, initial_model=SimpleNetwork()
    )
    
    metric_reporter = StatsDReporter("localhost", 8125)
    server_job_metrics_collector = JobMetricsCollector(tags=["server"]) 
    job.to_server(metric_reporter)
    job.to_server(server_job_metrics_collector)


    # Add clients
    for i in range(n_clients):
        executor = ScriptRunner(
            script=train_script, script_args=""  # f"--batch_size 32 --data_path /tmp/data/site-{i}"
        )
        client_job_metrics_collector = JobMetricsCollector(tags=[f"site-{i + 1}"]) 
        job.to_client(executor, f"site-{i + 1}")
        job.to_client(metric_reporter)
        
    
    job.export_job("/tmp/nvflare/jobs/job_config")
    job.simulator_run("/tmp/nvflare/jobs/workdir", gpu="0")
    
