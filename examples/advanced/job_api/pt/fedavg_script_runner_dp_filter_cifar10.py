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

from src.net import Net

from nvflare import FilterType
from nvflare.app_common.filters.percentile_privacy import PercentilePrivacy
from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.job_config.script_runner import ScriptRunner

if __name__ == "__main__":
    n_clients = 2
    num_rounds = 2
    train_script = "src/cifar10_fl.py"

    job = FedAvgJob(name="cifar10_fedavg_privacy", num_rounds=num_rounds, n_clients=n_clients, initial_model=Net())

    for i in range(n_clients):
        site_name = f"site-{i}"
        executor = ScriptRunner(script=train_script, script_args="")
        job.to(executor, site_name, tasks=["train"])

        # add privacy filter.
        pp_filter = PercentilePrivacy(percentile=10, gamma=0.01)
        job.to(pp_filter, site_name, tasks=["train"], filter_type=FilterType.TASK_RESULT)

    # job.export_job("/tmp/nvflare/jobs/job_config")
    job.simulator_run("/tmp/nvflare/jobs/workdir", gpu="0")
