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

from src.newton_raphson_persistor import NewtonRaphsonModelPersistor
from src.newton_raphson_workflow import FedAvgNewtonRaphson

from nvflare.app_common.tracking.tracker_types import ANALYTIC_EVENT_TYPE
from nvflare.app_common.widgets.convert_to_fed_event import ConvertToFedEvent
from nvflare.app_opt.tracking.tb.tb_receiver import TBAnalyticsReceiver
from nvflare.client.config import ExchangeFormat
from nvflare.job_config import FedJob
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner

if __name__ == "__main__":
    n_clients = 4
    num_rounds = 5

    # Create FedJob.
    job = FedJob(name="newton_raphson_fedavg")

    # Send custom model persistor to server.
    persistor_id = job.to_server(NewtonRaphsonModelPersistor(n_features=13), "persistor")

    # Send custom controller to server
    controller = FedAvgNewtonRaphson(
        num_clients=n_clients,
        num_rounds=num_rounds,
        damping_factor=0.8,
        persistor_id=persistor_id,
    )
    job.to(controller, "server")

    # Send TBAnalyticsReceiver to server for tensorboard streaming.
    analytics_receiver = TBAnalyticsReceiver()
    job.to_server(
        id="receiver",
        obj=analytics_receiver,
    )

    convert_to_fed_event = ConvertToFedEvent(events_to_convert=[ANALYTIC_EVENT_TYPE])

    # Add clients
    for i in range(n_clients):

        # Send ConvertToFedEvent to clients for tensorboard streaming.
        job.to(id="event_to_fed", obj=convert_to_fed_event, target=f"site-{i + 1}")

        runner = ScriptRunner(
            script="src/newton_raphson_train.py",
            script_args="--data_root /tmp/flare/dataset/heart_disease_data",
            launch_external_process=True,
            framework=FrameworkType.RAW,
            params_exchange_format=ExchangeFormat.RAW,
        )
        job.to(runner, f"site-{i + 1}")

    job.export_job("/tmp/nvflare/jobs/job_config")
    # job.simulator_run("/tmp/nvflare/jobs/workdir", gpu="0")
