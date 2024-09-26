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


from nvflare import FedJob
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.job_config.script_runner import ScriptRunner

if __name__ == "__main__":
    n_clients = 3
    num_rounds = 7
    train_script = "src/train_fl.py"

    # Create the FedJob
    job = FedJob(name="embed_fedavg", min_clients=3, mandatory_clients=["site-1", "site-2", "site-3"])

    # Define the FedAvg controller workflow and send to server
    controller = FedAvg(
        num_clients=n_clients,
        num_rounds=num_rounds,
    )
    job.to(controller, "server")

    # Define the model persistor and send to server
    # First send the model to the server
    job.to("src/st_model.py", "server")
    # Then send the model persistor to the server
    model_args = {"path": "src.st_model.SenTransModel", "args": {"model_name": "microsoft/mpnet-base"}}
    job.to(PTFileModelPersistor(model=model_args), "server", id="persistor")

    # Add model selection widget and send to server
    job.to(IntimeModelSelector(key_metric="eval_loss", negate_key_metric=True), "server", id="model_selector")

    # Send ScriptRunner to all clients
    runner = ScriptRunner(script=train_script, script_args="--dataset_name nli")
    job.to(runner, "site-1")
    runner = ScriptRunner(script=train_script, script_args="--dataset_name squad")
    job.to(runner, "site-2")
    runner = ScriptRunner(script=train_script, script_args="--dataset_name quora")
    job.to(runner, "site-3")

    job.export_job("/tmp/embed/nvflare/job_api")
    job.simulator_run("/tmp/embed/nvflare/workspace_api")
