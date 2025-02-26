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

from src.nlp_models import BertModel, GPTModel

from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.model import PTModel
from nvflare.job_config.api import FedJob
from nvflare.job_config.script_runner import ScriptRunner


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="Bert",
        help="Which model to choose, either Bert or GPT",
    )
    return parser.parse_args()


def main():
    args = define_parser()
    model_name = args.model_name

    # Create the FedJob
    if model_name.lower() == "bert":
        num_clients = 4
        job = FedJob(name="Bert", min_clients=num_clients)
        train_model_name = "bert-base-uncased"
        model = PTModel(BertModel(num_labels=3, model_name=train_model_name))
        output_path = "Bert"
    elif model_name.lower() == "gpt":
        num_clients = 2
        job = FedJob(name="GPT", min_clients=num_clients)
        train_model_name = "gpt2"
        model = PTModel(GPTModel(num_labels=3, model_name=train_model_name))
        output_path = "GPT"
    else:
        raise ValueError(f"Invalid model_name: {model_name}, only Bert and GPT are supported.")

    # Local training parameters
    num_rounds = 5
    dataset_path = f"/tmp/nvflare/dataset/nlp_ner/{num_clients}_split"
    train_script = "src/nlp_fl.py"
    train_args = f"--dataset_path {dataset_path} --model_name {train_model_name}"

    # Define the controller workflow and send to server
    controller = FedAvg(
        num_clients=num_clients,
        num_rounds=num_rounds,
    )
    job.to_server(controller)

    # Define the initial global model and send to server
    job.to_server(model)
    job.to(IntimeModelSelector(key_metric="eval_acc"), "server")

    # Add executor to clients
    executor = ScriptRunner(script=train_script, script_args=train_args)
    job.to_clients(executor)

    # Export job config and run the job
    job.export_job("/tmp/nvflare/workspace/jobs/")
    job.simulator_run(f"/tmp/nvflare/workspace/works/{output_path}", n_clients=num_clients, gpu="0")


if __name__ == "__main__":
    main()
