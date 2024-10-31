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

from nvflare import FedJob, FilterType
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.compression.model_compressor import ModelCompressor
from nvflare.app_opt.compression.model_extractor import ModelExtractor
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.job_config.script_runner import ScriptRunner


def main():
    args = define_parser()
    train_script = "src/hf_sft_peft_fl.py"
    num_clients = args.num_clients
    num_rounds = args.num_rounds
    workspace_dir = args.workspace_dir
    job_dir = args.job_dir
    model_name_or_path = args.model_name_or_path
    data_path_train = os.path.join(args.data_path, "training.jsonl")
    data_path_valid = os.path.join(args.data_path, "validation.jsonl")
    train_mode = args.train_mode

    # Create the FedJob
    if train_mode == 0:
        job = FedJob(name="llm_hf_sft", min_clients=num_clients)
        output_path = "sft"
    else:
        job = FedJob(name="llm_hf_peft", min_clients=num_clients)
        output_path = "peft"

    # Define the FedAvg controller workflow and send to server
    controller = FedAvg(
        num_clients=num_clients,
        num_rounds=num_rounds,
    )
    job.to(controller, "server")
    # Add compress filters.
    extractor = ModelExtractor()
    compressor = ModelCompressor()
    job.to(compressor, "server", tasks=["train"], filter_type=FilterType.TASK_DATA)
    job.to(extractor, "server", tasks=["train"], filter_type=FilterType.TASK_RESULT)

    # Define the model persistor and send to server
    # First send the model to the server
    job.to("src/hf_sft_model.py", "server")
    # Then send the model persistor to the server
    model_args = {"path": "src.hf_sft_model.CausalLMModel", "args": {"model_name_or_path": model_name_or_path}}
    job.to(PTFileModelPersistor(model=model_args), "server", id="persistor")

    # Add model selection widget and send to server
    job.to(IntimeModelSelector(key_metric="eval_loss", negate_key_metric=True), "server", id="model_selector")

    # Send ScriptRunner to all clients
    for i in range(num_clients):
        site_name = f"site-{i}"
        runner = ScriptRunner(
            script=train_script,
            script_args=f"--model_name_or_path {model_name_or_path} --data_path_train {data_path_train} --data_path_valid {data_path_valid} --output_path {output_path} --mode {train_mode}",
        )
        job.to(runner, site_name, tasks=["train"])
        job.to(compressor, site_name, tasks=["train"], filter_type=FilterType.TASK_RESULT)
        job.to(extractor, site_name, tasks=["train"], filter_type=FilterType.TASK_DATA)

    # Export the job
    print("job_dir=", job_dir)
    job.export_job(job_dir)

    # Run the job
    print("workspace_dir=", workspace_dir)
    job.simulator_run(workspace_dir)


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_clients",
        type=int,
        default=1,
        help="Number of clients, default to 1",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=3,
        help="Number of rounds, default to 5",
    )
    parser.add_argument(
        "--workspace_dir",
        type=str,
        default="/tmp/nvflare/jobs/llm_hf/workdir",
        help="work directory, default to '/tmp/nvflare/jobs/llm_hf/workdir'",
    )
    parser.add_argument(
        "--job_dir",
        type=str,
        default="/tmp/nvflare/jobs/llm_hf/jobdir",
        help="directory for job export, default to '/tmp/nvflare/jobs/llm_hf/jobdir'",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/llama-3.2-1b",
        help="model name or path",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="",
        help="root directory for training and validation data",
    )
    parser.add_argument(
        "--train_mode",
        type=int,
        default=0,
        help="training mode, 0: SFT, 1: PEFT, default to 0",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
