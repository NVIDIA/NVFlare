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
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.app_opt.pt.quantization.dequantizor import ModelDequantizor
from nvflare.app_opt.pt.quantization.quantizor import ModelQuantizor
from nvflare.job_config.script_runner import ScriptRunner


def main():
    args = define_parser()
    train_script = "src/hf_sft_peft_fl.py"
    client_ids = args.client_ids
    num_clients = len(client_ids)

    if args.threads:
        num_threads = args.threads
    else:
        num_threads = num_clients

    if num_threads < num_clients:
        print("The number of threads smaller than the number of clients, runner clean-up will be performed.")
        clean_up = 1
    else:
        clean_up = 0

    num_rounds = args.num_rounds
    workspace_dir = args.workspace_dir
    job_dir = args.job_dir
    model_name_or_path = args.model_name_or_path
    train_mode = args.train_mode
    message_mode = args.message_mode

    # Create the FedJob
    if train_mode.lower() == "sft":
        job = FedJob(name="llm_hf_sft", min_clients=num_clients)
        output_path = "sft"
    elif train_mode.lower() == "peft":
        job = FedJob(name="llm_hf_peft", min_clients=num_clients)
        output_path = "peft"
    else:
        raise ValueError(f"Invalid train_mode: {train_mode}, only SFT and PEFT are supported.")

    # Define the FedAvg controller workflow and send to server
    controller = FedAvg(
        num_clients=num_clients,
        num_rounds=num_rounds,
    )
    job.to(controller, "server")

    if args.quantize_mode:
        # If using quantization, add quantize filters.
        quantizor = ModelQuantizor(quantization_type=args.quantize_mode)
        dequantizor = ModelDequantizor()
        job.to(quantizor, "server", tasks=["train"], filter_type=FilterType.TASK_DATA)
        job.to(dequantizor, "server", tasks=["train"], filter_type=FilterType.TASK_RESULT)

    # Define the model persistor and send to server
    # First send the model to the server
    job.to("src/hf_peft_model.py", "server")
    # Then send the model persistor to the server
    model_args = {"path": "src.hf_peft_model.CausalLMPEFTModel", "args": {"model_name_or_path": model_name_or_path}}
    job.to(PTFileModelPersistor(model=model_args), "server", id="persistor")

    # Add model selection widget and send to server
    job.to(IntimeModelSelector(key_metric="eval_loss", negate_key_metric=True), "server", id="model_selector")

    # Send ScriptRunner to all clients
    for i in range(num_clients):
        client_id = client_ids[i]
        site_name = f"site-{client_id}"
        data_path_train = os.path.join(args.data_path, client_id, "training.jsonl")
        data_path_valid = os.path.join(args.data_path, client_id, "validation.jsonl")

        script_args = f"--model_name_or_path {model_name_or_path} --data_path_train {data_path_train} --data_path_valid {data_path_valid} --output_path {output_path} --train_mode {train_mode} --message_mode {message_mode} --clean_up {clean_up}"
        if message_mode == "tensor":
            params_exchange_format = "pytorch"
        elif message_mode == "numpy":
            params_exchange_format = "numpy"
        else:
            raise ValueError(f"Invalid message_mode: {message_mode}, only numpy and tensor are supported.")

        runner = ScriptRunner(
            script=train_script,
            script_args=script_args,
            params_exchange_format=params_exchange_format,
            launch_external_process=False,
        )
        job.to(runner, site_name, tasks=["train"])

        if args.quantize_mode:
            job.to(quantizor, site_name, tasks=["train"], filter_type=FilterType.TASK_RESULT)
            job.to(dequantizor, site_name, tasks=["train"], filter_type=FilterType.TASK_DATA)

    # Export the job
    print("job_dir=", job_dir)
    job.export_job(job_dir)

    # Run the job
    print("workspace_dir=", workspace_dir)
    print("num_threads=", num_threads)
    job.simulator_run(workspace_dir, threads=num_threads, gpu=args.gpu)


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--client_ids",
        nargs="+",
        type=str,
        default="",
        help="Clinet IDs, used to get the data path for each client",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=3,
        help="Number of rounds, default to 3",
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
        type=str,
        default="SFT",
        help="training mode, SFT or PEFT, default to SFT",
    )
    parser.add_argument(
        "--quantize_mode",
        type=str,
        default=None,
        help="quantization mode, default to None (no quantization)",
    )
    parser.add_argument(
        "--message_mode",
        type=str,
        default="numpy",
        help="message mode, numpy or tensor, default to numpy",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="number of threads to use for FL simulation, default to the number of clients",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="gpu assignments for simulating clients, comma separated, default to single gpu",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
