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
from typing import List, Optional

from nvflare import FedJob, FilterType
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.app_opt.pt.quantization.dequantizer import ModelDequantizer
from nvflare.app_opt.pt.quantization.quantizer import ModelQuantizer
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.private.fed.utils.fed_utils import split_gpus
from nvflare.recipe import ProdEnv, SimEnv
from nvflare.recipe.spec import Recipe


class LLMHFRecipe(Recipe):
    """Recipe wrapper around the existing llm_hf job configuration.

    This mirrors the behavior of examples/advanced/llm_hf/job.py while exposing
    a recipe-style API similar to the hello-world pt_recipe example.
    """

    def __init__(
        self,
        *,
        client_ids: List[str],
        num_rounds: int,
        model_name_or_path: str,
        data_path: str,
        train_mode: str,
        message_mode: str,
        quantize_mode: Optional[str],
        gpus: List[List[str]],
        ports: List[str],
        multi_node: bool,
        wandb_project: Optional[str],
        wandb_run_name: Optional[str],
    ):
        self.client_ids = client_ids
        self.client_names: List[str] = []
        self.num_clients = len(client_ids)
        self.train_mode = train_mode.lower()
        self.message_mode = message_mode.lower()
        self.quantize_mode = quantize_mode.lower() if quantize_mode else None

        # Create FedJob and controller
        if self.train_mode == "sft":
            job_name = "llm_hf_sft_recipe"
            output_path = "sft"
            model_file = "hf_sft_model.py"
            model_args = {"path": "hf_sft_model.CausalLMModel", "args": {"model_name_or_path": model_name_or_path}}
        elif self.train_mode == "peft":
            job_name = "llm_hf_peft_recipe"
            output_path = "peft"
            model_file = "hf_peft_model.py"
            model_args = {"path": "hf_peft_model.CausalLMPEFTModel", "args": {"model_name_or_path": model_name_or_path}}
        else:
            raise ValueError(
                f"Invalid train_mode: {self.train_mode}, only SFT and PEFT are supported (case-insensitive)."
            )

        job = FedJob(name=job_name, min_clients=self.num_clients)
        controller = FedAvg(num_clients=self.num_clients, num_rounds=num_rounds)
        job.to(controller, "server")

        # Optional quantization filters
        quantizer = None
        dequantizer = None
        if self.quantize_mode:
            quantizer = ModelQuantizer(quantization_type=self.quantize_mode)
            dequantizer = ModelDequantizer()
            job.to(quantizer, "server", tasks=["train"], filter_type=FilterType.TASK_DATA)
            job.to(dequantizer, "server", tasks=["train"], filter_type=FilterType.TASK_RESULT)

        # Persistor and model selector on server
        allow_numpy_conversion = self.message_mode != "tensor"
        job.to(model_file, "server")
        persistor = PTFileModelPersistor(model=model_args, allow_numpy_conversion=allow_numpy_conversion)
        job.to(persistor, "server", id="persistor")
        job.to(IntimeModelSelector(key_metric="eval_loss", negate_key_metric=True), "server", id="model_selector")

        # Add client runners
        for idx, client_id in enumerate(client_ids):
            site_name = f"site-{client_id}"
            self.client_names.append(site_name)
            data_path_train = os.path.join(data_path, client_id, "training.jsonl")
            data_path_valid = os.path.join(data_path, client_id, "validation.jsonl")

            script_args = (
                f"--model_name_or_path {model_name_or_path} "
                f"--data_path_train {data_path_train} "
                f"--data_path_valid {data_path_valid} "
                f"--output_path {output_path} "
                f"--train_mode {self.train_mode} "
                f"--message_mode {self.message_mode} "
                f"--num_rounds {num_rounds}"
            )

            if wandb_project:
                run_name = wandb_run_name if wandb_run_name else f"nvflare_{self.train_mode}_{client_id}"
                script_args += f" --wandb_project {wandb_project} --wandb_run_name {run_name}"

            if self.message_mode == "tensor":
                server_expected_format = "pytorch"
            elif self.message_mode == "numpy":
                server_expected_format = "numpy"
            else:
                raise ValueError(
                    f"Invalid message_mode: {self.message_mode}, only numpy and tensor are supported (case-insensitive)."
                )

            # Default: run in-process for single-GPU unless overridden below.
            launch_external_process = False
            site_gpus = gpus[idx]
            command = None
            if multi_node:
                job.to("client_wrapper.sh", site_name)
                command = "bash custom/client_wrapper.sh"
                launch_external_process = True
            elif len(site_gpus) > 1:
                command = (
                    f"python3 -m torch.distributed.run --nnodes=1 --nproc_per_node={len(site_gpus)} "
                    f"--master_port={ports[idx]}"
                )
                launch_external_process = True

            runner = ScriptRunner(
                script="client.py",
                script_args=script_args,
                server_expected_format=server_expected_format,
                launch_external_process=launch_external_process,
                command=command,
            )
            job.to(runner, site_name, tasks=["train"])

            if quantizer and dequantizer:
                job.to(quantizer, site_name, tasks=["train"], filter_type=FilterType.TASK_RESULT)
                job.to(dequantizer, site_name, tasks=["train"], filter_type=FilterType.TASK_DATA)

            # Add client params to reduce timeout failures for longer LLM runs.
            client_params = {"get_task_timeout": 300, "submit_task_result_timeout": 300}
            job.to(client_params, site_name)

        super().__init__(job)


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_ids", nargs="+", type=str, default="", help="Client IDs, used to build data paths")
    parser.add_argument("--num_rounds", type=int, default=3, help="Number of FL rounds")
    parser.add_argument(
        "--workspace_dir",
        type=str,
        default="/tmp/nvflare/jobs/llm_hf/workdir",
        help="Work directory for simulator runs",
    )
    parser.add_argument(
        "--job_dir",
        type=str,
        default="/tmp/nvflare/jobs/llm_hf/jobdir",
        help="Directory for job export",
    )
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/llama-3.2-1b", help="Model name or path")
    parser.add_argument("--data_path", type=str, default="", help="Root directory for training and validation data")
    parser.add_argument("--train_mode", type=str, default="SFT", help="Training mode, SFT or PEFT")
    parser.add_argument("--quantize_mode", type=str, default=None, help="Quantization mode, default None")
    parser.add_argument("--message_mode", type=str, default="numpy", help="Message mode: numpy or tensor")
    parser.add_argument("--threads", type=int, help="Number of threads for FL simulation")
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU assignments for simulated clients, comma separated, default single GPU",
    )
    parser.add_argument("--ports", nargs="+", default=["7777"], help="Ports for clients, default to 7777")
    parser.add_argument("--multi_node", action="store_true", help="Enable multi-node training")
    parser.add_argument("--startup_kit_location", type=str, default=None, help="Startup kit location")
    parser.add_argument("--username", type=str, default="admin@nvidia.com", help="Username for production mode")
    parser.add_argument("--wandb_project", type=str, default=None, help="WandB project name (optional)")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name (optional)")
    return parser.parse_args()


def main():
    print("Starting llm_hf recipe job...")
    args = define_parser()
    print("args:", args)

    client_ids = args.client_ids
    num_clients = len(client_ids)
    gpus = split_gpus(args.gpu)
    gpus = [g.split(",") for g in gpus]
    ports = args.ports if isinstance(args.ports, list) else [args.ports]

    print(f"Clients: {client_ids}, GPUs: {gpus}, ports: {ports}")
    if len(gpus) != num_clients:
        raise ValueError(f"Number of GPUs ({len(gpus)}) does not match number of clients ({num_clients}).")
    if len(ports) < num_clients:
        raise ValueError(f"Number of ports ({len(ports)}) is less than number of clients ({num_clients}).")

    num_threads = args.threads if args.threads else num_clients

    recipe = LLMHFRecipe(
        client_ids=client_ids,
        num_rounds=args.num_rounds,
        model_name_or_path=args.model_name_or_path,
        data_path=args.data_path,
        train_mode=args.train_mode,
        message_mode=args.message_mode,
        quantize_mode=args.quantize_mode,
        gpus=gpus,
        ports=ports,
        multi_node=args.multi_node,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    # Export job
    print("Exporting job to", args.job_dir)
    recipe.job.export_job(args.job_dir)

    # Run recipe
    if args.startup_kit_location:
        print("Running job in production mode...")
        env = ProdEnv(startup_kit_location=args.startup_kit_location, username=args.username)
    else:
        print("Running job in simulation mode...")
        env = SimEnv(
            clients=recipe.client_names, num_threads=num_threads, gpu_config=args.gpu, workspace_root=args.workspace_dir
        )

    run = recipe.execute(env)
    print("Job Status is:", run.get_status())
    print("Job Result is:", run.get_result())


if __name__ == "__main__":
    main()
