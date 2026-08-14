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

"""
Job configuration for LLM HuggingFace federated learning using FedAvgRecipe pattern.
"""

import argparse
import os
from typing import Dict

from nvflare.apis.job_def import JobMetaKey
from nvflare.app_opt.pt.quantization.dequantizer import ModelDequantizer
from nvflare.app_opt.pt.quantization.quantizer import ModelQuantizer
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.private.fed.utils.fed_utils import split_gpus
from nvflare.recipe import ProdEnv, SimEnv, add_experiment_tracking, set_per_site_config, set_recipe_meta


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--client_ids",
        nargs="+",
        type=str,
        default=None,
        help="Client/site names (space-separated). Used directly as site names and for data paths (e.g., 'dolly', 'hospital-1').",
    )
    parser.add_argument("--num_rounds", type=int, default=3, help="Number of FL rounds")
    parser.add_argument(
        "--workspace_dir",
        type=str,
        default="/tmp/nvflare/jobs/llm_hf/workdir",
        help="Work directory for simulator runs",
    )
    parser.add_argument("--model_name_or_path", type=str, default="EleutherAI/gpt-neo-1.3B", help="Model name or path")
    parser.add_argument("--data_path", type=str, default="", help="Root directory for training and validation data")
    parser.add_argument("--train_mode", type=str, default="SFT", help="Training mode, SFT or PEFT")
    parser.add_argument("--quantize_mode", type=str, default=None, help="Quantization mode, default None")
    parser.add_argument("--message_mode", type=str, default="numpy", help="Message mode: numpy or tensor")
    parser.add_argument("--local_epoch", type=int, default=1, help="Number of local training epochs per round")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help="Learning rate scheduler type")
    parser.add_argument("--threads", type=int, help="Number of threads for FL simulation")
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU assignments for simulated clients, comma separated, default single GPU",
    )
    parser.add_argument("--ports", nargs="+", default=["7777"], help="Ports for clients, default to 7777")
    parser.add_argument(
        "--slurm_nodes",
        type=int,
        default=None,
        help="Number of Slurm nodes per client site. Must be used with --slurm_gpus_per_node.",
    )
    parser.add_argument(
        "--slurm_gpus_per_node",
        type=int,
        default=None,
        help="Number of GPUs on each Slurm node. Must be used with --slurm_nodes.",
    )
    parser.add_argument("--startup_kit_location", type=str, default=None, help="Startup kit location")
    parser.add_argument("--username", type=str, default="admin@nvidia.com", help="Username for production mode")
    parser.add_argument(
        "--wandb_project", type=str, default="nvflare_llm", help="WandB project name (default: nvflare_llm)"
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default="nvflare_llm", help="WandB run name (default: nvflare_llm)"
    )
    parser.add_argument("--use_tracking", action="store_true", help="Enable TensorBoard tracking")
    return parser.parse_args()


def main():
    print("Starting llm_hf recipe job...")
    args = define_parser()
    print("args:", args)

    client_ids = args.client_ids
    if not client_ids:
        raise ValueError("client_ids cannot be empty. Please specify at least one client ID.")
    num_clients = len(client_ids)

    slurm_requested = args.slurm_nodes is not None or args.slurm_gpus_per_node is not None
    if slurm_requested:
        if args.slurm_nodes is None or args.slurm_gpus_per_node is None:
            raise ValueError("--slurm_nodes and --slurm_gpus_per_node must be specified together.")
        if args.slurm_nodes < 1 or args.slurm_gpus_per_node < 1:
            raise ValueError("--slurm_nodes and --slurm_gpus_per_node must be positive integers.")
        # Slurm owns GPU placement, so the simulator-only --gpu/--ports values
        # are not used to construct the client launch command.
        gpus = [[] for _ in client_ids]
        ports = []
    else:
        gpus = [g.split(",") for g in split_gpus(args.gpu)]
        ports = args.ports if isinstance(args.ports, list) else [args.ports]

    print(f"Clients: {client_ids}, GPUs: {gpus}, ports: {ports}")
    if not slurm_requested and len(gpus) != num_clients:
        raise ValueError(f"Number of GPUs ({len(gpus)}) does not match number of clients ({num_clients}).")
    if not slurm_requested and len(ports) < num_clients:
        raise ValueError(f"Number of ports ({len(ports)}) is less than number of clients ({num_clients}).")

    num_threads = args.threads if args.threads else num_clients

    # Determine train mode and model configuration
    # As LLMs can be large, we don't instantiate the full model here,
    # instead we provide a configuration dict that NVFlare uses
    # to instantiate the model on the server.
    train_mode = args.train_mode.lower()
    if train_mode == "sft":
        model = {"class_path": "hf_sft_model.CausalLMModel", "args": {"model_name_or_path": args.model_name_or_path}}
        model_file = "hf_sft_model.py"
        job_name = "llm_hf_sft"
        output_path = "sft"
    elif train_mode == "peft":
        model = {
            "class_path": "hf_peft_model.CausalLMPEFTModel",
            "args": {"model_name_or_path": args.model_name_or_path},
        }
        model_file = "hf_peft_model.py"
        job_name = "llm_hf_peft"
        output_path = "peft"
    else:
        raise ValueError(f"Invalid train_mode: {train_mode}, only SFT and PEFT are supported (case-insensitive).")

    # Determine message mode and server format
    message_mode = args.message_mode.lower()
    if message_mode == "tensor":
        server_expected_format = "pytorch"
    elif message_mode == "numpy":
        server_expected_format = "numpy"
    else:
        raise ValueError(f"Invalid message_mode: {message_mode}, only numpy and tensor are supported.")

    # Use client_ids directly as site names
    client_names = client_ids

    # Build per_site_config for multi-GPU or multi-node scenarios
    per_site_config: Dict[str, Dict] = {}
    for idx, client_id in enumerate(client_ids):
        site_name = client_names[idx]
        site_gpus = gpus[idx]
        data_path_train = os.path.join(args.data_path, client_id, "training.jsonl")
        data_path_valid = os.path.join(args.data_path, client_id, "validation.jsonl")

        # Build script arguments for this site
        script_args = (
            f"--model_name_or_path {args.model_name_or_path} "
            f"--data_path_train {data_path_train} "
            f"--data_path_valid {data_path_valid} "
            f"--output_path {output_path} "
            f"--train_mode {train_mode} "
            f"--num_rounds {args.num_rounds} "
            f"--local_epoch {args.local_epoch} "
            f"--lr_scheduler {args.lr_scheduler}"
        )

        # Add WandB arguments (will be enabled if WANDB_API_KEY is set)
        script_args += f" --wandb_project {args.wandb_project} --wandb_run_name {args.wandb_run_name}"

        # Determine command for local multi-GPU or launcher-owned Slurm runs.
        site_config = {"train_args": script_args}

        if slurm_requested:
            site_config["command"] = (
                f"python3 -m nvflare.app_opt.pt.torchrun_node --nproc-per-node={args.slurm_gpus_per_node}"
            )
        elif len(site_gpus) > 1:
            site_config["command"] = (
                f"python3 -m torch.distributed.run --nnodes=1 --nproc_per_node={len(site_gpus)} "
                f"--master_port={ports[idx]}"
            )

        per_site_config[site_name] = site_config

    # Create FedAvgRecipe
    recipe = FedAvgRecipe(
        name=job_name,
        model=model,
        min_clients=num_clients,
        num_rounds=args.num_rounds,
        train_script="client.py",
        server_expected_format=server_expected_format,
        launch_external_process=True,  # Always use external process for LLM training
        key_metric="model_score",
    )
    set_per_site_config(recipe, per_site_config)
    recipe.add_server_file(model_file)

    if slurm_requested:
        # The Slurm launcher starts one task on each node. During export,
        # ScriptRunner derives additional_node_command from each site's command
        # so non-master nodes join the same torchrun rendezvous without a wrapper.
        launcher_spec = {
            site_name: {
                "slurm": {
                    "nodes": args.slurm_nodes,
                    "gpus_per_node": args.slurm_gpus_per_node,
                }
            }
            for site_name in client_names
        }
        set_recipe_meta(recipe, JobMetaKey.JOB_LAUNCHER_SPEC, launcher_spec)

    # Add client params to reduce timeout failures for longer LLM runs
    recipe.add_client_config({"get_task_timeout": 300, "submit_task_result_timeout": 300})

    # Add quantization filters if specified
    if args.quantize_mode:
        quantizer = ModelQuantizer(quantization_type=args.quantize_mode.lower())
        dequantizer = ModelDequantizer()

        # Add to server: quantizer on output, dequantizer on input
        recipe.add_server_output_filter(quantizer, tasks=["train"])
        recipe.add_server_input_filter(dequantizer, tasks=["train"])

        # Add to all clients: quantizer on output, dequantizer on input
        recipe.add_client_output_filter(quantizer, tasks=["train"])
        recipe.add_client_input_filter(dequantizer, tasks=["train"])

    # Add experiment tracking if requested
    if args.use_tracking:
        add_experiment_tracking(recipe, tracking_type="tensorboard")

    # Run recipe
    if args.startup_kit_location:
        print("Running job in production mode...")
        env = ProdEnv(startup_kit_location=args.startup_kit_location, username=args.username)
    else:
        print("Running job in simulation mode...")
        env = SimEnv(
            clients=client_names, num_threads=num_threads, gpu_config=args.gpu, workspace_root=args.workspace_dir
        )

    run = recipe.execute(env)
    print()
    print("Job Status is:", run.get_status())
    print("Result can be found in:", run.get_result())
    print()


if __name__ == "__main__":
    main()
