# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Swarm LoRA fine-tuning job using the NVFlare Recipe API.

Usage:
    # Simulator run (in-memory sharding)
    python job.py --n_clients 4 --num_rounds 5

    # Simulator run with pre-split data from prepare_data.py
    python job.py --n_clients 4 --num_rounds 5 --data_dir /tmp/swarm_data

    # Export job directory for production deployment
    python job.py --export_dir /tmp/swarm_lora_job
"""

import argparse
import os
import shutil

from model import QwenLoRAModelWrapper

from nvflare.apis.dxo import DataKind
from nvflare.app_opt.pt.recipes.swarm import SwarmLearningRecipe
from nvflare.client.config import TransferType
from nvflare.recipe.sim_env import SimEnv

JOB_NAME = "ccwf_swarm_pt_lora"
MODEL_SIZES = {
    "0.5B": "Qwen/Qwen2.5-0.5B",
    "1.5B": "Qwen/Qwen2.5-1.5B",
}


def define_parser():
    parser = argparse.ArgumentParser(description="Swarm LoRA fine-tuning job")
    parser.add_argument("--n_clients", type=int, default=2, help="Number of clients")
    parser.add_argument("--num_rounds", type=int, default=3, help="Number of FL rounds")
    parser.add_argument("--export_dir", type=str, default="", help="Export job to this directory instead of running")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Root dir of pre-split data from prepare_data.py (e.g. /tmp/swarm_data). "
        "Leave empty to use in-memory sharding.",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="/tmp/nvflare/simulation",
        help="Root workspace directory for SimEnv (job results written to <workspace>/<job_name>)",
    )
    parser.add_argument("--local_steps", type=int, default=10, help="Gradient steps per client per round")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--max_seq_len", type=int, default=128, help="Maximum tokenized sequence length")
    parser.add_argument(
        "--model_size",
        choices=list(MODEL_SIZES.keys()),
        default="0.5B",
        help="Qwen2.5 model size to use (default: 0.5B)",
    )
    return parser.parse_args()


def main():
    args = define_parser()

    if args.n_clients < 2:
        raise ValueError("Swarm learning requires at least 2 clients.")

    model_path = MODEL_SIZES[args.model_size]

    # Build script args forwarded to the client.py subprocess
    script_args = f"--model_path {model_path}"
    if args.data_dir:
        script_args += f" --data_dir {args.data_dir}"
    script_args += f" --local_steps {args.local_steps} --batch_size {args.batch_size} --max_seq_len {args.max_seq_len}"
    script_args += f" --n_shards {args.n_clients}"

    recipe = SwarmLearningRecipe(
        name=JOB_NAME,
        model=QwenLoRAModelWrapper(model_path=model_path),
        num_rounds=args.num_rounds,
        train_script="client.py",
        train_args={"script_args": script_args},
        min_clients=args.n_clients,
        launch_external_process=True,
        cuda_empty_cache=True,
        # LoRA adapters are small — exchange full adapter state each round (FedAvg)
        expected_data_kind=DataKind.WEIGHT_DIFF,
        params_transfer_type=TransferType.DIFF,
        # Timeouts sized for large-model loading and LoRA training
        start_task_timeout=1200,
        progress_timeout=14400,
        max_status_report_interval=600,
    )

    # model.py is needed by client subprocesses (shared LoRA config constants).
    # client.py is already bundled by SwarmLearningRecipe via ScriptRunner.
    recipe.add_client_file(os.path.join(os.path.dirname(__file__), "model.py"))

    # Streaming chunk sizes for LoRA adapter transfers
    recipe.add_server_config(
        {
            "np_download_chunk_size": 20971520,
            "tensor_download_chunk_size": 20971520,
            "np_streaming_per_request_timeout": 120,
        }
    )
    recipe.add_client_config(
        {
            "np_download_chunk_size": 2097152,
            "tensor_download_chunk_size": 2097152,
            "np_streaming_per_request_timeout": 120,
        }
    )

    if args.export_dir:
        recipe.export(args.export_dir)
        print(f"Exported job to: {args.export_dir}")
        return

    job_workspace = os.path.join(args.workspace, JOB_NAME)
    if os.path.isdir(job_workspace):
        shutil.rmtree(job_workspace)

    env = SimEnv(num_clients=args.n_clients, workspace_root=args.workspace)
    recipe.execute(env)
    print(f"Simulation completed. Workspace: {args.workspace}")


if __name__ == "__main__":
    main()
