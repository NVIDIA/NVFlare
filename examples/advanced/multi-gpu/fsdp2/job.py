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

"""Fine-tune Qwen with two federated clients, each using two FSDP2 ranks."""

import argparse
from pathlib import Path

from model import DEFAULT_MODEL

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.client.config import ExchangeFormat
from nvflare.recipe import SimEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--data_dir", default="/tmp/nvflare/fsdp2/data")
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--local_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gpu_config", default="[0,1],[2,3]", help="One two-GPU group per simulated client")
    args = parser.parse_args()
    if min(args.num_rounds, args.local_steps, args.eval_steps, args.batch_size) < 1:
        parser.error("rounds, steps and batch_size must be positive")

    model_path = Path(args.model_name).expanduser()
    if model_path.is_dir():
        args.model_name = str(model_path.resolve())

    recipe = FedAvgRecipe(
        name="pt_fsdp2",
        # Defer the large CPU model allocation/download until the server starts.
        model={"class_path": "model.CausalLMModel", "args": {"model_name": args.model_name}},
        min_clients=2,
        num_rounds=args.num_rounds,
        train_script="client.py",
        train_args=[
            "--model_name",
            args.model_name,
            "--data_dir",
            str(Path(args.data_dir).expanduser().resolve()),
            "--local_steps",
            str(args.local_steps),
            "--eval_steps",
            str(args.eval_steps),
            "--batch_size",
            str(args.batch_size),
        ],
        launch_external_process=True,
        command="python3 -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2",
        # Let both ranks and torchrun exit after the final stop signal.
        shutdown_timeout=10.0,
        server_expected_format=ExchangeFormat.PYTORCH,
        key_metric="",
    )
    recipe.add_server_file("model.py")
    env = SimEnv(clients=["site-1", "site-2"], num_threads=2, gpu_config=args.gpu_config)
    run = recipe.execute(env)
    print("Result:", run.get_result())
    print("Status:", run.get_status())


if __name__ == "__main__":
    main()
