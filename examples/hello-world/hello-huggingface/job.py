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

import argparse
from pathlib import Path

from model import DEFAULT_MODEL_NAME

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.client.config import ExchangeFormat
from nvflare.recipe import SimEnv


def define_parser():
    parser = argparse.ArgumentParser(description="Hello HuggingFace Qwen Client API example")
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=2)
    parser.add_argument("--model_name_or_path", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--data_root", type=str, default="/tmp/nvflare/hello-huggingface/data")
    return parser.parse_args()


def validate_site_data(data_root: Path, n_clients: int):
    for site_idx in range(1, n_clients + 1):
        site_name = f"site-{site_idx}"
        for file_name in ("train.jsonl", "valid.jsonl"):
            path = data_root / site_name / file_name
            if not path.is_file():
                raise FileNotFoundError(f"Missing {path}. Run `python prepare_data.py` first.")


def main():
    args = define_parser()
    data_root = Path(args.data_root).expanduser().resolve()
    validate_site_data(data_root, args.n_clients)

    recipe = FedAvgRecipe(
        name="hello-huggingface",
        model={
            "class_path": "model.QwenLoRAModel",
            "args": {"model_name_or_path": args.model_name_or_path},
        },
        min_clients=args.n_clients,
        num_rounds=args.num_rounds,
        train_script="client.py",
        train_args=f"--model_name_or_path {args.model_name_or_path} --data_root {data_root}",
        launch_external_process=True,
        server_expected_format=ExchangeFormat.PYTORCH,  # Preserve bf16 tensors instead of converting through NumPy.
        key_metric="",  # Disable best-model selection in this API-focused example.
        enable_tensor_disk_offload=True,  # Stage large tensor payloads on disk to reduce process memory.
    )
    recipe.add_client_file("model.py")

    env = SimEnv(num_clients=args.n_clients)
    run = recipe.execute(env)
    print()
    print("Job Status is:", run.get_status())
    print("Result can be found in:", run.get_result())
    print()


if __name__ == "__main__":
    main()
