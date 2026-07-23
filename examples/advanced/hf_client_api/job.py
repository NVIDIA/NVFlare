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
import os
import shlex
from pathlib import Path

from model import DEFAULT_LORA_ALPHA, DEFAULT_LORA_DROPOUT, DEFAULT_LORA_R, DEFAULT_MODEL_NAME

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.client.config import ExchangeFormat
from nvflare.recipe import SimEnv, set_per_site_config

SCRIPT_DIR = Path(__file__).resolve().parent


def define_parser():
    parser = argparse.ArgumentParser(description="Qwen HuggingFace Client API example")
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=2)
    parser.add_argument("--model_name_or_path", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--train_mode", choices=("sft", "peft"), default="peft")
    parser.add_argument("--data_root", type=str, default="/tmp/nvflare/hf_client_api_qwen/data")
    parser.add_argument("--workspace_root", type=str, default="/tmp/nvflare/hf_client_api_qwen/workspace")
    parser.add_argument("--job_dir", type=str, default="/tmp/nvflare/hf_client_api_qwen/jobs/qwen_hf_client_api")
    parser.add_argument("--local_epochs", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lora_r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora_dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--export_config", action="store_true")
    return parser.parse_args()


def join_args(*args: object) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def validate_site_data(data_root: Path, client_names: list[str]):
    missing = []
    for site_name in client_names:
        for file_name in ("train.jsonl", "valid.jsonl"):
            path = data_root / site_name / file_name
            if not path.is_file():
                missing.append(str(path))

    if missing:
        prepare_data_path = shlex.quote(str(SCRIPT_DIR / "prepare_data.py"))
        data_root_arg = shlex.quote(str(data_root))
        raise FileNotFoundError(
            "Missing prepared data files:\n"
            + "\n".join(f"- {path}" for path in missing)
            + f"\nRun `python {prepare_data_path} --data_root {data_root_arg}` first."
        )


def model_config(args):
    if args.train_mode == "peft":
        return {
            "class_path": "model.QwenLoRAModel",
            "args": {
                "model_name_or_path": args.model_name_or_path,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
            },
        }

    return {
        "class_path": "model.QwenCausalLMModel",
        "args": {"model_name_or_path": args.model_name_or_path},
    }


def main():
    args = define_parser()
    client_names = [f"site-{idx}" for idx in range(1, args.n_clients + 1)]
    data_root = Path(args.data_root).expanduser().resolve()
    if not args.export_config:
        validate_site_data(data_root, client_names)

    per_site_config = {}
    for site_name in client_names:
        site_dir = data_root / site_name
        script_args = join_args(
            "--model_name_or_path",
            args.model_name_or_path,
            "--train_data",
            site_dir / "train.jsonl",
            "--eval_data",
            site_dir / "valid.jsonl",
            "--output_dir",
            f"outputs/{site_name}",
            "--train_mode",
            args.train_mode,
            "--local_epochs",
            args.local_epochs,
            "--max_length",
            args.max_length,
            "--per_device_train_batch_size",
            args.per_device_train_batch_size,
            "--gradient_accumulation_steps",
            args.gradient_accumulation_steps,
            "--learning_rate",
            args.learning_rate,
        )
        if args.train_mode == "peft":
            script_args += " " + join_args(
                "--lora_r",
                args.lora_r,
                "--lora_alpha",
                args.lora_alpha,
                "--lora_dropout",
                args.lora_dropout,
            )
        per_site_config[site_name] = {"train_args": script_args}

    original_cwd = Path.cwd()
    os.chdir(SCRIPT_DIR)
    try:
        recipe = FedAvgRecipe(
            name=f"qwen-hf-client-api-{args.train_mode}",
            model=model_config(args),
            min_clients=args.n_clients,
            num_rounds=args.num_rounds,
            train_script="client.py",
            launch_external_process=True,
            server_expected_format=ExchangeFormat.PYTORCH,
            key_metric="",
            enable_tensor_disk_offload=True,
        )
        set_per_site_config(recipe, per_site_config)
        recipe.add_client_file("model.py")

        recipe.export(args.job_dir)
    finally:
        os.chdir(original_cwd)
    print(f"Exported job to {args.job_dir}")
    if args.export_config:
        return

    env = SimEnv(clients=client_names, num_threads=args.n_clients, workspace_root=os.path.abspath(args.workspace_root))
    run = recipe.execute(env)
    print()
    print("Job Status is:", run.get_status())
    print("Result can be found in:", run.get_result())
    print()


if __name__ == "__main__":
    main()
