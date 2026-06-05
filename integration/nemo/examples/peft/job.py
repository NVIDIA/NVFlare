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
"""Recipe entrypoint for federated Nemotron 3 Nano PEFT with NeMo AutoModel."""

from __future__ import annotations

import argparse
import os
import re
import shlex

from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.recipe import SimEnv

DEFAULT_MODEL_NAME_OR_PATH = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
DEFAULT_INITIAL_ADAPTER_CKPT = "./models/nemotron3_nano_lora_init.pt"


def define_parser():
    parser = argparse.ArgumentParser(description="Federated Nemotron 3 Nano LoRA PEFT with NVFlare Recipe API.")
    parser.add_argument("--n_clients", type=int, default=3)
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--num_threads", type=int, default=1, help="Sequential by default to minimize GPU memory.")
    parser.add_argument("--gpu", type=str, default=None, help='Simulator GPU config, e.g. "[0]" or "[0],[1]".')
    parser.add_argument("--workspace", type=str, default="/tmp/nvflare/nemotron3_nano_peft")
    parser.add_argument("--initial_adapter_ckpt", type=str, default=DEFAULT_INITIAL_ADAPTER_CKPT)
    parser.add_argument("--model_name_or_path", type=str, default=DEFAULT_MODEL_NAME_OR_PATH)
    parser.add_argument("--train_split_dir", type=str, default="./data/FinancialPhraseBank-v1.0_split")
    parser.add_argument(
        "--validation_file",
        type=str,
        default="./data/FinancialPhraseBank-v1.0/financial_phrase_bank_val.jsonl",
    )
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--backend", choices=("automodel", "mock"), default="automodel")
    parser.add_argument("--automodel_command", default="automodel")
    parser.add_argument("--automodel_config_template", default=None)
    parser.add_argument("--automodel_extra_args", default="")
    parser.add_argument("--nproc_per_node", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--limit_train_samples", type=int, default=None)
    parser.add_argument("--limit_validation_samples", type=int, default=64)
    parser.add_argument(
        "--balance_train_labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic label-balanced sampling when limiting training samples.",
    )
    parser.add_argument(
        "--use_chat_template",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Format examples with the tokenizer chat template instead of raw prompt-completion text.",
    )
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--global_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", default="all-linear")
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--cp_size", type=int, default=1)
    parser.add_argument("--use_triton_lora", action="store_true")
    parser.add_argument(
        "--server_tensor_device",
        default="cpu",
        help="Outgoing adapter DIFF tensor device for the Client API script: auto, cpu, cuda:0, etc.",
    )
    parser.add_argument("--mock_delta", type=float, default=0.01)
    return parser.parse_args()


def _parse_gpu_string(gpu_str: str) -> list[str]:
    if not gpu_str or not gpu_str.strip():
        return []
    return re.findall(r"\[[^\]]*\]", gpu_str.strip())


def _build_train_file(train_split_dir: str, alpha: float, site_index: int) -> str:
    return os.path.join(train_split_dir, f"alpha{alpha}_site-{site_index}.jsonl")


def _configure_timeouts(recipe, client_names, task_timeout: int = 1800, tensor_timeout: int = 900):
    recipe.add_client_config(
        {
            "get_task_timeout": task_timeout,
            "submit_task_result_timeout": task_timeout,
            "tensor_min_download_timeout": tensor_timeout,
        },
        clients=client_names,
    )
    recipe.add_server_config(
        {
            "streaming_per_request_timeout": tensor_timeout,
            "tensor_min_download_timeout": tensor_timeout,
        }
    )


def _build_train_args(args, train_file: str, site_name: str) -> str:
    work_dir = os.path.join(os.path.abspath(args.workspace), "automodel_work", site_name)
    train_args = [
        "--backend",
        args.backend,
        "--model_name_or_path",
        args.model_name_or_path,
        "--train_file",
        os.path.abspath(train_file),
        "--work_dir",
        work_dir,
        "--automodel_command",
        args.automodel_command,
        "--nproc_per_node",
        str(args.nproc_per_node),
        "--max_steps",
        str(args.max_steps),
        "--seq_length",
        str(args.seq_length),
        "--limit_validation_samples",
        str(args.limit_validation_samples),
        "--learning_rate",
        str(args.learning_rate),
        "--micro_batch_size",
        str(args.micro_batch_size),
        "--global_batch_size",
        str(args.global_batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--lora_rank",
        str(args.lora_rank),
        "--lora_alpha",
        str(args.lora_alpha),
        "--lora_dropout",
        str(args.lora_dropout),
        "--target_modules",
        args.target_modules,
        "--tp_size",
        str(args.tp_size),
        "--cp_size",
        str(args.cp_size),
        "--server_tensor_device",
        args.server_tensor_device,
        "--mock_delta",
        str(args.mock_delta),
    ]
    if args.balance_train_labels:
        train_args.append("--balance_train_labels")
    else:
        train_args.append("--no-balance_train_labels")
    if args.use_chat_template:
        train_args.append("--use_chat_template")
    else:
        train_args.append("--no-use_chat_template")
    if args.validation_file:
        train_args.extend(["--validation_file", os.path.abspath(args.validation_file)])
    if args.limit_train_samples is not None:
        train_args.extend(["--limit_train_samples", str(args.limit_train_samples)])
    if args.automodel_config_template:
        train_args.extend(["--automodel_config_template", os.path.abspath(args.automodel_config_template)])
    if args.automodel_extra_args:
        train_args.extend(["--automodel_extra_args", args.automodel_extra_args])
    if args.use_triton_lora:
        train_args.append("--use_triton_lora")
    return shlex.join(train_args)


def _validate_inputs(args) -> None:
    if not os.path.isfile(args.initial_adapter_ckpt):
        raise FileNotFoundError(
            f"Initial adapter checkpoint not found: {args.initial_adapter_ckpt}. "
            "Run prepare_initial_adapter.py first."
        )
    if args.backend == "automodel":
        missing = []
        for site_idx in range(1, args.n_clients + 1):
            train_file = _build_train_file(args.train_split_dir, args.alpha, site_idx)
            if not os.path.isfile(train_file):
                missing.append(train_file)
        if args.validation_file and not os.path.isfile(args.validation_file):
            missing.append(args.validation_file)
        if missing:
            raise FileNotFoundError("Missing data files:\n" + "\n".join(missing))


def create_recipe(args):
    n_clients = args.n_clients
    client_names = [f"site-{idx}" for idx in range(1, n_clients + 1)]
    train_split_dir = os.path.abspath(args.train_split_dir)

    per_site_config = {}
    for site_idx, site_name in enumerate(client_names, start=1):
        train_file = _build_train_file(train_split_dir, args.alpha, site_idx)
        per_site_config[site_name] = {"train_args": _build_train_args(args, train_file, site_name)}

    model_persistor = PTFileModelPersistor(
        source_ckpt_file_full_name=os.path.abspath(args.initial_adapter_ckpt),
        allow_numpy_conversion=False,
        load_device="cpu",
    )
    recipe = FedAvgRecipe(
        name="nemotron3-nano-peft",
        min_clients=n_clients,
        num_rounds=args.num_rounds,
        model_persistor=model_persistor,
        train_script="automodel_peft_client.py",
        per_site_config=per_site_config,
        launch_external_process=True,
        command="python3 -u",
        server_expected_format=ExchangeFormat.PYTORCH,
        params_transfer_type=TransferType.DIFF,
        key_metric="",
        launch_once=False,
        client_memory_gc_rounds=1,
        cuda_empty_cache=True,
    )
    recipe.add_client_file("automodel_financial_phrase_dataset.py", clients=client_names)
    recipe.add_client_file("automodel_adapter_loader.py", clients=client_names)
    recipe.add_client_config({"max_resends": 3}, clients=client_names)
    _configure_timeouts(recipe, client_names)
    return recipe


def create_sim_env(args):
    client_names = [f"site-{idx}" for idx in range(1, args.n_clients + 1)]
    if args.gpu is not None:
        gpu_groups = _parse_gpu_string(args.gpu)
        if not gpu_groups:
            raise ValueError('--gpu must use bracket groups, e.g. "[0]" or "[0],[1]".')
        gpu_config = args.gpu
    else:
        gpu_config = None

    return SimEnv(
        clients=client_names,
        num_threads=args.num_threads,
        gpu_config=gpu_config,
        workspace_root=os.path.abspath(args.workspace),
    )


def main():
    args = define_parser()
    _validate_inputs(args)
    recipe = create_recipe(args)
    env = create_sim_env(args)
    run = recipe.execute(env)
    print()
    print("Job Status is:", run.get_status())
    print("Result can be found in:", run.get_result())
    print()


if __name__ == "__main__":
    main()
