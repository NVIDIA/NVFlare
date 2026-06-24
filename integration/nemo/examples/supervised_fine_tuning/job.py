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
"""Recipe entrypoint for federated Nemotron 3 Nano full-model SFT with NeMo AutoModel."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import sys

from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.recipe import SimEnv

DEFAULT_MODEL_NAME_OR_PATH = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
DEFAULT_INITIAL_MODEL_CKPT = "./models/nemotron3_nano_4b_sft_init.pt"
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))


def define_parser():
    parser = argparse.ArgumentParser(description="Federated Nemotron 3 Nano full-model SFT with NVFlare Recipe API.")
    parser.add_argument("--n_clients", type=int, default=3)
    parser.add_argument("--num_rounds", type=int, default=2)
    parser.add_argument("--num_threads", type=int, default=1, help="Sequential by default to minimize GPU memory.")
    parser.add_argument("--gpu", type=str, default=None, help='Simulator GPU config, e.g. "[0]" or "[0],[1]".')
    parser.add_argument("--workspace", type=str, default="/tmp/nvflare/nemotron3_nano_sft")
    parser.add_argument("--initial_model_ckpt", type=str, default=DEFAULT_INITIAL_MODEL_CKPT)
    parser.add_argument("--model_name_or_path", type=str, default=DEFAULT_MODEL_NAME_OR_PATH)
    parser.add_argument("--data_dir", type=str, default="./data/synthetic_sft")
    parser.add_argument("--train_files", nargs="*", default=None)
    parser.add_argument("--validation_file", type=str, default="./data/synthetic_sft/validation.jsonl")
    parser.add_argument("--backend", choices=("automodel", "mock"), default="automodel")
    parser.add_argument(
        "--client_command",
        default=None,
        help="Command used by NVFlare to launch the external client script. Defaults to this Python executable.",
    )
    parser.add_argument("--automodel_command", default="automodel")
    parser.add_argument("--automodel_config_template", default=None)
    parser.add_argument("--automodel_extra_args", default="")
    parser.add_argument("--nproc_per_node", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--limit_train_samples", type=int, default=None)
    parser.add_argument("--limit_validation_samples", type=int, default=8)
    parser.add_argument("--use_chat_template", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--global_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--cp_size", type=int, default=1)
    parser.add_argument("--server_tensor_device", default="cpu")
    parser.add_argument("--mock_delta", type=float, default=0.01)
    return parser.parse_args()


def _parse_gpu_string(gpu_str: str) -> list[str]:
    if not gpu_str or not gpu_str.strip():
        return []
    return re.findall(r"\[[^\]]*\]", gpu_str.strip())


def _build_train_file(args, site_index: int) -> str:
    if args.train_files:
        if len(args.train_files) != args.n_clients:
            raise ValueError("--train_files must provide exactly one file per client.")
        return args.train_files[site_index - 1]
    return os.path.join(args.data_dir, f"site-{site_index}_train.jsonl")


def _configure_timeouts(recipe, client_names, task_timeout: int = 3600, tensor_timeout: int = 1800):
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


def _example_file(file_name: str) -> str:
    return os.path.join(EXAMPLE_DIR, file_name)


def _client_script_resource(file_name: str) -> str:
    script_path = _example_file(file_name)
    relative_path = os.path.relpath(script_path, os.getcwd())
    if relative_path.startswith(f"..{os.sep}") or relative_path == "..":
        raise RuntimeError(f"Run job.py from the example directory or repository root so {file_name} can be bundled.")
    return relative_path


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
        "--tp_size",
        str(args.tp_size),
        "--cp_size",
        str(args.cp_size),
        "--server_tensor_device",
        args.server_tensor_device,
        "--mock_delta",
        str(args.mock_delta),
    ]
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
    return shlex.join(train_args)


def _validate_inputs(args) -> None:
    if not os.path.isfile(args.initial_model_ckpt):
        raise FileNotFoundError(
            f"Initial model checkpoint not found: {args.initial_model_ckpt}. " "Run prepare_initial_model.py first."
        )
    if args.backend == "automodel":
        missing = []
        for site_idx in range(1, args.n_clients + 1):
            train_file = _build_train_file(args, site_idx)
            if not os.path.isfile(train_file):
                missing.append(train_file)
        if args.validation_file and not os.path.isfile(args.validation_file):
            missing.append(args.validation_file)
        if missing:
            raise FileNotFoundError("Missing data files:\n" + "\n".join(missing))


def create_recipe(args):
    client_names = [f"site-{idx}" for idx in range(1, args.n_clients + 1)]
    per_site_config = {}
    for site_idx, site_name in enumerate(client_names, start=1):
        train_file = _build_train_file(args, site_idx)
        per_site_config[site_name] = {"train_args": _build_train_args(args, train_file, site_name)}

    model_persistor = PTFileModelPersistor(
        source_ckpt_file_full_name=os.path.abspath(args.initial_model_ckpt),
        allow_numpy_conversion=False,
        load_device="cpu",
    )
    client_command = args.client_command or f"{shlex.quote(sys.executable)} -u"
    recipe = FedAvgRecipe(
        name="nemotron3-nano-sft",
        min_clients=args.n_clients,
        num_rounds=args.num_rounds,
        model_persistor=model_persistor,
        train_script=_client_script_resource("automodel_sft_client.py"),
        per_site_config=per_site_config,
        launch_external_process=True,
        command=client_command,
        server_expected_format=ExchangeFormat.PYTORCH,
        params_transfer_type=TransferType.FULL,
        key_metric="",
        launch_once=False,
        client_memory_gc_rounds=1,
        cuda_empty_cache=True,
    )
    recipe.add_client_file(_client_script_resource("automodel_sft_dataset.py"), clients=client_names)
    recipe.add_client_file(_client_script_resource("automodel_full_model_loader.py"), clients=client_names)
    recipe.add_client_file(_client_script_resource("model_checkpoint.py"), clients=client_names)
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
