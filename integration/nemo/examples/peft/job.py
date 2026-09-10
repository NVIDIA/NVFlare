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
"""Recipe entrypoint for federated Nemotron 3 PEFT with NeMo AutoModel."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex

import adapter_checkpoint
import model_profiles
from adapter_checkpoint import AdapterPTFileModelPersistor

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.recipe import SimEnv, set_per_site_config

DEFAULT_MODEL_NAME_OR_PATH = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
DEFAULT_INITIAL_ADAPTER_CKPT = "./models/nemotron3_nano_lora_init.pt"


def define_parser():
    parser = argparse.ArgumentParser(description="Federated Nemotron 3 LoRA PEFT with NVFlare Recipe API.")
    model_profiles.add_model_profile_argument(parser)
    parser.add_argument("--n_clients", type=int, default=3)
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--num_threads", type=int, default=1, help="Sequential by default to minimize GPU memory.")
    parser.add_argument("--gpu", type=str, default=None, help='Simulator GPU config, e.g. "[0]" or "[0],[1]".')
    parser.add_argument("--workspace", type=str, default=None)
    parser.add_argument("--initial_adapter_ckpt", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--model_revision", type=str, default=None)
    parser.add_argument("--tokenizer_revision", type=str, default=None)
    parser.add_argument("--train_split_dir", type=str, default="./data/FinancialPhraseBank-v1.0_split")
    parser.add_argument(
        "--validation_file",
        type=str,
        default="./data/FinancialPhraseBank-v1.0/financial_phrase_bank_val.jsonl",
    )
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--backend", choices=("automodel", "mock"), default="automodel")
    parser.add_argument("--automodel_command", default="automodel")
    parser.add_argument("--client_command", default="python3 -u")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--global_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--target_modules", default=None)
    parser.add_argument("--exclude_modules", default=None)
    parser.add_argument("--tp_size", type=int, default=None)
    parser.add_argument("--cp_size", type=int, default=None)
    parser.add_argument("--ep_size", type=int, default=None)
    parser.add_argument("--use_triton_lora", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--activation_checkpointing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--server_tensor_device",
        default="cpu",
        help="Outgoing adapter tensor device for the Client API script: auto, cpu, cuda:0, etc.",
    )
    parser.add_argument(
        "--fp32_adapter_exchange",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Cast outgoing adapter tensors to FP32 before federation. Lightning always uses FP32 exchange; "
            "this opt-in flag enables the same aggregation precision for Nano without changing its default."
        ),
    )
    parser.add_argument("--mock_delta", type=float, default=0.01)
    parser.add_argument("--mock_site_steps", default=None, help="Comma-separated per-site mock weights for CPU tests.")
    parser.add_argument("--mock_site_deltas", default=None, help="Comma-separated per-site mock tensor deltas.")
    parser.add_argument("--task_timeout", type=int, default=7200)
    parser.add_argument("--tensor_timeout", type=int, default=1800)
    return model_profiles.resolve_model_profile(parser.parse_args())


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
    args = model_profiles.resolve_model_profile(args)
    site_index = int(site_name.rsplit("-", 1)[1]) - 1
    max_steps = _site_override(getattr(args, "mock_site_steps", None), site_index, args.max_steps, int)
    mock_delta = _site_override(getattr(args, "mock_site_deltas", None), site_index, args.mock_delta, float)
    work_dir = os.path.join(os.path.abspath(args.workspace), "automodel_work", site_name)
    train_args = [
        "--backend",
        args.backend,
        "--model_profile",
        args.model_profile,
        "--model_name_or_path",
        args.model_name_or_path,
        "--tokenizer_name_or_path",
        args.tokenizer_name_or_path,
        "--train_file",
        os.path.abspath(train_file),
        "--work_dir",
        work_dir,
        "--automodel_command",
        args.automodel_command,
        "--nproc_per_node",
        str(args.nproc_per_node),
        "--max_steps",
        str(max_steps),
        "--seed",
        str(getattr(args, "seed", 42)),
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
        "--exclude_modules",
        args.exclude_modules,
        "--tp_size",
        str(args.tp_size),
        "--cp_size",
        str(args.cp_size),
        "--ep_size",
        str(args.ep_size),
        "--server_tensor_device",
        args.server_tensor_device,
        "--mock_delta",
        str(mock_delta),
    ]
    if args.fp32_adapter_exchange:
        train_args.append("--fp32_adapter_exchange")
    else:
        train_args.append("--no-fp32_adapter_exchange")
    if model_profiles.is_lightning35(args):
        train_args.extend(["--adapter_contract", "custom/adapter_contract.json"])
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
    else:
        train_args.append("--no-use_triton_lora")
    if args.activation_checkpointing:
        train_args.append("--activation_checkpointing")
    else:
        train_args.append("--no-activation_checkpointing")
    if args.model_revision:
        train_args.extend(["--model_revision", args.model_revision])
    if args.tokenizer_revision:
        train_args.extend(["--tokenizer_revision", args.tokenizer_revision])
    return shlex.join(train_args)


def _site_override(value: str | None, site_index: int, fallback, converter):
    if not value:
        return fallback
    items = [item.strip() for item in value.split(",")]
    if site_index >= len(items):
        raise ValueError(f"Missing per-site value for site index {site_index + 1}: {value}")
    result = converter(items[site_index])
    if result <= 0:
        raise ValueError(f"Per-site values must be positive: {value}")
    return result


def _profile_settings(args) -> dict:
    return model_profiles.adapter_compatibility_settings(args)


def _validate_inputs(args) -> None:
    args = model_profiles.resolve_model_profile(args)
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
    for value in (getattr(args, "mock_site_steps", None), getattr(args, "mock_site_deltas", None)):
        if value and len(value.split(",")) != args.n_clients:
            raise ValueError("Mock per-site override counts must match --n_clients.")
    if model_profiles.is_lightning35(args):
        state = adapter_checkpoint.strip_model_prefix(adapter_checkpoint.load_adapter_state(args.initial_adapter_ckpt))
        manifest = adapter_checkpoint.load_adapter_manifest(args.initial_adapter_ckpt)
        adapter_checkpoint.validate_adapter_manifest(
            manifest,
            state,
            expected={
                "model_profile": args.model_profile,
                "base_model_name_or_path": args.model_name_or_path,
                "base_model_revision": args.model_revision,
                "tokenizer_name_or_path": args.tokenizer_name_or_path,
                "tokenizer_revision": args.tokenizer_revision,
                "profile_settings": _profile_settings(args),
            },
        )


def create_recipe(args):
    args = model_profiles.resolve_model_profile(args)
    n_clients = args.n_clients
    client_names = [f"site-{idx}" for idx in range(1, n_clients + 1)]
    train_split_dir = os.path.abspath(args.train_split_dir)

    per_site_config = {}
    for site_idx, site_name in enumerate(client_names, start=1):
        train_file = _build_train_file(train_split_dir, args.alpha, site_idx)
        per_site_config[site_name] = {"train_args": _build_train_args(args, train_file, site_name)}

    manifest_template = {
        "model_profile": args.model_profile,
        "base_model_name_or_path": args.model_name_or_path,
        "base_model_revision": args.model_revision,
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "tokenizer_revision": args.tokenizer_revision,
        "profile_settings": _profile_settings(args),
    }
    model_persistor = AdapterPTFileModelPersistor(
        source_ckpt_file_full_name=os.path.abspath(args.initial_adapter_ckpt),
        allow_numpy_conversion=False,
        load_device="cpu",
        manifest_template=manifest_template,
    )
    recipe = FedAvgRecipe(
        name="nemotron35-lightning-peft" if model_profiles.is_lightning35(args) else "nemotron3-nano-peft",
        min_clients=n_clients,
        num_rounds=args.num_rounds,
        model_persistor=model_persistor,
        train_script="automodel_peft_client.py",
        launch_external_process=True,
        command=getattr(args, "client_command", "python3 -u"),
        server_expected_format=ExchangeFormat.PYTORCH,
        params_transfer_type=TransferType.FULL,
        key_metric="",
        launch_once=False,
        client_memory_gc_rounds=1,
        cuda_empty_cache=True,
    )
    set_per_site_config(recipe, per_site_config)
    recipe.add_client_file("automodel_financial_phrase_dataset.py", clients=client_names)
    recipe.add_client_file("automodel_adapter_loader.py", clients=client_names)
    recipe.add_client_file("federated_automodel_trainer.py", clients=client_names)
    recipe.add_client_file("model_profiles.py", clients=client_names)
    if model_profiles.is_lightning35(args):
        initial_manifest = adapter_checkpoint.load_adapter_manifest(args.initial_adapter_ckpt)
        contract_path = os.path.join(os.path.abspath(args.workspace), "adapter_contract.json")
        os.makedirs(os.path.dirname(contract_path), exist_ok=True)
        with open(contract_path, "w") as f:
            json.dump(initial_manifest, f, indent=2, sort_keys=True)
        recipe.add_client_file(contract_path, clients=client_names)
    recipe.add_server_file("adapter_checkpoint.py")
    _configure_timeouts(
        recipe,
        client_names,
        task_timeout=getattr(args, "task_timeout", 7200),
        tensor_timeout=getattr(args, "tensor_timeout", 1800),
    )
    return recipe


def create_sim_env(args):
    args = model_profiles.resolve_model_profile(args)
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
