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
"""NVFlare Client API script for federated NeMo AutoModel LoRA PEFT."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
from collections import OrderedDict
from string import Template
from typing import Any, Mapping

import adapter_checkpoint
import torch

import nvflare.client as flare
from nvflare.apis.fl_constant import FLMetaKey

DEFAULT_MODEL_NAME_OR_PATH = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
DEFAULT_TARGET_MODULES = "all-linear"


def define_parser():
    parser = argparse.ArgumentParser(description="Federated NeMo AutoModel PEFT client.")
    parser.add_argument("--backend", choices=("automodel", "mock"), default="automodel")
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_NAME_OR_PATH)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--validation_file", default=None)
    parser.add_argument("--work_dir", default="./automodel_peft_work")
    parser.add_argument("--automodel_command", default="automodel")
    parser.add_argument(
        "--automodel_config_template",
        default=None,
        help=(
            "Optional YAML template. Supported placeholders: ${model_name_or_path}, ${train_file}, "
            "${validation_file}, ${checkpoint_dir}, ${incoming_adapter_dir}, ${output_adapter_dir}, "
            "${seq_length}, ${limit_train_samples}, ${limit_validation_samples}, ${balance_train_labels}, "
            "${use_chat_template}, ${learning_rate}, ${micro_batch_size}, ${global_batch_size}, "
            "${gradient_accumulation_steps}, ${lora_rank}, ${lora_alpha}, ${lora_dropout}, ${target_modules}."
        ),
    )
    parser.add_argument(
        "--automodel_extra_args",
        default="",
        help="Additional shell-split arguments appended to the AutoModel command.",
    )
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
    parser.add_argument("--target_modules", default=DEFAULT_TARGET_MODULES)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--cp_size", type=int, default=1)
    parser.add_argument("--use_triton_lora", action="store_true")
    parser.add_argument(
        "--server_tensor_device",
        default="cpu",
        help=(
            "Device for outgoing adapter tensors. The default keeps simulator server aggregation on CPU. "
            "Use 'auto' to match CUDA visibility, or set cuda:0 explicitly if the server should receive GPU tensors."
        ),
    )
    parser.add_argument("--mock_delta", type=float, default=0.01)
    return parser.parse_args()


def _read_metrics(metrics_file: str) -> dict[str, float]:
    if not os.path.isfile(metrics_file):
        return {}
    with open(metrics_file) as f:
        data = json.load(f)
    metrics = {}
    for key, value in data.items():
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
    return metrics


def _estimated_limit_train_samples(args) -> int | None:
    if args.limit_train_samples is not None:
        return args.limit_train_samples
    if args.max_steps <= 0:
        return None
    return args.max_steps * max(1, args.global_batch_size)


def _split_target_modules(target_modules: str) -> list[str]:
    if target_modules == "all-linear":
        return []
    return [item.strip() for item in target_modules.split(",") if item.strip()]


def _build_peft_config(args) -> dict[str, Any]:
    config = {
        "_target_": "nemo_automodel.components._peft.lora.PeftConfig",
        "target_modules": _split_target_modules(args.target_modules),
        "dim": args.lora_rank,
        "alpha": args.lora_alpha,
        "dropout": args.lora_dropout,
        "use_triton": args.use_triton_lora,
    }
    if args.target_modules == "all-linear":
        config["match_all_linear"] = True
    return config


def _dataset_factory_target() -> str:
    helper_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "automodel_financial_phrase_dataset.py")
    return f"{helper_path}:make_financial_phrase_dataset"


def _model_factory_target() -> str:
    helper_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "automodel_adapter_loader.py")
    return f"{helper_path}:from_pretrained_with_adapter"


def _default_automodel_config(args, checkpoint_dir: str, incoming_adapter_dir: str) -> dict[str, Any]:
    dataset_factory_target = _dataset_factory_target()
    validation = {
        "_target_": dataset_factory_target,
        "data_file": args.validation_file,
        "seq_length": args.seq_length,
        "limit_dataset_samples": args.limit_validation_samples,
        "padding": False,
        "truncation": True,
    }
    config = {
        "recipe": "TrainFinetuneRecipeForNextTokenPrediction",
        "model": {
            "_target_": _model_factory_target(),
            "pretrained_model_name_or_path": args.model_name_or_path,
            "incoming_adapter_dir": incoming_adapter_dir,
            "trust_remote_code": True,
        },
        "peft": _build_peft_config(args),
        "dataset": {
            "_target_": dataset_factory_target,
            "data_file": args.train_file,
            "seq_length": args.seq_length,
            "limit_dataset_samples": _estimated_limit_train_samples(args),
            "balance_labels": args.balance_train_labels,
            "use_chat_template": args.use_chat_template,
            "padding": False,
            "truncation": True,
        },
        "step_scheduler": {
            "num_epochs": 1,
            "max_steps": max(1, args.max_steps),
            "global_batch_size": args.global_batch_size,
            "local_batch_size": args.micro_batch_size,
            "ckpt_every_steps": max(1, args.max_steps),
            "val_every_steps": max(1, args.max_steps),
        },
        "dataloader": {
            "_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
            "collate_fn": "nemo_automodel.components.datasets.utils.default_collater",
            "batch_size": args.micro_batch_size,
            "shuffle": True,
        },
        "loss_fn": {"_target_": "nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy"},
        "optimizer": {"_target_": "torch.optim.Adam", "lr": args.learning_rate, "weight_decay": 0},
        "checkpoint": {
            "enabled": True,
            "checkpoint_dir": checkpoint_dir,
            "model_save_format": "safetensors",
            "save_consolidated": False,
        },
        "distributed": {
            "strategy": "fsdp2",
            "dp_size": None,
            "tp_size": args.tp_size,
            "cp_size": args.cp_size,
            "sequence_parallel": False,
        },
    }
    if args.validation_file:
        config["validation_dataset"] = validation
        config["validation_dataloader"] = {
            "_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
            "collate_fn": "nemo_automodel.components.datasets.utils.default_collater",
            "batch_size": args.micro_batch_size,
        }
    return config


def _template_context(args, checkpoint_dir: str, incoming_adapter_dir: str, output_adapter_dir: str) -> dict[str, str]:
    return {
        "model_name_or_path": args.model_name_or_path,
        "train_file": args.train_file,
        "validation_file": args.validation_file or "",
        "checkpoint_dir": checkpoint_dir,
        "incoming_adapter_dir": incoming_adapter_dir,
        "output_adapter_dir": output_adapter_dir,
        "seq_length": str(args.seq_length),
        "max_steps": str(max(1, args.max_steps)),
        "limit_train_samples": str(_estimated_limit_train_samples(args) or ""),
        "limit_validation_samples": str(args.limit_validation_samples or ""),
        "balance_train_labels": str(args.balance_train_labels).lower(),
        "use_chat_template": str(args.use_chat_template).lower(),
        "learning_rate": str(args.learning_rate),
        "micro_batch_size": str(args.micro_batch_size),
        "global_batch_size": str(args.global_batch_size),
        "gradient_accumulation_steps": str(args.gradient_accumulation_steps),
        "lora_rank": str(args.lora_rank),
        "lora_alpha": str(args.lora_alpha),
        "lora_dropout": str(args.lora_dropout),
        "target_modules": args.target_modules,
    }


def _write_automodel_config(args, round_dir: str, incoming_adapter_dir: str, output_adapter_dir: str) -> str:
    checkpoint_dir = os.path.join(round_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    config_path = os.path.join(round_dir, "finetune_config.yaml")

    if args.automodel_config_template:
        with open(args.automodel_config_template) as f:
            rendered = Template(f.read()).safe_substitute(
                _template_context(args, checkpoint_dir, incoming_adapter_dir, output_adapter_dir)
            )
        with open(config_path, "w") as f:
            f.write(rendered)
        return config_path

    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("PyYAML is required to write the default NeMo AutoModel config.") from e

    config = _default_automodel_config(args, checkpoint_dir, incoming_adapter_dir)
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return config_path


def _latest_adapter_dir(checkpoint_dir: str) -> str | None:
    if not os.path.isdir(checkpoint_dir):
        return None
    candidates = []
    for root, _dirs, files in os.walk(checkpoint_dir):
        if "adapter_model.safetensors" in files or "pytorch_model.bin" in files or "adapter_model.bin" in files:
            candidates.append(root)
    if not candidates:
        return None
    candidates.sort(key=_adapter_dir_sort_key, reverse=True)
    return candidates[0]


def _adapter_dir_sort_key(path: str) -> tuple[int, float, str]:
    match = re.search(r"(\d+)$", os.path.basename(path))
    step = int(match.group(1)) if match else -1
    return step, os.path.getmtime(path), path


def _build_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{script_dir}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else script_dir
    return env


def _run_automodel_round(args, round_dir: str, incoming_state: Mapping[str, torch.Tensor]) -> tuple[dict, dict, int]:
    incoming_adapter_dir = os.path.join(round_dir, "incoming_adapter")
    output_adapter_dir = os.path.join(round_dir, "output_adapter")
    checkpoint_dir = os.path.join(round_dir, "checkpoints")
    adapter_config = {
        "base_model_name_or_path": args.model_name_or_path,
        "r": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": args.target_modules,
    }
    adapter_checkpoint.save_hf_adapter_state_dir(incoming_state, incoming_adapter_dir, adapter_config=adapter_config)
    config_path = _write_automodel_config(args, round_dir, incoming_adapter_dir, output_adapter_dir)

    command = [args.automodel_command]
    if args.nproc_per_node > 1:
        command.append(f"--nproc-per-node={args.nproc_per_node}")
    command.append(config_path)
    if args.automodel_extra_args:
        extra_args = Template(args.automodel_extra_args).safe_substitute(
            _template_context(args, checkpoint_dir, incoming_adapter_dir, output_adapter_dir)
        )
        command.extend(shlex.split(extra_args))

    env = _build_subprocess_env()

    print(f"Running NeMo AutoModel: {shlex.join(command)}")
    subprocess.run(command, cwd=round_dir, check=True, env=env)

    adapter_dir = _latest_adapter_dir(checkpoint_dir) or output_adapter_dir
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(
            "NeMo AutoModel did not produce a PEFT adapter directory. "
            f"Looked under {checkpoint_dir} and {output_adapter_dir}."
        )

    updated_state = adapter_checkpoint.load_adapter_state(adapter_dir)
    metrics = _read_metrics(os.path.join(round_dir, "metrics.json"))
    return updated_state, metrics, max(1, args.max_steps)


def _mock_round(args, state_dict: Mapping[str, torch.Tensor]) -> tuple[dict, dict, int]:
    updated = {}
    for key, value in state_dict.items():
        if torch.is_floating_point(value):
            updated[key] = value + args.mock_delta
        else:
            updated[key] = value.clone()
    return updated, {"mock_loss": 0.0}, max(1, args.max_steps)


def _resolve_server_tensor_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _move_tensor_params(params: Mapping[str, Any], device: torch.device) -> OrderedDict:
    moved = OrderedDict()
    for key, value in params.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.detach().to(device)
        else:
            moved[key] = value
    return moved


def _build_param_update(
    updated_params: Mapping[str, Any], device: torch.device
) -> tuple[flare.ParamsType, OrderedDict]:
    return flare.ParamsType.FULL, _move_tensor_params(updated_params, device)


def main():
    args = define_parser()
    signal.signal(signal.SIGTERM, lambda _signum, _frame: sys.exit(0))
    os.makedirs(args.work_dir, exist_ok=True)
    server_tensor_device = _resolve_server_tensor_device(args.server_tensor_device)

    flare.init()
    client_name = flare.system_info().get("site_name", "unknown")
    while flare.is_running():
        input_model = flare.receive()
        if input_model is None:
            break

        current_round = input_model.current_round if input_model.current_round is not None else 0
        round_dir = os.path.abspath(os.path.join(args.work_dir, f"{client_name}_round_{current_round}"))
        os.makedirs(round_dir, exist_ok=True)
        incoming_state = adapter_checkpoint.strip_model_prefix(input_model.params or {})
        print(
            f"site={client_name}, round={current_round}, "
            f"received_adapter_mb={adapter_checkpoint.state_dict_size_mb(incoming_state):.2f}"
        )

        if args.backend == "mock":
            updated_state, metrics, steps = _mock_round(args, incoming_state)
        else:
            updated_state, metrics, steps = _run_automodel_round(args, round_dir, incoming_state)

        updated_state = adapter_checkpoint.match_adapter_state_to_reference(updated_state, incoming_state)
        if not updated_state:
            raise RuntimeError("No common adapter keys between the received and updated adapter states.")
        updated_params = adapter_checkpoint.add_model_prefix(updated_state)
        params_type, params = _build_param_update(updated_params, server_tensor_device)
        meta = {FLMetaKey.NUM_STEPS_CURRENT_ROUND: steps}
        flare.send(flare.FLModel(params_type=params_type, params=params, metrics=metrics, meta=meta))
        print(
            f"site={client_name}, round={current_round}, "
            f"sent_adapter_mb={adapter_checkpoint.state_dict_size_mb(params):.2f}, steps={steps}"
        )


if __name__ == "__main__":
    main()
