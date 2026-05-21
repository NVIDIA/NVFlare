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

"""NVFlare job generator for the local 3-site medical VLM starter."""

import argparse
import os
import shlex
import sys
from pathlib import Path

PROFILE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROFILE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from tasks.shared.custom_aggregators import WeightedAggregator
from med_vlm_data_utils import DEFAULT_SITE_DATASETS, DEFAULT_VLM_REPO_ROOT, parse_site_datasets
from model import (
    DEFAULT_MODEL_ARCH,
    QWEN3VL_ADAPTER_SHAPE_FIELDS,
    available_model_architectures,
    build_model,
    count_parameters,
    resolve_qwen3vl_adapter_shape,
)

from nvflare.apis.dxo import DataKind
from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator
from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval
from nvflare.app_opt.pt.file_model_locator import PTFileModelLocator
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking


def define_parser():
    parser = argparse.ArgumentParser(description="Local medical VLM Auto-FL starter")
    parser.add_argument("--task", choices=["med-vlm"], default="med-vlm")
    parser.add_argument("--n_clients", type=int, default=3)
    parser.add_argument("--num_rounds", type=int, default=20)
    parser.add_argument("--train_script", type=str, default="client.py")
    parser.add_argument("--key_metric", type=str, default="token_f1")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--aggregator", choices=["weighted", "default"], default="weighted")

    parser.add_argument("--launch_external_process", action="store_true")
    parser.add_argument("--client_memory_gc_rounds", type=int, default=0)
    parser.add_argument("--cross_site_eval", action="store_true")
    parser.add_argument(
        "--sim_workspace_root",
        type=str,
        default=os.environ.get("AUTOFL_SIM_WORKSPACE_ROOT", "/tmp/nvflare/simulation"),
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vlm_repo_root", type=str, default=os.environ.get("VLM_BENCHMARK_ROOT", DEFAULT_VLM_REPO_ROOT))
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--hf_cache_dir", type=str, default=os.environ.get("HF_HOME"))
    parser.add_argument("--site_datasets", type=str, default=DEFAULT_SITE_DATASETS)
    parser.add_argument("--max_samples_per_site", type=int, default=512)
    parser.add_argument("--max_eval_samples", type=int, default=512)
    parser.add_argument("--reserve_validation_from_train", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--model_arch", type=str, default=DEFAULT_MODEL_ARCH, choices=available_model_architectures())
    parser.add_argument("--max_model_params", type=int, default=8_000_000)
    parser.add_argument("--aggregation_epochs", type=int, default=1)
    parser.add_argument("--local_train_steps", type=int, default=4)
    parser.add_argument("--site_local_steps_spec", type=str, default="")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--site_lr_scale_spec", type=str, default="")
    parser.add_argument("--site_lr_scale_end_spec", type=str, default="")
    parser.add_argument("--site_lr_scale_decay_rounds", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--fedproxloss_mu", type=float, default=0.0)
    parser.add_argument("--feddyn_alpha", type=float, default=0.0)
    parser.add_argument("--sam_rho", type=float, default=0.0)
    parser.add_argument("--sam_eps", type=float, default=1e-12)
    parser.add_argument("--evaluate_local", action="store_true")
    parser.add_argument("--eval_global_every_round", action="store_true")
    parser.add_argument("--no_deterministic_training", action="store_true")
    parser.add_argument("--save_local_ckpt", action="store_true")
    parser.add_argument("--final_eval_clients", type=str, default="all")

    parser.add_argument("--max_pixels", type=int, default=50176)
    parser.add_argument("--min_pixels", type=int, default=784)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--train_lora_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--adapter_num_hidden_layers", type=int, default=0)
    parser.add_argument("--adapter_hidden_size", type=int, default=0)
    parser.add_argument("--adapter_num_key_value_heads", type=int, default=0)
    parser.add_argument("--adapter_head_dim", type=int, default=0)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn_implementation", choices=["flash_attention_2", "sdpa", "eager"], default="sdpa")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    return parser.parse_args()


def parse_final_eval_clients(client_spec: str, n_clients: int):
    normalized = client_spec.strip()
    if normalized.lower() in {"all", "*"}:
        return None

    clients = []
    for item in normalized.split(","):
        client = item.strip()
        if not client:
            continue
        if client.isdigit():
            client = f"site-{client}"
        clients.append(client)

    known_clients = {f"site-{i}" for i in range(1, n_clients + 1)}
    unknown_clients = [client for client in clients if client not in known_clients]
    if unknown_clients:
        raise ValueError(
            "final_eval_clients contains clients outside this job: "
            f"{', '.join(unknown_clients)}; expected one of {', '.join(sorted(known_clients))} or 'all'"
        )
    if not clients:
        raise ValueError("final_eval_clients must name at least one client or be 'all'")
    return clients


def add_final_global_evaluation(recipe, participating_clients):
    comp_ids = getattr(recipe.job, "comp_ids", {})
    model_locator_id = comp_ids.get("locator_id", "")

    if not model_locator_id:
        persistor_id = comp_ids.get("persistor_id", "")
        if not persistor_id:
            raise ValueError("Final evaluation requires a PyTorch model persistor, but no persistor_id was found")
        model_locator_id = recipe.job.to_server(PTFileModelLocator(pt_persistor_id=persistor_id))

    recipe.job.to_server(ValidationJsonGenerator())
    recipe.job.to_server(
        CrossSiteModelEval(
            model_locator_id=model_locator_id,
            submit_model_task_name="",
            participating_clients=participating_clients,
        )
    )


def write_result_dir_sidecar(result_dir: str):
    sidecar_path = os.environ.get("AUTOFL_RESULT_DIR_FILE")
    if not sidecar_path:
        return

    path = Path(sidecar_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(f"{result_dir}\n", encoding="utf-8")
    os.replace(tmp_path, path)


def build_recipe_model(args):
    return {
        "class_path": "model.Qwen3VLLoRAAdapterState",
        "args": {
            "seed": args.seed,
            "lora_r": args.lora_r,
            "num_hidden_layers": args.adapter_num_hidden_layers,
            "hidden_size": args.adapter_hidden_size,
            "num_key_value_heads": args.adapter_num_key_value_heads,
            "head_dim": args.adapter_head_dim,
        },
    }


def build_train_args(args):
    train_args = [
        "--task",
        args.task,
        "--vlm_repo_root",
        args.vlm_repo_root,
        "--model_name_or_path",
        args.model_name_or_path,
        "--site_datasets",
        args.site_datasets,
        "--num_workers",
        args.num_workers,
        "--seed",
        args.seed,
        "--model_arch",
        args.model_arch,
        "--max_model_params",
        args.max_model_params,
        "--lr",
        args.lr,
        "--batch_size",
        args.batch_size,
        "--grad_accum",
        args.grad_accum,
        "--eval_batch_size",
        args.eval_batch_size,
        "--aggregation_epochs",
        args.aggregation_epochs,
        "--local_train_steps",
        args.local_train_steps,
        "--weight_decay",
        args.weight_decay,
        "--fedproxloss_mu",
        args.fedproxloss_mu,
        "--feddyn_alpha",
        args.feddyn_alpha,
        "--sam_rho",
        args.sam_rho,
        "--sam_eps",
        args.sam_eps,
        "--max_samples_per_site",
        args.max_samples_per_site,
        "--max_eval_samples",
        args.max_eval_samples,
        "--max_pixels",
        args.max_pixels,
        "--min_pixels",
        args.min_pixels,
        "--lora_r",
        args.lora_r,
        "--lora_alpha",
        args.lora_alpha,
        "--lora_dropout",
        args.lora_dropout,
        "--train_lora_modules",
        args.train_lora_modules,
        "--adapter_num_hidden_layers",
        args.adapter_num_hidden_layers,
        "--adapter_hidden_size",
        args.adapter_hidden_size,
        "--adapter_num_key_value_heads",
        args.adapter_num_key_value_heads,
        "--adapter_head_dim",
        args.adapter_head_dim,
        "--attn_implementation",
        args.attn_implementation,
        "--max_new_tokens",
        args.max_new_tokens,
    ]
    if args.site_local_steps_spec:
        train_args.extend(["--site_local_steps_spec", args.site_local_steps_spec])
    if args.site_lr_scale_spec:
        train_args.extend(["--site_lr_scale_spec", args.site_lr_scale_spec])
    if args.site_lr_scale_end_spec:
        train_args.extend(["--site_lr_scale_end_spec", args.site_lr_scale_end_spec])
    if args.site_lr_scale_decay_rounds:
        train_args.extend(["--site_lr_scale_decay_rounds", args.site_lr_scale_decay_rounds])
    if args.hf_cache_dir:
        train_args.extend(["--hf_cache_dir", args.hf_cache_dir])
    if not args.reserve_validation_from_train:
        train_args.append("--no-reserve_validation_from_train")
    if args.evaluate_local:
        train_args.append("--evaluate_local")
    if args.eval_global_every_round:
        train_args.append("--eval_global_every_round")
    if args.no_deterministic_training:
        train_args.append("--no_deterministic_training")
    if args.save_local_ckpt:
        train_args.append("--save_local_ckpt")
    if not args.bf16:
        train_args.append("--no-bf16")
    if not args.gradient_checkpointing:
        train_args.append("--no-gradient_checkpointing")
    return " ".join(shlex.quote(str(item)) for item in train_args)


def main():
    args = define_parser()

    if args.aggregation_epochs <= 0:
        raise ValueError("aggregation_epochs must be > 0")
    if args.local_train_steps < 0:
        raise ValueError("local_train_steps must be >= 0")
    if args.site_local_steps_spec and args.local_train_steps <= 0:
        raise ValueError("--site_local_steps_spec requires --local_train_steps > 0")
    if args.feddyn_alpha < 0.0:
        raise ValueError("feddyn_alpha must be >= 0")
    if args.sam_rho < 0.0:
        raise ValueError("sam_rho must be >= 0")
    if args.sam_eps <= 0.0:
        raise ValueError("sam_eps must be > 0")
    if args.model_arch != "qwen3vl_lora_adapter":
        raise ValueError("med-vlm requires --model_arch qwen3vl_lora_adapter")

    site_datasets = parse_site_datasets(args.site_datasets)
    if args.n_clients != len(site_datasets):
        raise ValueError(
            f"med-vlm expects --n_clients to match --site_datasets "
            f"({args.n_clients} != {len(site_datasets)}: {site_datasets})"
        )

    adapter_shape = resolve_qwen3vl_adapter_shape(
        args.model_name_or_path,
        **{name: getattr(args, name) for name in QWEN3VL_ADAPTER_SHAPE_FIELDS},
    )
    for name, value in adapter_shape.items():
        setattr(args, name, value)
    print("Resolved Qwen3-VL adapter shape.")

    seed_model = build_model(
        model_arch=args.model_arch,
        seed=args.seed,
        max_model_params=args.max_model_params,
        lora_r=args.lora_r,
        **adapter_shape,
    )
    print(
        f"Using model_arch={args.model_arch} "
        f"params={count_parameters(seed_model):,} max_model_params={args.max_model_params:,}"
    )

    job_name = args.name or f"autofl_medvlm_{'_'.join(site_datasets)}_{args.aggregator}_seed{args.seed}"
    train_script = os.path.join(os.path.dirname(__file__), args.train_script)

    recipe = FedAvgRecipe(
        name=job_name,
        min_clients=args.n_clients,
        num_rounds=args.num_rounds,
        model=build_recipe_model(args),
        train_script=train_script,
        train_args=build_train_args(args),
        aggregator=WeightedAggregator() if args.aggregator == "weighted" else None,
        key_metric=args.key_metric,
        aggregator_data_kind=DataKind.WEIGHT_DIFF,
        launch_external_process=args.launch_external_process,
        client_memory_gc_rounds=args.client_memory_gc_rounds,
    )

    add_experiment_tracking(recipe, tracking_type="tensorboard")
    if args.cross_site_eval:
        add_final_global_evaluation(recipe, parse_final_eval_clients(args.final_eval_clients, args.n_clients))

    env = SimEnv(num_clients=args.n_clients, workspace_root=args.sim_workspace_root)
    run = recipe.execute(env)
    result_dir = str(run.get_result())
    write_result_dir_sidecar(result_dir)

    print()
    print("Job finished.")
    print("Job results were written to the configured result directory.")
    print("TensorBoard logs are available in the job result directory.")
    print()


if __name__ == "__main__":
    main()
