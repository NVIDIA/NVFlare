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

"""
NVFlare client for the local 3-site medical VLM adapter workload.

Provenance:
- Client API flow and optional evaluate-task branch follow the public NVFlare hello-pt pattern.
- DIFF uploads, model construction, and model-diff computation follow NVFlare PyTorch examples.
- The surrounding mutation discipline is designed to work well with the public autoresearch
  program.md-style loop.
"""

import argparse
import copy
import os
import random
import re

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.optim as optim  # noqa: E402
from data.med_vlm_data_utils import (  # noqa: E402
    DEFAULT_SITE_DATASETS,
    DEFAULT_VLM_REPO_ROOT,
    create_vlm_datasets,
    create_vlm_train_collator,
)
from model import (  # noqa: E402
    DEFAULT_MAX_MODEL_PARAMS,
    DEFAULT_MODEL_ARCH,
    QWEN3VL_ADAPTER_SHAPE_FIELDS,
    adapter_state_to_peft_state,
    available_model_architectures,
    build_model,
    count_parameters,
    peft_state_to_adapter_state,
    resolve_qwen3vl_adapter_shape,
)
from train_utils import compute_model_diff, evaluate_vlm_generative, get_lr_values  # noqa: E402

import nvflare.client as flare  # noqa: E402
from nvflare.app_common.abstract.fl_model import ParamsType  # noqa: E402
from nvflare.client.tracking import SummaryWriter  # noqa: E402

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LORA_MODULE_CHOICES = ("q_proj", "k_proj", "v_proj", "o_proj")


def build_parser():
    parser = argparse.ArgumentParser(description="NVFlare Auto-FL client for 3-site medical VLM adapter FL")
    parser.add_argument("--task", choices=["med-vlm"], default="med-vlm")
    parser.add_argument("--vlm_repo_root", type=str, default=os.environ.get("VLM_BENCHMARK_ROOT", DEFAULT_VLM_REPO_ROOT))
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--hf_cache_dir", type=str, default=os.environ.get("HF_HOME"))
    parser.add_argument("--site_datasets", type=str, default=DEFAULT_SITE_DATASETS)
    parser.add_argument("--max_samples_per_site", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--reserve_validation_from_train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aggregation_epochs", type=int, default=4)
    parser.add_argument(
        "--local_train_steps",
        type=int,
        default=0,
        help="Exact optimizer steps per client per round. Use 0 for epoch-based training with --aggregation_epochs.",
    )
    parser.add_argument(
        "--site_local_steps_spec",
        type=str,
        default="",
        help=(
            "Optional comma-separated per-site exact local optimizer steps for med-vlm, "
            "for example site-1:8,site-2:8,site-3:12. Requires --local_train_steps > 0."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model_arch",
        type=str,
        default=DEFAULT_MODEL_ARCH,
        choices=available_model_architectures(),
        help="Registered model architecture to instantiate on every client.",
    )
    parser.add_argument(
        "--max_model_params",
        type=int,
        default=DEFAULT_MAX_MODEL_PARAMS,
        help="Maximum allowed model parameters for architecture-search campaigns. Use 0 to disable.",
    )
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--site_lr_scale_spec",
        type=str,
        default="",
        help="Optional comma-separated per-site LR multipliers, for example site-1:1,site-2:1,site-3:1.2.",
    )
    parser.add_argument(
        "--site_lr_scale_end_spec",
        type=str,
        default="",
        help=(
            "Optional per-site LR multipliers to linearly decay toward over "
            "--site_lr_scale_decay_rounds for med-vlm clients."
        ),
    )
    parser.add_argument(
        "--site_lr_scale_decay_rounds",
        type=int,
        default=0,
        help="Number of rounds for med-vlm per-site LR scale decay. 0 disables decay.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--evaluate_local", action="store_true")
    parser.add_argument(
        "--eval_global_every_round",
        action="store_true",
        help="Evaluate the received global model on every training round for telemetry.",
    )
    parser.add_argument("--save_local_ckpt", action="store_true")
    parser.add_argument(
        "--fedproxloss_mu",
        type=float,
        default=0.0,
        help="FedProx proximal-loss coefficient. 0 disables the proximal term.",
    )
    parser.add_argument(
        "--feddyn_alpha",
        type=float,
        default=0.0,
        help=(
            "FedDyn-local dynamic regularization coefficient for med-vlm LoRA training. "
            "0 disables the client-local drift state."
        ),
    )
    parser.add_argument(
        "--sam_rho",
        type=float,
        default=0.0,
        help="Sharpness-aware minimization rho for med-vlm LoRA local training. 0 disables SAM.",
    )
    parser.add_argument(
        "--sam_eps",
        type=float,
        default=1e-12,
        help="Numerical stabilizer for SAM gradient norm.",
    )
    parser.add_argument(
        "--no_deterministic_training",
        action="store_true",
        help="Disable deterministic PyTorch and DataLoader seeding for faster but noisier runs.",
    )
    parser.add_argument("--max_pixels", type=int, default=50176)
    parser.add_argument("--min_pixels", type=int, default=784)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--train_lora_modules",
        type=str,
        default=",".join(LORA_MODULE_CHOICES),
        help="Comma-separated Qwen attention LoRA modules to train locally, or 'all'.",
    )
    parser.add_argument("--adapter_num_hidden_layers", type=int, default=0)
    parser.add_argument("--adapter_hidden_size", type=int, default=0)
    parser.add_argument("--adapter_num_key_value_heads", type=int, default=0)
    parser.add_argument("--adapter_head_dim", type=int, default=0)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn_implementation", choices=["flash_attention_2", "sdpa", "eager"], default="sdpa")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument(
        "--prediction_audit_samples",
        type=int,
        default=0,
        help="Print the first N generative validation predictions per client. Default 0 disables audit logging.",
    )
    return parser


def _site_seed(base_seed, site_name):
    match = re.search(r"(\d+)$", site_name or "")
    if match:
        return base_seed + max(0, int(match.group(1)) - 1)
    return base_seed + sum(ord(ch) for ch in site_name or "")


def _seed_everything(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    elif torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _create_seeded_data_loaders(
    train_dataset,
    valid_dataset,
    batch_size,
    eval_batch_size,
    num_workers,
    seed,
    train_collate_fn=None,
):
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=_make_generator(seed),
        collate_fn=train_collate_fn,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=_make_generator(seed + 1),
    )
    return train_loader, valid_loader


def _resolve_attn(requested: str) -> str:
    if requested == "flash_attention_2":
        import importlib.util

        if importlib.util.find_spec("flash_attn") is None:
            print("flash_attn not installed; falling back to sdpa")
            return "sdpa"
    return requested


def _resolve_vlm_dtype(args):
    if args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _parse_train_lora_modules(spec: str):
    normalized = (spec or "all").strip()
    if normalized.lower() in {"all", "*"}:
        return LORA_MODULE_CHOICES
    modules = tuple(item.strip() for item in normalized.split(",") if item.strip())
    if not modules:
        raise ValueError("--train_lora_modules must name at least one module or 'all'")
    invalid = [module for module in modules if module not in LORA_MODULE_CHOICES]
    if invalid:
        raise ValueError(
            "--train_lora_modules contains invalid module(s): "
            f"{', '.join(invalid)}; expected choices: {', '.join(LORA_MODULE_CHOICES)}"
        )
    return modules


def _parse_site_float_spec(spec: str, arg_name: str):
    normalized = (spec or "").strip()
    if not normalized:
        return {}
    parsed = {}
    for item in normalized.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"{arg_name} entry {item!r} must use site:value format")
        site, raw_value = item.split(":", 1)
        site = site.strip()
        if not site:
            raise ValueError(f"{arg_name} contains an empty site name")
        if site in parsed:
            raise ValueError(f"{arg_name} contains duplicate site {site!r}")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"{arg_name} has invalid value for {site}: {raw_value!r}") from exc
        if value <= 0.0:
            raise ValueError(f"{arg_name} values must be > 0; got {site}:{value}")
        parsed[site] = value
    if not parsed:
        raise ValueError(f"{arg_name} did not contain any site:value entries")
    return parsed


def _parse_site_int_spec(spec: str, arg_name: str):
    normalized = (spec or "").strip()
    if not normalized:
        return {}
    parsed = {}
    for item in normalized.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"{arg_name} entry {item!r} must use site:value format")
        site, raw_value = item.split(":", 1)
        site = site.strip()
        if not site:
            raise ValueError(f"{arg_name} contains an empty site name")
        if site in parsed:
            raise ValueError(f"{arg_name} contains duplicate site {site!r}")
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"{arg_name} has invalid integer value for {site}: {raw_value!r}") from exc
        if value <= 0:
            raise ValueError(f"{arg_name} values must be > 0; got {site}:{value}")
        parsed[site] = value
    if not parsed:
        raise ValueError(f"{arg_name} did not contain any site:value entries")
    return parsed


def _round_site_lr_scale(start_scale: float, end_scale: float, current_round, decay_rounds: int) -> float:
    if decay_rounds <= 0:
        return start_scale
    round_index = 0 if current_round is None else max(0, int(current_round))
    fraction = min(1.0, round_index / float(decay_rounds))
    return start_scale + (end_scale - start_scale) * fraction


def _apply_train_lora_module_filter(vlm_model, train_lora_modules):
    allowed = set(train_lora_modules)
    trainable = 0
    frozen = 0
    matched = 0
    for name, param in vlm_model.named_parameters():
        if "lora_" not in name:
            continue
        module_name = None
        for candidate in LORA_MODULE_CHOICES:
            if f".{candidate}." in name:
                module_name = candidate
                break
        if module_name is None:
            param.requires_grad = False
            frozen += param.numel()
            continue
        matched += param.numel()
        param.requires_grad = module_name in allowed
        if param.requires_grad:
            trainable += param.numel()
        else:
            frozen += param.numel()
    if matched == 0:
        raise ValueError("No Qwen LoRA module parameters matched the train_lora_modules filter.")
    print(
        "Trainable LoRA module filter: "
        f"modules={','.join(train_lora_modules)} trainable={trainable:,} frozen={frozen:,}"
    )


def _build_vlm_runtime(args):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    pretrained_kwargs = {}
    if args.hf_cache_dir:
        pretrained_kwargs["cache_dir"] = args.hf_cache_dir

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, **pretrained_kwargs)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    if hasattr(processor, "image_processor"):
        processor.image_processor.max_pixels = args.max_pixels
        processor.image_processor.min_pixels = args.min_pixels

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name_or_path,
        torch_dtype=_resolve_vlm_dtype(args),
        attn_implementation=_resolve_attn(args.attn_implementation),
        **pretrained_kwargs,
    )
    for param in model.parameters():
        param.requires_grad = False

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    _apply_train_lora_module_filter(model, _parse_train_lora_modules(args.train_lora_modules))
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.to(DEVICE)
    model.print_trainable_parameters()
    return processor, model


def _make_vlm_optimizer(args, vlm_model):
    params = [param for param in vlm_model.parameters() if param.requires_grad]
    if not params:
        raise ValueError("No trainable VLM adapter parameters found")
    return optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)


def _load_adapter_state_into_vlm(adapter_model, vlm_model):
    from peft.utils.save_and_load import set_peft_model_state_dict

    peft_state = adapter_state_to_peft_state(adapter_model.state_dict())
    result = set_peft_model_state_dict(vlm_model, peft_state)
    unexpected = getattr(result, "unexpected_keys", None)
    if unexpected:
        raise ValueError(f"Unexpected VLM adapter keys while loading global state: {unexpected[:4]}")


def _copy_vlm_adapter_into_state(vlm_model, adapter_model):
    from peft.utils.save_and_load import get_peft_model_state_dict

    peft_state = get_peft_model_state_dict(vlm_model)
    adapter_model.load_state_dict(peft_state_to_adapter_state(peft_state), strict=True)


def _move_batch_to_device(batch):
    return {key: value.to(DEVICE) if hasattr(value, "to") else value for key, value in batch.items()}


def _saved_adapter_key_from_trainable_name(name: str) -> str:
    return name.replace(".default.", ".")


def _vlm_prox_penalty(vlm_model, global_peft_state):
    penalty = None
    for name, param in vlm_model.named_parameters():
        if not param.requires_grad:
            continue
        key = _saved_adapter_key_from_trainable_name(name)
        ref = global_peft_state.get(key)
        if ref is None:
            continue
        diff = param.float() - ref.to(device=param.device, dtype=torch.float32)
        term = torch.sum(diff * diff)
        penalty = term if penalty is None else penalty + term
    return penalty


def _zero_vlm_feddyn_state(vlm_model):
    return {
        name: torch.zeros_like(param.detach(), dtype=torch.float32, device=DEVICE)
        for name, param in vlm_model.named_parameters()
        if param.requires_grad
    }


def _vlm_feddyn_state_matches(state, vlm_model):
    if state is None:
        return False
    expected = {
        name: tuple(param.shape)
        for name, param in vlm_model.named_parameters()
        if param.requires_grad
    }
    return set(state) == set(expected) and all(tuple(state[name].shape) == shape for name, shape in expected.items())


def _vlm_feddyn_penalty(args, vlm_model, global_peft_state, feddyn_state):
    if args.feddyn_alpha <= 0.0 or feddyn_state is None:
        return None

    linear_term = None
    prox_term = None
    for name, param in vlm_model.named_parameters():
        if not param.requires_grad:
            continue
        param_float = param.float()
        state = feddyn_state.get(name)
        if state is not None:
            term = torch.sum(param_float * state.to(device=param.device, dtype=torch.float32))
            linear_term = term if linear_term is None else linear_term + term

        key = _saved_adapter_key_from_trainable_name(name)
        ref = global_peft_state.get(key)
        if ref is None:
            continue
        diff = param_float - ref.to(device=param.device, dtype=torch.float32)
        term = torch.sum(diff * diff)
        prox_term = term if prox_term is None else prox_term + term

    penalty = None
    if prox_term is not None:
        penalty = 0.5 * args.feddyn_alpha * prox_term
    if linear_term is not None:
        penalty = -linear_term if penalty is None else penalty - linear_term
    return penalty


def _update_vlm_feddyn_state(vlm_model, global_adapter_model, feddyn_state, alpha):
    if alpha <= 0.0:
        return feddyn_state, 0.0, 0.0
    if not _vlm_feddyn_state_matches(feddyn_state, vlm_model):
        feddyn_state = _zero_vlm_feddyn_state(vlm_model)

    global_peft_state = adapter_state_to_peft_state(global_adapter_model.state_dict())
    state_norm_sq = 0.0
    drift_norm_sq = 0.0
    with torch.no_grad():
        for name, param in vlm_model.named_parameters():
            if not param.requires_grad:
                continue
            key = _saved_adapter_key_from_trainable_name(name)
            ref = global_peft_state.get(key)
            if ref is None:
                continue
            drift = param.detach().float() - ref.to(device=param.device, dtype=torch.float32)
            feddyn_state[name].sub_(alpha * drift)
            state_norm_sq += float(torch.sum(feddyn_state[name] * feddyn_state[name]).detach().cpu())
            drift_norm_sq += float(torch.sum(drift * drift).detach().cpu())

    return feddyn_state, float(state_norm_sq**0.5), float(drift_norm_sq**0.5)


def _vlm_train_loss(args, vlm_model, batch, global_peft_state, feddyn_state=None):
    batch = _move_batch_to_device(batch)
    outputs = vlm_model(**batch)
    loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
    if args.fedproxloss_mu > 0:
        prox = _vlm_prox_penalty(vlm_model, global_peft_state)
        if prox is not None:
            loss = loss + 0.5 * args.fedproxloss_mu * prox
    feddyn_penalty = _vlm_feddyn_penalty(args, vlm_model, global_peft_state, feddyn_state)
    if feddyn_penalty is not None:
        loss = loss + feddyn_penalty
    return loss


def _sam_grad_norm(vlm_model):
    norms = []
    for param in vlm_model.parameters():
        if param.requires_grad and param.grad is not None:
            norms.append(torch.norm(param.grad.detach().float(), p=2))
    if not norms:
        return torch.zeros((), device=DEVICE)
    return torch.norm(torch.stack(norms), p=2)


def _sam_first_step(vlm_model, rho, eps):
    grad_norm = _sam_grad_norm(vlm_model)
    if not torch.isfinite(grad_norm) or float(grad_norm.detach().cpu()) <= 0.0:
        return [], grad_norm

    scale = rho / (grad_norm + eps)
    perturbations = []
    with torch.no_grad():
        for param in vlm_model.parameters():
            if not param.requires_grad or param.grad is None:
                continue
            perturbation = param.grad.detach() * scale.to(device=param.device, dtype=param.dtype)
            param.add_(perturbation)
            perturbations.append((param, perturbation))
    return perturbations, grad_norm


def _sam_restore(perturbations):
    with torch.no_grad():
        for param, perturbation in perturbations:
            param.sub_(perturbation)


def _train_vlm_one_round(args, vlm_model, train_loader, optimizer, global_adapter_model, current_round, feddyn_state=None):
    if args.grad_accum <= 0:
        raise ValueError("grad_accum must be > 0")
    global_peft_state = adapter_state_to_peft_state(global_adapter_model.state_dict())
    optimizer_steps = 0
    micro_steps = 0
    running_loss = 0.0

    def backward_batches(batches, divisor):
        loss_sum = 0.0
        for batch in batches:
            loss = _vlm_train_loss(args, vlm_model, batch, global_peft_state, feddyn_state)
            loss_sum += float(loss.detach().cpu())
            (loss / divisor).backward()
        return loss_sum

    def train_sam_step(batches):
        nonlocal micro_steps, optimizer_steps, running_loss
        divisor = max(1, len(batches))
        optimizer.zero_grad(set_to_none=True)
        first_loss_sum = backward_batches(batches, divisor)
        perturbations, grad_norm = _sam_first_step(vlm_model, args.sam_rho, args.sam_eps)
        optimizer.zero_grad(set_to_none=True)
        backward_batches(batches, divisor)
        _sam_restore(perturbations)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        running_loss += first_loss_sum
        micro_steps += len(batches)
        optimizer_steps += 1
        return float(grad_norm.detach().cpu())

    def train_batch(batch):
        nonlocal micro_steps, optimizer_steps, running_loss
        loss = _vlm_train_loss(args, vlm_model, batch, global_peft_state, feddyn_state)
        running_loss += float(loss.detach().cpu())
        (loss / args.grad_accum).backward()
        micro_steps += 1
        if micro_steps % args.grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

    vlm_model.train()
    optimizer.zero_grad(set_to_none=True)
    sam_enabled = args.sam_rho > 0.0
    sam_grad_norms = []
    if args.local_train_steps > 0 and sam_enabled:
        loader_iter = iter(train_loader)
        while optimizer_steps < args.local_train_steps:
            batches = []
            for _ in range(args.grad_accum):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(train_loader)
                    batch = next(loader_iter)
                batches.append(batch)
            sam_grad_norms.append(train_sam_step(batches))
        denom = max(1, micro_steps)
    elif args.local_train_steps > 0:
        loader_iter = iter(train_loader)
        while optimizer_steps < args.local_train_steps:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                batch = next(loader_iter)
            train_batch(batch)
        denom = max(1, micro_steps)
    elif sam_enabled:
        for _ in range(args.aggregation_epochs):
            pending_batches = []
            for batch in train_loader:
                pending_batches.append(batch)
                if len(pending_batches) == args.grad_accum:
                    sam_grad_norms.append(train_sam_step(pending_batches))
                    pending_batches = []
            if pending_batches:
                sam_grad_norms.append(train_sam_step(pending_batches))
        denom = max(1, micro_steps)
    else:
        for _ in range(args.aggregation_epochs):
            for batch in train_loader:
                train_batch(batch)
        if micro_steps % args.grad_accum:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        denom = max(1, micro_steps)

    avg_loss = running_loss / denom
    sam_text = ""
    if sam_enabled:
        mean_grad_norm = sum(sam_grad_norms) / max(1, len(sam_grad_norms))
        sam_text = f" sam_rho={args.sam_rho:.4g} sam_grad_norm={mean_grad_norm:.4f}"
    feddyn_text = ""
    if args.feddyn_alpha > 0.0:
        feddyn_text = f" feddyn_alpha={args.feddyn_alpha:.4g}"
    print(
        f"VLM round {current_round}: optimizer_steps={optimizer_steps} "
        f"micro_steps={micro_steps} loss={avg_loss:.4f}{sam_text}{feddyn_text}"
    )
    return optimizer_steps, avg_loss


def main(args):
    if args.model_arch != "qwen3vl_lora_adapter":
        raise ValueError("med-vlm task requires --model_arch qwen3vl_lora_adapter")
    if args.eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be > 0")
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
    if args.site_lr_scale_decay_rounds < 0:
        raise ValueError("site_lr_scale_decay_rounds must be >= 0")
    if args.site_lr_scale_decay_rounds > 0 and not args.site_lr_scale_end_spec:
        raise ValueError("--site_lr_scale_decay_rounds > 0 requires --site_lr_scale_end_spec")
    if args.site_lr_scale_end_spec and args.site_lr_scale_decay_rounds <= 0:
        raise ValueError("--site_lr_scale_end_spec requires --site_lr_scale_decay_rounds > 0")

    adapter_shape = resolve_qwen3vl_adapter_shape(
        args.model_name_or_path,
        **{name: getattr(args, name) for name in QWEN3VL_ADAPTER_SHAPE_FIELDS},
    )
    for name, value in adapter_shape.items():
        setattr(args, name, value)
    print(
        "Using Qwen3-VL adapter shape: "
        + ", ".join(f"{name}={value}" for name, value in adapter_shape.items())
    )

    flare.init()
    site_name = flare.get_site_name()
    site_steps_by_site = _parse_site_int_spec(args.site_local_steps_spec, "--site_local_steps_spec")
    if site_steps_by_site:
        original_steps = args.local_train_steps
        args.local_train_steps = site_steps_by_site.get(site_name, original_steps)
        print(
            f"{site_name}: site_local_steps={args.local_train_steps} "
            f"(default_local_train_steps={original_steps})"
        )
    base_lr = args.lr
    lr_scale_by_site = _parse_site_float_spec(args.site_lr_scale_spec, "--site_lr_scale_spec")
    lr_scale_end_by_site = _parse_site_float_spec(args.site_lr_scale_end_spec, "--site_lr_scale_end_spec")
    site_lr_start_scale = lr_scale_by_site.get(site_name, 1.0)
    site_lr_end_scale = lr_scale_end_by_site.get(site_name, site_lr_start_scale)
    site_lr_decay_enabled = bool(lr_scale_end_by_site) and args.site_lr_scale_decay_rounds > 0
    if site_lr_decay_enabled:
        print(
            f"{site_name}: site_lr_scale_start={site_lr_start_scale:.6g} "
            f"site_lr_scale_end={site_lr_end_scale:.6g} "
            f"decay_rounds={args.site_lr_scale_decay_rounds} base_lr={base_lr:.8g}"
        )
    elif lr_scale_by_site:
        args.lr = base_lr * site_lr_start_scale
        print(
            f"{site_name}: site_lr_scale={site_lr_start_scale:.6g} "
            f"base_lr={base_lr:.8g} effective_lr={args.lr:.8g}"
        )
    site_seed = _site_seed(args.seed, site_name)
    deterministic_training = not args.no_deterministic_training
    _seed_everything(site_seed, deterministic=deterministic_training)
    print(f"{site_name}: seed={site_seed} " f"(base_seed={args.seed}, deterministic_training={deterministic_training})")

    model = build_model(
        model_arch=args.model_arch,
        seed=site_seed,
        max_model_params=args.max_model_params,
        lora_r=args.lora_r,
        **adapter_shape,
    )
    print(
        f"{site_name}: model_arch={args.model_arch} "
        f"params={count_parameters(model):,} max_model_params={args.max_model_params:,}"
    )
    print(f"{site_name}: loading local Qwen3-VL runtime")
    vlm_processor, vlm_model = _build_vlm_runtime(args)
    print("Creating VLM datasets for configured site")
    train_dataset, valid_dataset, dataset_name = create_vlm_datasets(
        site_name,
        vlm_repo_root=args.vlm_repo_root,
        site_datasets=args.site_datasets,
        hf_cache_dir=args.hf_cache_dir,
        seed=site_seed,
        max_samples_per_site=args.max_samples_per_site,
        max_eval_samples=args.max_eval_samples,
        reserve_validation_from_train=args.reserve_validation_from_train,
    )
    print("VLM datasets loaded for configured site")
    train_collate_fn = create_vlm_train_collator(vlm_processor, vlm_repo_root=args.vlm_repo_root)
    train_loader, valid_loader = _create_seeded_data_loaders(
        train_dataset,
        valid_dataset,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=site_seed,
        train_collate_fn=train_collate_fn,
    )

    summary_writer = SummaryWriter()
    feddyn_state = None

    while flare.is_running():
        input_model = flare.receive()
        current_round = input_model.current_round
        print(f"\n[site={site_name}] round={current_round}\n")

        model.load_state_dict(input_model.params, strict=True)
        _load_adapter_state_into_vlm(model, vlm_model)

        if flare.is_evaluate():
            print(f"{site_name}: cross-site evaluation task")
            val_score_global_model = evaluate_vlm_generative(
                vlm_model,
                vlm_processor,
                valid_dataset,
                batch_size=args.eval_batch_size,
                max_new_tokens=args.max_new_tokens,
                audit_samples=args.prediction_audit_samples,
                audit_prefix=f"{site_name}: prediction_audit",
                device=DEVICE,
            )
            print(f"{site_name}: global validation token_f1={val_score_global_model:.4f}")
            summary_writer.add_scalar(
                tag="val_score_global_model",
                scalar=val_score_global_model,
                global_step=current_round,
            )
            # Cross-site validation expects a metrics-only DXO (DataKind.METRICS).
            # Sending no params lets FLModelUtils emit DataKind.METRICS instead of WEIGHT_DIFF.
            flare.send(
                flare.FLModel(
                    metrics={"token_f1": val_score_global_model},
                    meta={"NUM_STEPS_CURRENT_ROUND": 0},
                )
            )
            continue

        global_model = copy.deepcopy(model)
        for p in global_model.parameters():
            p.requires_grad = False

        metrics = {}
        if args.feddyn_alpha > 0.0 and not _vlm_feddyn_state_matches(feddyn_state, vlm_model):
            feddyn_state = _zero_vlm_feddyn_state(vlm_model)
            print(
                f"{site_name}: initialized FedDyn-local state "
                f"params={len(feddyn_state)} alpha={args.feddyn_alpha:.4g}"
            )
        if args.eval_global_every_round:
            val_score_global_model = evaluate_vlm_generative(
                vlm_model,
                vlm_processor,
                valid_dataset,
                batch_size=args.eval_batch_size,
                max_new_tokens=args.max_new_tokens,
                device=DEVICE,
            )
            metrics["token_f1"] = val_score_global_model
            print(f"{site_name}: global validation token_f1={val_score_global_model:.4f}")
            summary_writer.add_scalar(
                tag="val_score_global_model",
                scalar=val_score_global_model,
                global_step=current_round,
            )

        train_batches = len(train_loader)
        if train_batches == 0:
            raise ValueError(
                f"{site_name}: training data_loader is empty for round {current_round}; "
                "check site_idx and data split configuration."
            )

        if site_lr_decay_enabled:
            lr_scale = _round_site_lr_scale(
                site_lr_start_scale,
                site_lr_end_scale,
                current_round,
                args.site_lr_scale_decay_rounds,
            )
            args.lr = base_lr * lr_scale
            print(
                f"{site_name}: round={current_round} site_lr_scale={lr_scale:.6g} "
                f"base_lr={base_lr:.8g} effective_lr={args.lr:.8g}"
            )
        optimizer = _make_vlm_optimizer(args, vlm_model)
        steps, avg_loss = _train_vlm_one_round(
            args,
            vlm_model,
            train_loader,
            optimizer,
            global_model,
            current_round,
            feddyn_state=feddyn_state,
        )
        _copy_vlm_adapter_into_state(vlm_model, model)
        if args.feddyn_alpha > 0.0:
            feddyn_state, feddyn_state_norm, feddyn_drift_norm = _update_vlm_feddyn_state(
                vlm_model,
                global_model,
                feddyn_state,
                args.feddyn_alpha,
            )
            print(
                f"{site_name}: feddyn_state_norm={feddyn_state_norm:.6f} "
                f"feddyn_drift_norm={feddyn_drift_norm:.6f}"
            )
            metrics["feddyn_state_norm"] = feddyn_state_norm
            metrics["feddyn_drift_norm"] = feddyn_drift_norm
        curr_lr = get_lr_values(optimizer)[0]
        summary_writer.add_scalar("global_round", current_round, current_round)
        summary_writer.add_scalar("local_train_steps", steps, current_round)
        summary_writer.add_scalar("train_loss", avg_loss, current_round)
        summary_writer.add_scalar("learning_rate", curr_lr, current_round)
        if args.feddyn_alpha > 0.0:
            summary_writer.add_scalar("feddyn_state_norm", metrics["feddyn_state_norm"], current_round)
            summary_writer.add_scalar("feddyn_drift_norm", metrics["feddyn_drift_norm"], current_round)
        if args.evaluate_local:
            val_score_local_model = evaluate_vlm_generative(
                vlm_model,
                vlm_processor,
                valid_dataset,
                batch_size=args.eval_batch_size,
                max_new_tokens=args.max_new_tokens,
                device=DEVICE,
            )
            print(f"{site_name}: local validation token_f1={val_score_local_model:.4f}")
            summary_writer.add_scalar(
                tag="val_score_local_model",
                scalar=val_score_local_model,
                global_step=current_round,
            )

        print(f"{site_name}: finished training for round {current_round}")

        if args.save_local_ckpt:
            ckpt_path = f"./model_{site_name}_round{current_round}.pt"
            torch.save(model.cpu().state_dict(), ckpt_path)

        model_diff, diff_norm = compute_model_diff(model, global_model)
        metrics["train_loss"] = float(avg_loss)
        metrics["diff_norm"] = diff_norm.item() if hasattr(diff_norm, "item") else float(diff_norm)
        summary_writer.add_scalar(
            tag="diff_norm",
            scalar=metrics["diff_norm"],
            global_step=current_round,
        )

        output_meta = {
            "NUM_STEPS_CURRENT_ROUND": steps,
            "site_name": site_name,
        }

        output_model = flare.FLModel(
            params=model_diff,
            params_type=ParamsType.DIFF,
            metrics=metrics,
            meta=output_meta,
        )

        flare.send(output_model)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
