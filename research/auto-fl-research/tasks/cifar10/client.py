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
Merged NVFlare client baseline.

Provenance:
- Client API flow and optional evaluate-task branch follow the public NVFlare hello-pt pattern.
- DIFF uploads, model construction, cosine scheduling, and model-diff computation are adapted from
  the public NVFlare CIFAR-10 simulation examples.
- The surrounding mutation discipline is designed to work well with the public autoresearch
  program.md-style loop.
"""

import argparse
import copy
import os
import random
import re
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
from data.cifar10_data_utils import create_datasets  # noqa: E402
from model import (  # noqa: E402
    DEFAULT_MAX_MODEL_PARAMS,
    DEFAULT_MODEL_ARCH,
    available_model_architectures,
    build_model,
    count_parameters,
)
from train_utils import compute_model_diff, evaluate, get_lr_values  # noqa: E402

import nvflare.client as flare  # noqa: E402
from nvflare.app_common.abstract.fl_model import ParamsType  # noqa: E402
from nvflare.app_common.app_constant import AlgorithmConstants  # noqa: E402
from nvflare.app_opt.pt.fedproxloss import PTFedProxLoss  # noqa: E402
from nvflare.client.tracking import SummaryWriter  # noqa: E402

SCAFFOLD_CTRL_DIFF = AlgorithmConstants.SCAFFOLD_CTRL_DIFF
SCAFFOLD_CTRL_GLOBAL = AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_parser():
    parser = argparse.ArgumentParser(description="Merged NVFlare CIFAR10 client for Auto-FL")
    parser.add_argument("--train_idx_root", type=str, default="/tmp/cifar10_splits")
    parser.add_argument("--aggregation_epochs", type=int, default=4)
    parser.add_argument(
        "--local_train_steps",
        type=int,
        default=0,
        help="Exact optimizer steps per client per round. Use 0 for epoch-based training with --aggregation_epochs.",
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
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--no_lr_scheduler", action="store_true")
    parser.add_argument("--cosine_lr_eta_min_factor", type=float, default=0.01)
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
        "--scaffold",
        action="store_true",
        help="Enable SCAFFOLD control-variate correction using FLModel meta.",
    )
    parser.add_argument(
        "--no_deterministic_training",
        action="store_true",
        help="Disable deterministic PyTorch and DataLoader seeding for faster but noisier runs.",
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


def _zero_scaffold_controls(model):
    return {
        key: torch.zeros_like(param, device=DEVICE)
        for key, param in model.named_parameters()
        if torch.is_floating_point(param)
    }


def _scaffold_controls_match(controls, model):
    expected = _zero_scaffold_controls(model)
    return set(controls) == set(expected) and all(controls[key].shape == expected[key].shape for key in expected)


def _load_scaffold_global_controls(model, meta):
    controls = _zero_scaffold_controls(model)
    raw_controls = (meta or {}).get(SCAFFOLD_CTRL_GLOBAL) or {}
    if not raw_controls:
        return controls

    expected_keys = set(controls)
    raw_keys = set(raw_controls)
    if raw_keys != expected_keys:
        missing = sorted(expected_keys - raw_keys)
        extra = sorted(raw_keys - expected_keys)
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError("SCAFFOLD global controls do not match model parameters: " + ", ".join(details))

    for key, value in raw_controls.items():
        tensor_value = torch.as_tensor(
            value,
            dtype=controls[key].dtype,
            device=DEVICE,
        )
        if tensor_value.shape != controls[key].shape:
            raise ValueError(
                f"SCAFFOLD global control {key!r} has shape {tuple(tensor_value.shape)}, "
                f"expected {tuple(controls[key].shape)}"
            )
        controls[key] = tensor_value
    return controls


def _apply_scaffold_correction(model, curr_lr, global_controls, local_controls):
    with torch.no_grad():
        for key, param in model.named_parameters():
            if key in global_controls and key in local_controls:
                param.sub_(curr_lr * (global_controls[key] - local_controls[key]))


def _update_scaffold_controls(
    model,
    global_model,
    curr_lr,
    global_controls,
    local_controls,
    scaffold_steps,
):
    if scaffold_steps <= 0:
        raise RuntimeError("SCAFFOLD requires at least one local optimizer step")
    if curr_lr <= 0.0:
        raise RuntimeError("SCAFFOLD control update requires positive learning rate")

    model_state = model.state_dict()
    global_state = global_model.state_dict()
    new_local_controls = {}
    delta_controls = {}
    for key, local_value in local_controls.items():
        new_value = (
            local_value - global_controls[key] + (global_state[key] - model_state[key]) / (scaffold_steps * curr_lr)
        )
        delta_controls[key] = (new_value - local_value).detach().cpu().numpy()
        new_local_controls[key] = new_value.detach().clone()
    return new_local_controls, delta_controls


def main(args):
    if args.eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be > 0")
    if args.aggregation_epochs <= 0:
        raise ValueError("aggregation_epochs must be > 0")
    if args.local_train_steps < 0:
        raise ValueError("local_train_steps must be >= 0")

    flare.init()
    site_name = flare.get_site_name()
    site_seed = _site_seed(args.seed, site_name)
    deterministic_training = not args.no_deterministic_training
    _seed_everything(site_seed, deterministic=deterministic_training)
    print(f"{site_name}: seed={site_seed} " f"(base_seed={args.seed}, deterministic_training={deterministic_training})")

    model = build_model(
        model_arch=args.model_arch,
        seed=site_seed,
        max_model_params=args.max_model_params,
    )
    print(
        f"{site_name}: model_arch={args.model_arch} "
        f"params={count_parameters(model):,} max_model_params={args.max_model_params:,}"
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scheduler = None
    criterion_prox = None
    if args.fedproxloss_mu > 0:
        criterion_prox = PTFedProxLoss(mu=args.fedproxloss_mu)

    print(f"Creating datasets for site={site_name}")
    train_dataset, valid_dataset = create_datasets(
        site_name,
        train_idx_root=args.train_idx_root,
    )
    train_loader, valid_loader = _create_seeded_data_loaders(
        train_dataset,
        valid_dataset,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=site_seed,
    )

    summary_writer = SummaryWriter()
    scaffold_local_controls = None

    while flare.is_running():
        input_model = flare.receive()
        current_round = input_model.current_round
        print(f"\n[site={site_name}] round={current_round}\n")

        if scheduler is None and not args.no_lr_scheduler:
            total_rounds = input_model.total_rounds
            scheduler_units = args.local_train_steps if args.local_train_steps > 0 else args.aggregation_epochs
            eta_min = args.lr * args.cosine_lr_eta_min_factor
            t_max = total_rounds * scheduler_units
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=t_max,
                eta_min=eta_min,
            )
            print(f"{site_name}: CosineAnnealingLR init " f"(initial_lr={args.lr}, eta_min={eta_min}, T_max={t_max})")

        model.load_state_dict(input_model.params, strict=True)

        model.to(DEVICE)

        if flare.is_evaluate():
            print(f"{site_name}: cross-site evaluation task")
            val_acc_global_model = evaluate(model, valid_loader, DEVICE)
            print(f"{site_name}: global validation accuracy={100 * val_acc_global_model:.2f}%")
            summary_writer.add_scalar(
                tag="val_acc_global_model",
                scalar=val_acc_global_model,
                global_step=current_round,
            )
            # Cross-site validation expects a metrics-only DXO (DataKind.METRICS).
            # Sending no params lets FLModelUtils emit DataKind.METRICS instead of WEIGHT_DIFF.
            flare.send(
                flare.FLModel(
                    metrics={"accuracy": val_acc_global_model},
                    meta={"NUM_STEPS_CURRENT_ROUND": 0},
                )
            )
            continue

        global_model = copy.deepcopy(model)
        for p in global_model.parameters():
            p.requires_grad = False
        global_model.to(DEVICE)

        scaffold_global_controls = None
        scaffold_ctrl_diff = None
        scaffold_steps = 0
        if args.scaffold:
            if scaffold_local_controls is None or not _scaffold_controls_match(scaffold_local_controls, model):
                scaffold_local_controls = _zero_scaffold_controls(model)
            scaffold_global_controls = _load_scaffold_global_controls(model, input_model.meta)

        metrics = {}
        if args.eval_global_every_round:
            val_acc_global_model = evaluate(global_model, valid_loader, DEVICE)
            metrics["accuracy"] = val_acc_global_model
            print(f"{site_name}: global validation accuracy={100 * val_acc_global_model:.2f}%")
            summary_writer.add_scalar(
                tag="val_acc_global_model",
                scalar=val_acc_global_model,
                global_step=current_round,
            )

        train_batches = len(train_loader)
        if train_batches == 0:
            raise ValueError(
                f"{site_name}: training data_loader is empty for round {current_round}; "
                "check site_idx and data split configuration."
            )

        steps = args.local_train_steps if args.local_train_steps > 0 else args.aggregation_epochs * train_batches
        curr_lr = get_lr_values(optimizer)[0]

        if args.local_train_steps > 0:
            model.train()
            running_loss = 0.0
            loader_iter = iter(train_loader)

            for step in range(args.local_train_steps):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(train_loader)
                    batch = next(loader_iter)

                inputs, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if criterion_prox is not None:
                    loss = loss + criterion_prox(model, global_model)

                loss.backward()
                optimizer.step()

                curr_lr = get_lr_values(optimizer)[0]
                if args.scaffold:
                    _apply_scaffold_correction(
                        model,
                        curr_lr,
                        scaffold_global_controls,
                        scaffold_local_controls,
                    )
                    scaffold_steps += 1
                running_loss += loss.item()

                if scheduler is not None:
                    scheduler.step()

            avg_loss = running_loss / args.local_train_steps
            global_step = current_round * args.local_train_steps + args.local_train_steps - 1

            summary_writer.add_scalar("global_round", current_round, global_step)
            summary_writer.add_scalar("local_train_steps", args.local_train_steps, global_step)
            summary_writer.add_scalar("train_loss", avg_loss, global_step)
            summary_writer.add_scalar("learning_rate", curr_lr, global_step)

            print(
                f"{site_name}: local steps [{args.local_train_steps}/{args.local_train_steps}] "
                f"loss={avg_loss:.4f} lr={curr_lr:.6f}"
            )

            if args.evaluate_local:
                val_acc_local_model = evaluate(model, valid_loader, DEVICE)
                print(f"{site_name}: local validation accuracy={100 * val_acc_local_model:.2f}%")
                summary_writer.add_scalar(
                    tag="val_acc_local_model",
                    scalar=val_acc_local_model,
                    global_step=global_step,
                )
        else:
            for epoch in range(args.aggregation_epochs):
                model.train()
                running_loss = 0.0

                for batch in train_loader:
                    inputs, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)

                    optimizer.zero_grad(set_to_none=True)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if criterion_prox is not None:
                        loss = loss + criterion_prox(model, global_model)

                    loss.backward()
                    optimizer.step()

                    curr_lr = get_lr_values(optimizer)[0]
                    if args.scaffold:
                        _apply_scaffold_correction(
                            model,
                            curr_lr,
                            scaffold_global_controls,
                            scaffold_local_controls,
                        )
                        scaffold_steps += 1
                    running_loss += loss.item()

                avg_loss = running_loss / train_batches
                global_epoch = current_round * args.aggregation_epochs + epoch

                summary_writer.add_scalar("global_round", current_round, global_epoch)
                summary_writer.add_scalar("global_epoch", global_epoch, global_epoch)
                summary_writer.add_scalar("train_loss", avg_loss, global_epoch)
                summary_writer.add_scalar("learning_rate", curr_lr, global_epoch)

                print(
                    f"{site_name}: epoch [{epoch + 1}/{args.aggregation_epochs}] "
                    f"loss={avg_loss:.4f} lr={curr_lr:.6f}"
                )

                if args.evaluate_local:
                    val_acc_local_model = evaluate(model, valid_loader, DEVICE)
                    print(f"{site_name}: local validation accuracy={100 * val_acc_local_model:.2f}%")
                    summary_writer.add_scalar(
                        tag="val_acc_local_model",
                        scalar=val_acc_local_model,
                        global_step=global_epoch,
                    )

                if scheduler is not None:
                    scheduler.step()

        if args.scaffold:
            scaffold_local_controls, scaffold_ctrl_diff = _update_scaffold_controls(
                model,
                global_model,
                curr_lr,
                scaffold_global_controls,
                scaffold_local_controls,
                scaffold_steps,
            )

        print(f"{site_name}: finished training for round {current_round}")

        if args.save_local_ckpt:
            ckpt_path = f"./model_{site_name}_round{current_round}.pt"
            torch.save(model.cpu().state_dict(), ckpt_path)
            model.to(DEVICE)

        model_diff, diff_norm = compute_model_diff(model, global_model)
        summary_writer.add_scalar(
            tag="diff_norm",
            scalar=diff_norm.item() if hasattr(diff_norm, "item") else float(diff_norm),
            global_step=current_round,
        )

        output_meta = {
            "NUM_STEPS_CURRENT_ROUND": steps,
            "site_name": site_name,
        }
        if scaffold_ctrl_diff is not None:
            output_meta[SCAFFOLD_CTRL_DIFF] = scaffold_ctrl_diff

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
