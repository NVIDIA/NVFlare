# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standard Hugging Face Trainer FedAvg job template.

Copy this file beside ``client.py``, ``server_model.py``, and packaged
project-local modules such as ``model.py``. Adapt the client argument names and
capability-gated recipe options to the source project. Keep the required Recipe,
packaging, and execution structure; do not replace local file names with parent
traversal paths.
"""

import argparse
from pathlib import Path

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv

DEFAULT_MAX_STEPS = 10


def _token(value, name: str) -> str:
    text = str(value)
    if not text or any(char.isspace() for char in text):
        raise ValueError(f"{name} must be a non-empty whitespace-free value")
    return text


def build_train_args(
    model_name_or_path: str,
    data_root: str,
    num_clients: int,
    *,
    max_steps: int | None = None,
    num_train_epochs: float | None = None,
    preserve_source_budget: bool = False,
) -> str:
    """Build arguments consumed by the generated client's strict parser."""
    if preserve_source_budget and (max_steps is not None or num_train_epochs is not None):
        raise ValueError("preserve_source_budget cannot be combined with max_steps or num_train_epochs")
    if max_steps is not None and num_train_epochs is not None:
        raise ValueError("specify only one of max_steps or num_train_epochs")
    if max_steps is not None and (isinstance(max_steps, bool) or max_steps <= 0):
        raise ValueError("max_steps must be a positive integer")
    if num_train_epochs is not None and (
        isinstance(num_train_epochs, bool) or num_train_epochs <= 0
    ):
        raise ValueError("num_train_epochs must be positive")

    args = [
        "--model_name_or_path",
        _token(model_name_or_path, "model_name_or_path"),
        "--data_root",
        _token(data_root, "data_root"),
        "--num_clients",
        str(num_clients),
    ]
    if not preserve_source_budget:
        if max_steps is None and num_train_epochs is None:
            max_steps = DEFAULT_MAX_STEPS
        if max_steps is not None:
            args.extend(("--max_steps", str(max_steps)))
        else:
            args.extend(("--num_train_epochs", str(num_train_epochs)))
    return " ".join(args)


def build_recipe(
    *,
    name: str,
    model_name_or_path: str,
    data_root: str,
    num_clients: int,
    num_rounds: int,
    key_metric: str,
    max_steps: int | None = None,
    num_train_epochs: float | None = None,
    preserve_source_budget: bool = False,
    recipe_options: dict | None = None,
):
    """Build FedAvg using only options confirmed by ``recipe show``."""
    train_args = build_train_args(
        model_name_or_path,
        data_root,
        num_clients,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        preserve_source_budget=preserve_source_budget,
    )
    recipe = FedAvgRecipe(
        name=name,
        model={
            "class_path": "server_model.ServerModel",
            "args": {"model_name_or_path": model_name_or_path},
        },
        min_clients=num_clients,
        num_rounds=num_rounds,
        train_script="client.py",
        train_args=train_args,
        key_metric=key_metric,
        **(recipe_options or {}),
    )
    recipe.add_server_file("server_model.py")
    recipe.add_server_file("model.py")
    recipe.add_client_file("model.py")
    return recipe


def main():
    # Importing the Recipe API before parsing lets it consume --export and
    # --export-dir; this parser owns only the generated job's local options.
    parser = argparse.ArgumentParser(description="Hugging Face Trainer FedAvg job", allow_abbrev=False)
    parser.add_argument("--name", default="hf-trainer-fedavg")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=2)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--max_steps", type=int)
    budget.add_argument("--num_train_epochs", type=float)
    budget.add_argument("--preserve_source_budget", action="store_true")
    parser.add_argument("--key_metric", required=True)
    parser.add_argument("--workspace_root", type=Path, default=Path("/tmp/nvflare/hf-trainer"))
    args = parser.parse_args()

    recipe = build_recipe(
        name=args.name,
        model_name_or_path=args.model_name_or_path,
        data_root=args.data_root,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        preserve_source_budget=args.preserve_source_budget,
        key_metric=args.key_metric,
    )
    run = recipe.execute(
        SimEnv(
            num_clients=args.num_clients,
            num_threads=args.num_clients,
            workspace_root=str(args.workspace_root),
        )
    )
    print("Job Status is:", run.get_status())
    print("Result can be found in:", run.get_result())


if __name__ == "__main__":
    main()
