# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standard Hugging Face Trainer FedAvg job template.

Adapt the client argument names and capability-gated recipe options to the
source project. Keep the required Recipe, packaging, and execution structure.
"""

import argparse
from pathlib import Path

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv


def _token(value, name: str) -> str:
    text = str(value)
    if not text or any(char.isspace() for char in text):
        raise ValueError(f"{name} must be a non-empty whitespace-free value")
    return text


def build_train_args(
    model_name_or_path: str,
    data_root: str,
    num_clients: int,
    num_train_epochs: float,
) -> str:
    """Build arguments consumed by the generated client's strict parser."""
    return " ".join(
        (
            "--model_name_or_path",
            _token(model_name_or_path, "model_name_or_path"),
            "--data_root",
            _token(data_root, "data_root"),
            "--num_clients",
            str(num_clients),
            "--num_train_epochs",
            str(num_train_epochs),
        )
    )


def build_recipe(
    *,
    name: str,
    model_name_or_path: str,
    data_root: str,
    num_clients: int,
    num_rounds: int,
    num_train_epochs: float,
    key_metric: str,
    recipe_options: dict | None = None,
):
    """Build FedAvg using only options confirmed by ``recipe show``."""
    train_args = build_train_args(model_name_or_path, data_root, num_clients, num_train_epochs)
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
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--key_metric", required=True)
    parser.add_argument("--workspace_root", type=Path, default=Path("/tmp/nvflare/hf-trainer"))
    args = parser.parse_args()

    recipe = build_recipe(
        name=args.name,
        model_name_or_path=args.model_name_or_path,
        data_root=args.data_root,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        num_train_epochs=args.num_train_epochs,
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
