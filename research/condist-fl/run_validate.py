# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from src.data import DataManager
from src.utils.get_model import get_model
from src.validator import Validator


def load_ckpt(app: str, model: torch.nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path)
    state_dict = ckpt["model"]
    model.load_state_dict(state_dict)
    return model


def run_validation(args):
    ws = Path(args.workspace)
    if not ws.exists():
        raise ValueError(f"Workspace path {ws} does not exists.")
    prefix = ws / "simulate_job"

    server = "app_server"
    clients = [p.name for p in prefix.glob("app_*") if "server" not in p.name]

    # Collect all checkpoints to evaluate
    checkpoints = {"app_server": str(prefix / "app_server/best_FL_global_model.pt")}
    checkpoints.update({c: str(prefix / c / "models/best_model.pt") for c in clients})

    # Collect configs from clients
    config = {
        c: {"data": str(prefix / c / "config/config_data.json"), "task": str(prefix / c / "config/config_task.json")}
        for c in clients
    }

    metrics = {app: {} for app in [server] + clients}
    for client in clients:
        with open(config[client]["data"]) as f:
            data_config = json.load(f)

        with open(config[client]["task"]) as f:
            task_config = json.load(f)

        print(f"Loading test cases from {client}'s dataset.")

        dm = DataManager(str(prefix), data_config)
        dm.setup("test")

        # Create model & validator from task config
        model = get_model(task_config["model"])
        validator = Validator(task_config)

        # Run server validation
        model = load_ckpt("app_server", model, checkpoints["app_server"])
        model = model.eval().cuda()

        print("Start validation using global best model.")
        raw_metrics = validator.run(model, dm.get_data_loader("test"))
        raw_metrics.pop("val_meandice")
        metrics["app_server"].update(raw_metrics)

        # Run client validation
        for c in clients:
            model = load_ckpt(c, model, checkpoints[c])
            model = model.eval().cuda()

            print(f"Start validation using {c}'s best model.")
            raw_metrics = validator.run(model, dm.get_data_loader("test"))
            raw_metrics.pop("val_meandice")
            metrics[c].update(raw_metrics)

    # Calculate correct val_meandice
    for site in metrics:
        metrics[site]["val_meandice"] = np.mean([val for _, val in metrics[site].items()])

    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--workspace", "-w", type=str, help="Workspace path.")
    parser.add_argument("--output", "-o", type=str, help="Output result JSON.")
    args = parser.parse_args()

    run_validation(args)
