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


def load_ckpt(model: torch.nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, weights_only=True)
    state_dict = ckpt["model"]
    model.load_state_dict(state_dict)
    return model


def _client_ckpt(ws: Path, site: str) -> str:
    base = ws / site / "models"
    for name in ("best_model.pt", "last.pt"):
        p = base / name
        if p.is_file():
            return str(p)
    return str(base / "best_model.pt")


def run_validation(args):
    ws = Path(args.workspace)
    if not ws.exists():
        raise ValueError(f"Workspace path {ws} does not exist.")

    server = "app_server"
    sites = args.clients
    clients = [f"app_{s}" for s in sites]
    client_root = {f"app_{s}": ws / s / "simulate_job" / f"app_{s}" for s in sites}

    server_dir = ws / "server" / "simulate_job" / "app_server"
    for name in ("best_FL_global_model.pt", "FL_global_model.pt"):
        p = server_dir / name
        if p.is_file():
            server_ckpt = str(p)
            break
    else:
        server_ckpt = str(server_dir / "best_FL_global_model.pt")

    checkpoints = {server: server_ckpt}
    for c in clients:
        checkpoints[c] = _client_ckpt(ws, c[4:])

    config = {
        c: {
            "data": str(client_root[c] / "config" / "config_data.json"),
            "task": str(client_root[c] / "config" / "config_task.json"),
        }
        for c in clients
    }

    metrics = {app: {} for app in [server] + clients}

    for client in clients:
        with open(config[client]["data"]) as f:
            data_config = json.load(f)
        with open(config[client]["task"]) as f:
            task_config = json.load(f)

        print(f"Loading test cases from {client}'s dataset.")
        dm = DataManager(str(client_root[client]), data_config)
        dm.setup("test")

        model = get_model(task_config["model"])
        validator = Validator(task_config)

        model = load_ckpt(model, checkpoints[server])
        model = model.eval().cuda()
        print("Start validation using global best model.")
        raw_metrics = validator.run(model, dm.get_data_loader("test"))
        raw_metrics.pop("val_meandice")
        metrics[server].update(raw_metrics)

        for c in clients:
            model = load_ckpt(model, checkpoints[c])
            model = model.eval().cuda()
            print(f"Start validation using {c}'s best model.")
            raw_metrics = validator.run(model, dm.get_data_loader("test"))
            raw_metrics.pop("val_meandice")
            metrics[c].update(raw_metrics)

    for site in metrics:
        vals = [val for _, val in metrics[site].items()]
        metrics[site]["val_meandice"] = float(np.mean(vals)) if vals else None

    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--workspace", "-w", type=str, help="Workspace path.")
    parser.add_argument("--output", "-o", type=str, help="Output result JSON.")
    parser.add_argument(
        "--clients",
        "-c",
        nargs="+",
        required=True,
        metavar="SITE",
        help="Site directory names under workspace (e.g. spleen kidney liver pancreas). "
        "Checkpoints: workspace/<SITE>/models/best_model.pt or last.pt. "
        "Configs: workspace/<SITE>/simulate_job/app_<SITE>/config/.",
    )
    args = parser.parse_args()

    run_validation(args)
