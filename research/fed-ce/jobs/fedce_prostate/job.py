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

import argparse
import shlex
from pathlib import Path

import torch
from model import UNet

from nvflare.app_opt.pt.recipes import FedCERecipe
from nvflare.recipe import SimEnv, add_experiment_tracking

HERE = Path(__file__).resolve().parent
CLIENTS = [
    "client_I2CVB",
    "client_MSD",
    "client_NCI_ISBI_3T",
    "client_NCI_ISBI_Dx",
    "client_Promise12",
    "client_PROSTATEx",
]


def _build_train_args(args):
    return shlex.join(
        [
            "--data-root",
            str(Path(args.data_root).expanduser().resolve()),
            "--batch-size",
            str(args.batch_size),
            "--cache-rate",
            str(args.cache_rate),
            "--learning-rate",
            str(args.learning_rate),
            "--local-epochs",
            str(args.local_epochs),
            "--num-workers",
            str(args.num_workers),
            "--seed",
            str(args.seed),
        ]
    )


def build_recipe(args):
    torch.manual_seed(args.seed)
    recipe = FedCERecipe(
        name="fedce_prostate",
        model=UNet(in_channels=1, out_channels=1),
        min_clients=len(CLIENTS),
        num_rounds=args.num_rounds,
        train_script=str(HERE / "client.py"),
        train_args=_build_train_args(args),
        fedce_mode=args.fedce_mode,
        key_metric="dice",
    )
    model_path = str(HERE / "model.py")
    recipe.add_server_file(model_path)
    recipe.add_client_file(model_path)
    add_experiment_tracking(recipe, "tensorboard", tracking_config={"tb_folder": "tb_events"})
    return recipe


def main(args):
    recipe = build_recipe(args)
    environment = SimEnv(
        clients=CLIENTS,
        num_threads=args.num_threads,
        gpu_config=args.gpu_config,
        workspace_root=str(Path(args.workspace_root).expanduser().resolve()),
    )
    run = recipe.execute(environment)
    print(f"Job status: {run.get_status()}")
    print(f"Results: {run.get_result()}")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the FedCE prostate segmentation research example.")
    parser.add_argument("--data-root", required=True, help="Directory containing dataset_2D and datalist_2D.")
    parser.add_argument("--workspace-root", default="/tmp/nvflare/fedce_prostate")
    parser.add_argument("--num-rounds", type=int, default=100)
    parser.add_argument("--num-threads", type=int, default=len(CLIENTS))
    parser.add_argument("--gpu-config", default="0,1,0,1,0,1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cache-rate", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fedce-mode", choices=("plus", "times"), default="plus")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(_parse_args())
