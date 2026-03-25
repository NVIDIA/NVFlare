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
"""Recipe entrypoint for the hello-jax MNIST example."""

import argparse
import os

from nvflare.client.config import ExchangeFormat
from nvflare.fuel.utils.constants import FrameworkType
from nvflare.recipe import FedAvgRecipe, SimEnv

DEFAULT_INITIAL_CKPT = "/tmp/nvflare/data/hello-jax/initial_model.npy"
DEFAULT_DATA_DIR = "/tmp/nvflare/data/hello-jax/mnist"
REQUIRED_DATA_FILES = ("train_images.npy", "train_labels.npy", "test_images.npy", "test_labels.npy")


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--initial_ckpt", type=str, default=DEFAULT_INITIAL_CKPT)
    parser.add_argument("--train_script", type=str, default="client.py")
    parser.add_argument(
        "--launch_external_process",
        action="store_true",
        help="Run train_script in a separate subprocess instead of in-process.",
    )
    return parser.parse_args()


def _validate_inputs(initial_ckpt: str, data_dir: str) -> None:
    if not os.path.isfile(initial_ckpt):
        raise FileNotFoundError(
            f"Initial checkpoint not found: {initial_ckpt}. "
            f"Run `python prepare_model.py --output {initial_ckpt}` first."
        )

    missing_files = [name for name in REQUIRED_DATA_FILES if not os.path.isfile(os.path.join(data_dir, name))]
    if missing_files:
        missing_str = ", ".join(missing_files)
        raise FileNotFoundError(
            f"Prepared MNIST files missing in {data_dir}: {missing_str}. "
            f"Run `python prepare_data.py --data_dir {data_dir}` first."
        )


def main():
    args = define_parser()

    _validate_inputs(args.initial_ckpt, args.data_dir)
    train_args = (
        f"--epochs {args.epochs} "
        f"--batch_size {args.batch_size} "
        f"--learning_rate {args.learning_rate} "
        f"--momentum {args.momentum} "
        f"--num_partitions {args.n_clients} "
        f"--data_dir {args.data_dir}"
    )

    recipe = FedAvgRecipe(
        name="hello-jax",
        min_clients=args.n_clients,
        num_rounds=args.num_rounds,
        initial_ckpt=args.initial_ckpt,
        train_script=args.train_script,
        train_args=train_args,
        launch_external_process=args.launch_external_process,
        framework=FrameworkType.NUMPY,
        server_expected_format=ExchangeFormat.NUMPY,
    )

    env = SimEnv(num_clients=args.n_clients)
    run = recipe.execute(env)
    print()
    print("Job Status is:", run.get_status())
    print("Result can be found in :", run.get_result())
    print()


if __name__ == "__main__":
    main()
