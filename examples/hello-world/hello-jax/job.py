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
Recipe entrypoint for the hello-jax MNIST example.
"""

import argparse
import os
import subprocess
import sys
import tempfile

from nvflare.app_common.np.np_model_persistor import NPModelPersistor
from nvflare.client.config import ExchangeFormat
from nvflare.fuel.utils.constants import FrameworkType
from nvflare.recipe import FedAvgRecipe, SimEnv


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--data_dir", type=str, default="/tmp/nvflare/data/hello-jax/mnist")
    parser.add_argument("--train_script", type=str, default="client.py")
    parser.add_argument(
        "--launch_external_process",
        action="store_true",
        help="Run train_script in a separate subprocess instead of in-process.",
    )
    return parser.parse_args()


def prepare_initial_model_ckpt() -> str:
    init_dir = tempfile.mkdtemp(prefix="hello-jax-init-")
    ckpt_path = os.path.join(init_dir, "initial_model.npy")
    init_script = os.path.join(os.path.dirname(__file__), "prepare_model.py")
    subprocess.run([sys.executable, init_script, "--output", ckpt_path], check=True)
    return ckpt_path


def prepare_dataset(data_dir: str) -> None:
    data_script = os.path.join(os.path.dirname(__file__), "prepare_data.py")
    subprocess.run([sys.executable, data_script, "--data_dir", data_dir], check=True)


def main():
    args = define_parser()

    initial_ckpt = prepare_initial_model_ckpt()
    prepare_dataset(args.data_dir)
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
        model_persistor=NPModelPersistor(source_ckpt_file_full_name=initial_ckpt),
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
