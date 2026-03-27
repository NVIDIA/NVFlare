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

"""Client-side training script for hello-numpy-robust."""

import argparse

import numpy as np

import nvflare.client as flare
from nvflare.app_common.np.constants import NPConstants
from nvflare.client.tracking import SummaryWriter


def train(input_numpy_array: np.ndarray) -> np.ndarray:
    """Mock training by adding 1 to each weight."""
    return input_numpy_array + 1


def evaluate(input_numpy_array: np.ndarray) -> dict:
    """Simple metric for visibility in simulation logs."""
    return {"weight_mean": float(np.mean(input_numpy_array))}


def maybe_poison_update(update: np.ndarray, client_name: str, poison_client_name: str, poison_scale: float) -> np.ndarray:
    """Inject a large outlier update for a selected client to simulate Byzantine behavior."""
    if poison_client_name and client_name == poison_client_name:
        return update * poison_scale
    return update


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update_type", choices=["full", "diff"], default="full")
    parser.add_argument("--poison_client_name", type=str, default="site-1")
    parser.add_argument("--poison_scale", type=float, default=1000.0)
    args = parser.parse_args()

    flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]

    print(f"Client {client_name} initialized")
    summary_writer = SummaryWriter()

    while flare.is_running():
        input_model = flare.receive()
        input_np_arr = input_model.params[NPConstants.NUMPY_KEY]
        print(f"Client {client_name}, current_round={input_model.current_round}, received={input_np_arr}")

        new_params = train(input_np_arr)
        new_params = maybe_poison_update(new_params, client_name, args.poison_client_name, args.poison_scale)

        metrics = evaluate(new_params)
        summary_writer.add_scalar(tag="weight_mean", scalar=metrics["weight_mean"], global_step=input_model.current_round)

        if args.update_type == "diff":
            params_to_send = new_params - input_np_arr
            params_type = flare.ParamsType.DIFF
        else:
            params_to_send = new_params
            params_type = flare.ParamsType.FULL

        print(f"Client {client_name} sending={params_to_send}")
        output_model = flare.FLModel(
            params={NPConstants.NUMPY_KEY: params_to_send},
            params_type=params_type,
            metrics=metrics,
            current_round=input_model.current_round,
        )
        flare.send(output_model)


if __name__ == "__main__":
    main()
