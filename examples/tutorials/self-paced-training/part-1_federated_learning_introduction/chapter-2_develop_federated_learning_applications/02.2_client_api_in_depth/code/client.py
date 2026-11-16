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

import numpy as np
from model import mse, train
from prepare_data import make_data

import nvflare.client as flare
from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.np.constants import NPConstants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_steps", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    args = parser.parse_args()

    flare.init()
    try:
        site = flare.get_site_name()
        train_x, train_y, valid_x, valid_y = make_data(site)
        while flare.is_running():
            incoming = flare.receive()
            if incoming is None:
                break
            if not flare.is_train():
                raise RuntimeError(f"This example handles train tasks, received {flare.get_task_name()}")
            global_weights = incoming.params[NPConstants.NUMPY_KEY]
            # These metrics describe the received global model, before local training.
            global_mse = mse(global_weights, valid_x, valid_y)
            local_weights = train(global_weights, train_x, train_y, args.local_steps, args.learning_rate)
            local_mse = mse(local_weights, valid_x, valid_y)
            print(
                f"{site} | task={flare.get_task_name()} | round={incoming.current_round + 1}/{incoming.total_rounds} "
                f"| received={np.round(global_weights, 3)} | global_mse={global_mse:.4f} "
                f"| local_mse={local_mse:.4f} | steps={args.local_steps}",
                flush=True,
            )
            outgoing = flare.FLModel(
                params={NPConstants.NUMPY_KEY: local_weights},
                params_type=flare.ParamsType.FULL,
                metrics={"mse": global_mse},
                current_round=incoming.current_round,
                meta={FLMetaKey.NUM_STEPS_CURRENT_ROUND: args.local_steps},
            )
            flare.send(outgoing)
            # send() may release incoming.params and outgoing.params; retain copies beforehand if needed.
    finally:
        flare.shutdown()


if __name__ == "__main__":
    main()
