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

"""External trainer for the Swarm tensor disk-offload integration job."""

import torch

import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType


def main():
    flare.init()
    site_name = flare.get_site_name()
    delta = 1.0 if site_name == "site-1" else 2.0

    while flare.is_running():
        input_model = flare.receive()
        trained_params = {
            name: tensor.detach().to(device="cpu", dtype=torch.float32) + delta
            for name, tensor in input_model.params.items()
        }
        flare.send(
            FLModel(
                params=trained_params,
                params_type=ParamsType.FULL,
                metrics={"delta": delta},
                meta={"NUM_STEPS_CURRENT_ROUND": 1},
            )
        )
        print(
            f"tensor offload trainer site={site_name} round={input_model.current_round} delta={delta}",
            flush=True,
        )


if __name__ == "__main__":
    main()
