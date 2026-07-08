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

import numpy as np

import nvflare.client as flare
from nvflare.app_common.np.constants import NPConstants


def main() -> None:
    flare.init()
    site_name = flare.system_info()["site_name"]

    while flare.is_running():
        input_model = flare.receive()
        input_weights = input_model.params[NPConstants.NUMPY_KEY]

        # Stand in for local training with a deterministic update.
        output_weights = input_weights + 1
        weight_mean = float(np.mean(output_weights))
        print(
            f"site={site_name} round={input_model.current_round} weight_mean={weight_mean:.4f}",
            flush=True,
        )

        flare.send(
            flare.FLModel(
                params={NPConstants.NUMPY_KEY: output_weights},
                params_type=flare.ParamsType.FULL,
                metrics={"weight_mean": weight_mean},
                current_round=input_model.current_round,
            )
        )


if __name__ == "__main__":
    main()
