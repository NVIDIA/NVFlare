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

"""Small NumPy client used by the standalone OpenShift end-to-end example."""

import argparse

import numpy as np

import nvflare.client as flare
from nvflare.app_common.np.constants import NPConstants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update_type", choices=["full", "diff"], default="full")
    args = parser.parse_args()

    flare.init()
    while flare.is_running():
        input_model = flare.receive()
        input_array = input_model.params[NPConstants.NUMPY_KEY]
        trained_array = input_array + 1
        if args.update_type == "diff":
            params = trained_array - input_array
            params_type = flare.ParamsType.DIFF
        else:
            params = trained_array
            params_type = flare.ParamsType.FULL
        flare.send(
            flare.FLModel(
                params={NPConstants.NUMPY_KEY: params},
                params_type=params_type,
                metrics={"weight_mean": float(np.mean(trained_array))},
                current_round=input_model.current_round,
            )
        )


if __name__ == "__main__":
    main()
