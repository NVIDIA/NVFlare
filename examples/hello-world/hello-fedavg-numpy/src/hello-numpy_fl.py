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

import copy

import numpy as np

import nvflare.client as flare


def train(input_arr):
    output_arr = copy.deepcopy(input_arr)
    # mock training with plus 1
    return output_arr + 1


def evaluate(input_arr):
    # mock evaluation metrics
    return np.mean(input_arr)


def main():
    flare.init()

    sys_info = flare.system_info()
    print(f"system info is: {sys_info}", flush=True)

    while flare.is_running():
        input_model = flare.receive()
        print(f"current_round={input_model.current_round}")
        print(f"received weights: {input_model.params}")

        sys_info = flare.system_info()
        print(f"system info is: {sys_info}")

        if input_model.params == {}:
            params = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        else:
            params = np.array(input_model.params["numpy_key"], dtype=np.float32)

        # training
        new_params = train(params)

        # evaluation
        metrics = evaluate(params)

        sys_info = flare.system_info()
        print(f"system info is: {sys_info}", flush=True)
        print(f"finished round: {input_model.current_round}", flush=True)

        print(f"sending weights: {new_params}")

        output_model = flare.FLModel(
            params={"numpy_key": new_params},
            params_type="FULL",
            metrics={"accuracy": metrics},
            current_round=input_model.current_round,
        )

        flare.send(output_model)


if __name__ == "__main__":
    main()
