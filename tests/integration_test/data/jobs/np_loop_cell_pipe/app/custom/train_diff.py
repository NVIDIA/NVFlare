# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import nvflare.client as flare


def train(input_arr):
    output_arr = copy.deepcopy(input_arr)
    # mock training with plus 1
    return output_arr + 1


def evaluate(input_arr):
    # mock evaluation metrics
    return 100


def main():
    # initializes NVFlare interface
    flare.init()

    # get model from NVFlare
    input_model = flare.receive()
    print(f"received weights is: {input_model.params}")

    # get system information
    sys_info = flare.system_info()
    print(f"system info is: {sys_info}")

    input_numpy_array = input_model.params["numpy_key"]

    # training
    output_numpy_array = train(input_numpy_array)

    # evaluation
    metrics = evaluate(input_numpy_array)

    # calculate difference here
    diff = output_numpy_array - input_numpy_array

    # send back the model difference
    print(f"send back: {diff}")
    flare.send(flare.FLModel(params={"numpy_key": diff}, params_type="DIFF", metrics={"accuracy": metrics}))


if __name__ == "__main__":
    main()
