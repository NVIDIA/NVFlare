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

import torch

import nvflare.client as flare


def main():
    flare.init()
    input_model = flare.receive()
    if input_model is None or input_model.params is None:
        raise RuntimeError("missing input model parameters")

    updated_params = {}
    for name, value in input_model.params.items():
        value = torch.as_tensor(value, dtype=torch.float32)
        updated_params[name] = value + torch.ones_like(value)
    result = flare.FLModel(
        params=updated_params,
        metrics={"accuracy": 1.0},
        meta={"NUM_STEPS_CURRENT_ROUND": 1},
    )
    flare.send(result)


if __name__ == "__main__":
    main()
