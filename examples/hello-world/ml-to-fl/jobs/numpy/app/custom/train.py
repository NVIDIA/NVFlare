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

import nvflare.client as flare


def main():
    # initializes NVFlare interface
    flare.init()

    # get model from NVFlare
    input_weights, input_metadata = flare.receive()

    print(f"received weights is: {input_weights}")
    print(f"received metadata is: {input_metadata}")
    input_weights["numpy_key"] += 1

    flare.submit_model(model=input_weights, meta={"NUM_STEPS_CURRENT_ROUND": 0})


if __name__ == "__main__":
    main()
