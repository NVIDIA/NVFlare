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

SPECIAL_KEY = "_nvf_"


def flat_layer_weights_dict(data: dict):
    """Flattens layer weights dict."""
    result = {}
    for layer_name, weights in data.items():
        if len(weights) != 0:
            # If the original layer get_weights return: {"layer0": [array1, array2]}
            # We will convert it to: {"layer0_nvf_0": array1, "layer0_nvf_1": array2}
            for i, item in enumerate(weights):
                result[f"{layer_name}{SPECIAL_KEY}{i}"] = item
    return result


def unflat_layer_weights_dict(data: dict):
    """Unflattens layer weights dict."""
    result = {}
    for k, v in data.items():
        if SPECIAL_KEY in k:
            # If the weight is: {"layer0_nvf_0": array1, "layer0_nvf_1": array2}
            # We will convert it back to: {"layer0": [array1, array2]} and load it back
            layer_name, _ = k.split(SPECIAL_KEY)
            if layer_name not in result:
                result[layer_name] = []
            result[layer_name].append(v)
    return result
