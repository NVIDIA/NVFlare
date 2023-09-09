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


def get_flat_weights(network):
    """Gets flat network weights.

    network get_weights() will give a dict of list of arrays.
    for NVFlare server side to work, it needs a dict of arrays.
    So we flatten this list using different key.
    For example:
      If the original network get_weights return: {"layer0": [array1, array2]}
      We will flat it to: {"layer0_nvf_0": array1, "layer0_nvf_1": array2}
    """
    result = {}
    for layer in network.layers:
        weights = layer.get_weights()
        if len(weights) != 0:
            for i, item in enumerate(weights):
                result[f"{layer.name}{SPECIAL_KEY}{i}"] = item
    return result


def load_flat_weights(network, data):
    """Loads the flat weights.

    For example:
      If the flat weight is: {"layer0_nvf_0": array1, "layer0_nvf_1": array2}
      We will convert it back to: {"layer0": [array1, array2]} and load it back
    """
    result = {}
    for k, v in data.items():
        if SPECIAL_KEY in k:
            layer_name, _ = k.split(SPECIAL_KEY)
            if layer_name not in result:
                result[layer_name] = []
            result[layer_name].append(v)
    for k in result:
        layer = network.get_layer(k)
        layer.set_weights(result[k])
