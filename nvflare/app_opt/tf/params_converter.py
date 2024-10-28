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

from typing import Any

from nvflare.app_common.abstract.params_converter import ParamsConverter
from nvflare.app_opt.tf.utils import flat_layer_weights_dict, unflat_layer_weights_dict


class NumpyToKerasModelParamsConverter(ParamsConverter):
    """This class converts a flattened numpy dict back into a dict of {layer_name: layer_weights}.

    The result can be used by ```keras_model.get_layer(layer_name).set_weights(layer_weights)```
    """

    def convert(self, params: Any, fl_ctx) -> Any:
        """Unflattens layer weights dict."""
        return unflat_layer_weights_dict(params)


class KerasModelToNumpyParamsConverter(ParamsConverter):
    """This class converts the dict of {layer_name: layer_weights} to a flattened numpy dict.

    The layer_name is from a keras ```layer.name```
    The layer_weights is from ```layer.get_weights()```
    """

    def convert(self, params: Any, fl_ctx) -> Any:
        """Flattens layer weights dict."""
        return flat_layer_weights_dict(params)
