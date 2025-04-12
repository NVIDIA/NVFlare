# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Union

from nvflare.apis.dxo_filter import DXO, DXOFilter
from nvflare.apis.fl_constant import ParamFormat
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.fuel.utils.import_utils import optional_import

# Optional imports
NumpyToPTParamsConverter, has_pt_np2pt = optional_import(
    "nvflare.app_opt.pt.numpy_params_converter", name="NumpyToPTParamsConverter"
)
PTToNumpyParamsConverter, has_pt_pt2np = optional_import(
    "nvflare.app_opt.pt.numpy_params_converter", name="PTToNumpyParamsConverter"
)

KerasModelToNumpyParamsConverter, has_tf_keras2np = optional_import(
    "nvflare.app_opt.tf.params_converter", name="KerasModelToNumpyParamsConverter"
)
NumpyToKerasModelParamsConverter, has_tf_np2keras = optional_import(
    "nvflare.app_opt.tf.params_converter", name="NumpyToKerasModelParamsConverter"
)

# Conditionally build exchange format map
AUTO_REGISTERED_EXCHANGE_FORMAT_COMBINATIONS = {}

if has_pt_np2pt:
    AUTO_REGISTERED_EXCHANGE_FORMAT_COMBINATIONS[(ParamFormat.NUMPY, ParamFormat.PYTORCH)] = NumpyToPTParamsConverter()

if has_pt_pt2np:
    AUTO_REGISTERED_EXCHANGE_FORMAT_COMBINATIONS[(ParamFormat.PYTORCH, ParamFormat.NUMPY)] = PTToNumpyParamsConverter()

if has_tf_keras2np:
    AUTO_REGISTERED_EXCHANGE_FORMAT_COMBINATIONS[(ParamFormat.KERAS_LAYER_WEIGHTS, ParamFormat.NUMPY)] = (
        KerasModelToNumpyParamsConverter()
    )

if has_tf_np2keras:
    AUTO_REGISTERED_EXCHANGE_FORMAT_COMBINATIONS[(ParamFormat.NUMPY, ParamFormat.KERAS_LAYER_WEIGHTS)] = (
        NumpyToKerasModelParamsConverter()
    )


class ParamsConverterFilter(DXOFilter):
    def __init__(self, source: str, target: str):
        """Call ParamsConverter.

        Args:
            source (str): Source ParamFormat.
            target (str): Target ParamFormat.

        """
        # TODO: any data kinds or supported types?
        super().__init__(supported_data_kinds=None, data_kinds_to_filter=None)
        self.source = source
        self.target = target
        combination = (source, target)
        if combination not in AUTO_REGISTERED_EXCHANGE_FORMAT_COMBINATIONS:
            raise ValueError(f"({source=},{target=}) does not have built in converter.")
        self._params_converter = AUTO_REGISTERED_EXCHANGE_FORMAT_COMBINATIONS[combination]

    def process_dxo(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext) -> Union[None, DXO]:
        """Convert the dxo.data using the specified converter in the order that is specified.

        Args:
            dxo (DXO): DXO to be filtered.
            shareable: that the dxo belongs to
            fl_ctx (FLContext): only used for logging.

        Returns:
            Converted dxo.
        """
        dxo.data = self._params_converter.convert(dxo.data, fl_ctx)
        return dxo
