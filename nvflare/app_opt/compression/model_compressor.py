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

from typing import Union

import numpy as np

from nvflare.apis.dxo import DXO, DataKind, MetaKey
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_opt.compression.constant import COMPRESSION_TYPE, DATA_TYPE


class ModelCompressor(DXOFilter):
    def __init__(
        self,
        source_data_type="float32",
        compression_type="float16",
    ):
        """Filter to compress Shareable object to reduce communication burden.

        Args:
            source_data_type: original data type of the model
            compression_type: method used for compression

        """

        # support weight and weight_diff data kinds
        data_kinds = [DataKind.WEIGHTS, DataKind.WEIGHT_DIFF]
        super().__init__(supported_data_kinds=data_kinds, data_kinds_to_filter=data_kinds)

        # assign data and compression types
        self.logger.info("Using model compressor.")
        # check if source data type is valid
        if source_data_type.upper() not in DATA_TYPE:
            raise ValueError(f"Invalid source data type: {source_data_type}")
        else:
            self.source_data_type = source_data_type
        # check if compression type is valid
        if compression_type.upper() not in COMPRESSION_TYPE:
            raise ValueError(f"Invalid compression type: {compression_type}")
        else:
            self.compression_type = compression_type
        # compression constants
        self.FP16_MIN = np.finfo(np.float32).min
        self.FP16_MAX = np.finfo(np.float32).max

    def compression(self, params: dict, fl_ctx: FLContext):
        n_params = len(params.keys())
        self.log_info(fl_ctx, f"Running compression {n_params} variables")
        n_bytes_before = 0
        n_bytes_after = 0
        for i, param_name in enumerate(params.keys()):
            values = params[param_name]
            n_bytes_before += values.nbytes
            if self.source_data_type == "float32":
                if self.compression_type == "float16":
                    # first clamp the values to the range of float16
                    values = np.clip(values, self.FP16_MIN, self.FP16_MAX)
                    # then convert to float16
                    values = values.astype(np.float16)
            n_bytes_after += values.nbytes
            params[param_name] = values

        self.log_info(
            fl_ctx,
            f"Compressed all {n_params} params"
            f" Before compression: {n_bytes_before} bytes"
            f" After compression: {n_bytes_after} bytes",
        )
        return params

    def process_dxo(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext) -> Union[None, DXO]:
        """Filter process apply to the Shareable object.

        Args:
            dxo: data to be processed
            shareable: that the dxo belongs to
            fl_ctx: FLContext

        Returns: DXO object with compressed weights

        """

        self.log_info(fl_ctx, "Running compression...")
        compressed_params = self.compression(params=dxo.data, fl_ctx=fl_ctx)
        # Compose new DXO with compressed data
        new_dxo = DXO(data_kind=dxo.data_kind, data=compressed_params, meta=dxo.meta)
        new_dxo.set_meta_prop(key=MetaKey.PROCESSED_ALGORITHM, value=self.compression_type)
        self.log_info(fl_ctx, f"Compressed from {self.source_data_type} with {self.compression_type}")

        return new_dxo
