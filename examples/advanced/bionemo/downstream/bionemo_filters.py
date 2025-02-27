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

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable


class BioNeMoParamsFilter(DXOFilter):
    def __init__(self, precision="bf16-mixed"):
        """Filter to add a prefix to global state dict to avoid key mismatches between global and local state dictionaries.
        This is needed because of NeMo training framework adding module wrappers depending on the used training precision.

        Args:
            precision: training precision
        """

        # support weight and weight_diff data kinds
        data_kinds = [DataKind.WEIGHTS, DataKind.WEIGHT_DIFF]
        super().__init__(supported_data_kinds=data_kinds, data_kinds_to_filter=data_kinds)

        self._precision = precision
        if self._precision == "bf16-mixed":
            self._prefix = "module.module."
        else:
            self._prefix = "module."

    def process_dxo(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext) -> Union[None, DXO]:
        """Filter process apply to the Shareable object.

        Args:
            dxo: data to be processed
            shareable: that the dxo belongs to
            fl_ctx: FLContext

        Returns: DXO object with updated state dictionary

        """

        self.log_info(fl_ctx, f"Adding `{self._prefix}` prefix...")

        params = dxo.data
        new_params = {}
        for k, v in params.items():
            new_key = self._prefix + k
            new_params[new_key] = v

        dxo.data = new_params
        return dxo


class BioNeMoExcludeParamsFilter(DXOFilter):
    def __init__(self, exclude_vars="head"):
        """Filter to remove parameters from state dictionary that shouldn't be shared with other party.

        Args:
            exclude_vars: variables will be excluded if the string is part of a state dictionary key.
        """

        # support weight and weight_diff data kinds
        data_kinds = [DataKind.WEIGHTS, DataKind.WEIGHT_DIFF]
        super().__init__(supported_data_kinds=data_kinds, data_kinds_to_filter=data_kinds)

        self.exclude_vars = exclude_vars

    def process_dxo(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext) -> Union[None, DXO]:
        """Filter process apply to the Shareable object.

        Args:
            dxo: data to be processed
            shareable: that the dxo belongs to
            fl_ctx: FLContext

        Returns: DXO object with updated state dictionary

        """

        params = dxo.data
        new_params = {}
        for k, v in params.items():
            if self.exclude_vars not in k:
                new_params[k] = v

        if len(new_params) < len(params):
            self.log_info(fl_ctx, f"Excluded {len(params) - len(new_params)} parameters matching '{self.exclude_vars}'")
        else:
            raise ValueError(f"State dictionary did not match any exclude keys that matched '{self.exclude_vars}'")

        dxo.data = new_params
        return dxo
