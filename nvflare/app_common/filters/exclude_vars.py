# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import re
from typing import List, Union

from nvflare.apis.dxo import DataKind
from nvflare.apis.dxo_filter import DXO, DXOFilter
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable


class ExcludeVars(DXOFilter):
    def __init__(self, exclude_vars: Union[List[str], str, None] = None, data_kinds: List[str] = None):
        """Exclude/Remove variables from Shareable.

        Args:
            exclude_vars (Union[List[str], str, None] , optional): variables/layer names to be excluded.
            data_kinds: kinds of DXO object to filter

        Notes:
            Based on different types of exclude_vars, this filter has different behavior:
                if a list of variable/layer names, only specified variables will be excluded.
                if a string, it will be converted into a regular expression, only matched variables will be excluded.
                if not provided or other formats the Shareable remains unchanged.
        """
        if not data_kinds:
            data_kinds = [DataKind.WEIGHT_DIFF, DataKind.WEIGHTS]

        super().__init__(supported_data_kinds=[DataKind.WEIGHTS, DataKind.WEIGHT_DIFF], data_kinds_to_filter=data_kinds)
        self.exclude_vars = exclude_vars
        self.skip = False
        if self.exclude_vars is not None:
            if not (isinstance(self.exclude_vars, list) or isinstance(self.exclude_vars, str)):
                self.skip = True
                self.logger.debug(
                    "Need to provide a list of layer names or a string for regex matching, but got {}".format(
                        type(self.exclude_vars)
                    )
                )
                return

            if isinstance(self.exclude_vars, list):
                for var in self.exclude_vars:
                    if not isinstance(var, str):
                        self.skip = True
                        self.logger.debug(
                            "encrypt_layers needs to be a list of layer names to encrypt, but contains element of type {}".format(
                                type(var)
                            )
                        )
                        return
                self.logger.debug(f"Excluding {self.exclude_vars} from shareable")
            elif isinstance(self.exclude_vars, str):
                self.exclude_vars = re.compile(self.exclude_vars) if self.exclude_vars else None
                if self.exclude_vars is None:
                    self.skip = True
                self.logger.debug(f'Excluding all layers based on regex matches with "{self.exclude_vars}"')
        else:
            self.logger.debug("Not excluding anything")
            self.skip = True

    def process_dxo(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext) -> Union[None, DXO]:
        """Called by upper layer to remove variables in weights/weight_diff dictionary.

        When the return code of shareable is not ReturnCode.OK, this
        function will not perform any process and returns the shareable back.

        Args:
            dxo (DXO): DXO to be filtered.
            shareable: that the dxo belongs to
            fl_ctx (FLContext): only used for logging.

        Returns: filtered dxo
        """
        if self.skip:
            return None

        weights = dxo.data
        # remove variables
        n_excluded = 0
        var_names = list(weights.keys())  # make a copy of keys
        n_vars = len(var_names)

        for var_name in var_names:
            if (isinstance(self.exclude_vars, re.Pattern) and self.exclude_vars.search(var_name)) or (
                isinstance(self.exclude_vars, list) and var_name in self.exclude_vars
            ):
                self.log_debug(fl_ctx, f"Excluding {var_name}")
                weights.pop(var_name, None)
                n_excluded += 1

        if isinstance(self.exclude_vars, re.Pattern) and n_excluded == 0:
            self.log_warning(fl_ctx, f"No matching layers found with regex {self.exclude_vars}")

        self.log_debug(fl_ctx, f"Excluded {n_excluded} of {n_vars} variables. {len(weights.keys())} remaining.")

        dxo.data = weights
        return dxo
