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

import re

import numpy as np

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.filter import Filter
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable


class ExcludeVars(Filter):
    """
        Exclude/Remove variables from Sharable

    Args:
        exclude_vars: if not specified (None), all layers are being encrypted;
                      if list of variable/layer names, only specified variables are excluded;
                      if string containing regular expression (e.g. "conv"), only matched variables are being excluded.
    """

    def __init__(self, exclude_vars=None):
        super().__init__()
        self.exclude_vars = exclude_vars
        self.skip = False
        if self.exclude_vars is not None:
            if not (isinstance(self.exclude_vars, list) or isinstance(self.exclude_vars, str)):
                self.skip = True
                self.logger.debug("Need to provide a list of layer names or a string for regex matching")
                return

            if isinstance(self.exclude_vars, list):
                for var in self.exclude_vars:
                    if not isinstance(var, str):
                        self.skip = True
                        self.logger.debug("encrypt_layers needs to be a list of layer names to encrypt.")
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

    def process(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:

        self.log_debug(fl_ctx, "inside filter")
        if self.skip:
            return shareable

        try:
            dxo = from_shareable(shareable)
        except:
            self.log_exception(fl_ctx, "shareable data is not a valid DXO")
            return shareable

        assert isinstance(dxo, DXO)
        if dxo.data_kind not in (DataKind.WEIGHT_DIFF, DataKind.WEIGHTS):
            self.log_debug(fl_ctx, "I cannot handle {}".format(dxo.data_kind))
            return shareable

        if dxo.data is None:
            self.log_debug(fl_ctx, "no data to filter")
            return shareable

        weights = dxo.data

        # parse regex encrypt layers
        if isinstance(self.exclude_vars, re.Pattern):
            re_pattern = self.exclude_vars
            self.exclude_vars = []
            for var_name in weights.keys():
                if re_pattern.search(var_name):
                    self.exclude_vars.append(var_name)
            self.log_debug(fl_ctx, f"Regex found {self.exclude_vars} matching layers.")
            if len(self.exclude_vars) == 0:
                self.log_warning(fl_ctx, f"No matching layers found with regex {re_pattern}")

        # remove variables
        n_excluded = 0
        var_names = list(weights.keys())  # needs to recast to list to be used in for loop
        n_vars = len(var_names)
        for var_name in var_names:
            # self.logger.info(f"Checking {var_name}")
            if var_name in self.exclude_vars:
                self.log_debug(fl_ctx, f"Excluding {var_name}")
                weights[var_name] = np.zeros(weights[var_name].shape)
                n_excluded += 1
        self.log_debug(
            fl_ctx,
            f"Excluded {n_excluded} of {n_vars} variables. {len(weights.keys())} remaining.",
        )

        dxo.data = weights
        return dxo.update_shareable(shareable)
