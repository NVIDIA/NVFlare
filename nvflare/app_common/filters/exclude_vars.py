# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.dxo import from_shareable
from nvflare.apis.filter import Filter
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable


class ExcludeVars(Filter):
    def __init__(self, exclude_vars: Union[List[str], str, None] = None):
        """Exclude/Remove variables from Shareable.

        Args:
            exclude_vars (Union[List[str], str, None] , optional): variables/layer names to be excluded.

        Notes:
            Based on different types of exclude_vars, this filter has different behavior:
                if a list of variable/layer names, only specified variables will be excluded.
                if a string, it will be converted into a regular expression, only matched variables will be excluded.
                if not provided or other formats the Shareable remains unchanged.
        """
        super().__init__()
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

    def process(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        """Called by upper layer to remove variables in weights/weight_diff dictionary.

        When the return code of shareable is not ReturnCode.OK, this
        function will not perform any process and returns the shareable back.

        Args:
            shareable (Shareable): shareable must conform to DXO format.
            fl_ctx (FLContext): only used for logging.

        Returns:
            Shareable: a shareable with excluded weights
        """
        if self.skip:
            return shareable

        rc = shareable.get_return_code()
        if rc != ReturnCode.OK:
            # don't process if RC not OK
            return shareable

        try:
            dxo = from_shareable(shareable)
        except:
            self.log_exception(fl_ctx, "shareable data is not a valid DXO")
            return shareable

        if dxo.data is None:
            self.log_debug(fl_ctx, "no data to filter")
            return shareable

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
        return dxo.update_shareable(shareable)
