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

from typing import Union

import numpy as np

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable


class GaussianPrivacy(DXOFilter):
    def __init__(self, sigma0=0.1, max_percentile=95, data_kinds: [str] = None):
        """Add Gaussian noise to shared model updates

        Args:
            sigma0: must be in >= 0, fraction of max value to compute noise
            max_percentile: must be in 0..100, only update nonzero abs diff greater than percentile
            data_kinds: kinds of DXO data to filter. If None,
                `[DataKind.WEIGHT_DIFF, DataKind.WEIGHTS]` is used.
        """
        if not data_kinds:
            data_kinds = [DataKind.WEIGHT_DIFF, DataKind.WEIGHTS]

        super().__init__(
            supported_data_kinds=[DataKind.WEIGHTS, DataKind.WEIGHT_DIFF],
            data_kinds_to_filter=data_kinds,
        )

        if not np.isscalar(sigma0):
            raise ValueError(f"Expected a positive scalar for `sigma0` but received type {type(sigma0)}")
        if sigma0 < 0.0:
            raise ValueError(f"Expected a positive float for `sigma0` but received {sigma0}.")

        if not np.isscalar(max_percentile):
            raise ValueError(
                f"Expected a positive scalar for `max_percentile` but received type {type(max_percentile)}"
            )
        if max_percentile < 0.0 or max_percentile > 100.0:
            raise ValueError(f"Expected a float for `sigma0` between 0 and 100 but received {max_percentile}.")

        self.sigma0 = sigma0
        self.max_percentile = max_percentile

    def process_dxo(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext) -> Union[None, DXO]:
        """Add Gaussian noise to data in dxo.

        Args:
            dxo: information from client
            shareable: that the dxo belongs to
            fl_ctx: context provided by workflow

        Returns: filtered result.
        """
        if self.sigma0 > 0.0:
            weights = dxo.data

            # abs delta
            all_abs_values = np.concatenate([np.abs(weights[name].ravel()) for name in weights])
            all_abs_nonzero = all_abs_values[all_abs_values > 0.0]
            max_value = np.percentile(a=all_abs_nonzero, q=self.max_percentile, overwrite_input=False)

            noise_sigma = self.sigma0 * max_value

            n_vars = len(weights)
            for var_name in weights:
                weights[var_name] = weights[var_name] + np.random.normal(0.0, noise_sigma, np.shape(weights[var_name]))
            self.log_info(
                fl_ctx,
                f"Added Gaussian noise to {n_vars} vars with sigma"
                f" {noise_sigma}, "
                f"sigma fraction: {self.sigma0}, "
                f"{self.max_percentile:.4f}th percentile of nonzero values: {max_value:.4f}",
            )

            dxo.data = weights
        else:
            self.log_warning(fl_ctx, "Sigma fraction is zero. No noise is being applied...")

        return dxo
