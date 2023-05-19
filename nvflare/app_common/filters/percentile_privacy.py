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

from typing import List, Union

import numpy as np

from nvflare.apis.dxo import DXO, DataKind, MetaKey
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable


class PercentilePrivacy(DXOFilter):
    def __init__(self, percentile=10, gamma=0.01, data_kinds: List[str] = None):
        """Implementation of "largest percentile to share" privacy preserving policy.

        Shokri and Shmatikov, Privacy-preserving deep learning, CCS '15

        Args:
            percentile (int, optional): Only abs diff greater than this percentile is updated.
              Allowed range 0..100.  Defaults to 10.
            gamma (float, optional): The upper limit to truncate abs values of weight diff. Defaults to 0.01.  Any weight diff with abs<gamma will become 0.
            data_kinds: kinds of DXO to filter
        """
        if not data_kinds:
            data_kinds = [DataKind.WEIGHT_DIFF, DataKind.WEIGHTS]

        super().__init__(supported_data_kinds=[DataKind.WEIGHTS, DataKind.WEIGHT_DIFF], data_kinds_to_filter=data_kinds)

        # must be in 0..100, only update abs diff greater than percentile
        self.percentile = percentile
        # must be positive
        self.gamma = gamma  # truncate absolute value of delta W

    def process_dxo(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext) -> Union[None, DXO]:
        """Compute the percentile on the abs delta_W.

        Only share the params where absolute delta_W greater than
        the percentile value

        Args:
            dxo: information from client
            shareable: that the dxo belongs to
            fl_ctx: context provided by workflow

        Returns: filtered dxo
        """
        self.log_debug(fl_ctx, "inside filter")
        self.logger.debug("check gamma")
        if self.gamma <= 0:
            self.log_debug(fl_ctx, "no partial model: gamma: {}".format(self.gamma))
            return None
        if self.percentile < 0 or self.percentile > 100:
            self.log_debug(fl_ctx, "no partial model: percentile: {}".format(self.percentile))
            return None  # do nothing

        # invariant to local steps
        model_diff = dxo.data
        total_steps = dxo.get_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, 1)

        delta_w = {name: model_diff[name] / total_steps for name in model_diff}
        # abs delta
        all_abs_values = np.concatenate([np.abs(delta_w[name].ravel()) for name in delta_w])
        cutoff = np.percentile(a=all_abs_values, q=self.percentile, overwrite_input=False)
        self.log_info(
            fl_ctx,
            f"Max abs delta_w: {np.max(all_abs_values)}, Min abs delta_w: {np.min(all_abs_values)},"
            f"cutoff: {cutoff}, scale: {total_steps}.",
        )

        for name in delta_w:
            diff_w = delta_w[name]
            if np.ndim(diff_w) == 0:  # single scalar, no clipping
                delta_w[name] = diff_w * total_steps
                continue
            selector = (diff_w > -cutoff) & (diff_w < cutoff)
            diff_w[selector] = 0.0
            diff_w = np.clip(diff_w, a_min=-self.gamma, a_max=self.gamma)
            delta_w[name] = diff_w * total_steps

        dxo.data = delta_w
        return dxo
