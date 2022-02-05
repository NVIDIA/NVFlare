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

from nvflare.apis.dxo import DataKind, MetaKey, from_shareable
from nvflare.apis.filter import Filter
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable


class ConvertWeights(Filter):

    WEIGHTS_TO_DIFF = "weights_to_diff"
    DIFF_TO_WEIGHTS = "diff_to_weights"

    def __init__(self, direction: str):
        """Convert WEIGHTS to WEIGHT_DIFF or vice versa.

        Args:
            direction (str): control conversion direction.  Either weights_to_diff or diff_to_weights.

        Raises:
            ValueError: when the direction string is neither weights_to_diff nor diff_to_weights
        """
        Filter.__init__(self)
        if direction not in (self.WEIGHTS_TO_DIFF, self.DIFF_TO_WEIGHTS):
            raise ValueError(
                "invalid convert direction {}: must be in {}".format(
                    direction, (self.WEIGHTS_TO_DIFF, self.DIFF_TO_WEIGHTS)
                )
            )

        self.direction = direction

    def _get_base_weights(self, fl_ctx: FLContext):
        task_data = fl_ctx.get_prop(FLContextKey.TASK_DATA, None)
        if not isinstance(task_data, Shareable):
            self.log_error(fl_ctx, "invalid task data: expect Shareable but got {}".format(type(task_data)))
            return None

        try:
            dxo = from_shareable(task_data)
        except ValueError:
            self.log_error(fl_ctx, "invalid task data: no DXO")
            return None

        if dxo.data_kind != DataKind.WEIGHTS:
            self.log_info(fl_ctx, "ignored task: expect data to be WEIGHTS but got {}".format(dxo.data_kind))
            return None

        processed_algo = dxo.get_meta_prop(MetaKey.PROCESSED_ALGORITHM, None)
        if processed_algo:
            self.log_info(fl_ctx, "ignored task since its processed by {}".format(processed_algo))
            return None

        return dxo.data

    def process(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        """Called by runners to perform weight conversion.

        When the return code of shareable is not ReturnCode.OK, this
        function will not perform any process and returns the shareable back.

        Args:
            shareable (Shareable): shareable must conform to DXO format.
            fl_ctx (FLContext): this context must include TASK_DATA, which is another shareable containing base weights.
              If not, the input shareable will be returned.

        Returns:
            Shareable: a shareable with converted weights
        """
        rc = shareable.get_return_code()
        if rc != ReturnCode.OK:
            # don't process if RC not OK
            return shareable

        base_weights = self._get_base_weights(fl_ctx)
        if not base_weights:
            return shareable

        try:
            dxo = from_shareable(shareable)
        except ValueError:
            self.log_error(fl_ctx, "invalid task result: no DXO")
            return shareable

        processed_algo = dxo.get_meta_prop(MetaKey.PROCESSED_ALGORITHM, None)
        if processed_algo:
            self.log_info(fl_ctx, "cannot process task result since its processed by {}".format(processed_algo))
            return shareable

        if self.direction == self.WEIGHTS_TO_DIFF:
            if dxo.data_kind != DataKind.WEIGHTS:
                self.log_warning(fl_ctx, "cannot process task result: expect WEIGHTS but got {}".format(dxo.data_kind))
                return shareable

            new_weights = dxo.data
            for k, _ in new_weights.items():
                if k in base_weights:
                    new_weights[k] -= base_weights[k]
            dxo.data_kind = DataKind.WEIGHT_DIFF
        else:
            # diff to weights
            if dxo.data_kind != DataKind.WEIGHT_DIFF:
                self.log_warning(
                    fl_ctx, "cannot process task result: expect WEIGHT_DIFF but got {}".format(dxo.data_kind)
                )
                return shareable

            new_weights = dxo.data
            for k, _ in new_weights.items():
                if k in base_weights:
                    new_weights[k] += base_weights[k]
            dxo.data_kind = DataKind.WEIGHTS

        return dxo.update_shareable(shareable)
