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

from typing import Union

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable


class ConvertWeights(DXOFilter):

    WEIGHTS_TO_DIFF = "weights_to_diff"
    DIFF_TO_WEIGHTS = "diff_to_weights"

    def __init__(self, direction: str):
        """Convert WEIGHTS to WEIGHT_DIFF or vice versa.

        Args:
            direction (str): control conversion direction.  Either weights_to_diff or diff_to_weights.

        Raises:
            ValueError: when the direction string is neither weights_to_diff nor diff_to_weights
        """
        DXOFilter.__init__(
            self, supported_data_kinds=[DataKind.WEIGHT_DIFF, DataKind.WEIGHTS], data_kinds_to_filter=None
        )
        if direction not in (self.WEIGHTS_TO_DIFF, self.DIFF_TO_WEIGHTS):
            raise ValueError(
                f"invalid convert direction {direction}: must be in {(self.WEIGHTS_TO_DIFF, self.DIFF_TO_WEIGHTS)}"
            )

        self.direction = direction

    def _get_base_weights(self, fl_ctx: FLContext):
        task_data = fl_ctx.get_prop(FLContextKey.TASK_DATA, None)
        if not isinstance(task_data, Shareable):
            self.log_error(fl_ctx, f"invalid task data: expect Shareable but got {type(task_data)}")
            return None

        try:
            dxo = from_shareable(task_data)
        except ValueError:
            self.log_error(fl_ctx, "invalid task data: no DXO")
            return None

        if dxo.data_kind != DataKind.WEIGHTS:
            self.log_info(fl_ctx, f"ignored task: expect data to be WEIGHTS but got {dxo.data_kind}")
            return None

        processed_algo = dxo.get_meta_prop(MetaKey.PROCESSED_ALGORITHM, None)
        if processed_algo:
            self.log_info(fl_ctx, f"ignored task since its processed by {processed_algo}")
            return None

        return dxo.data

    def process_dxo(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext) -> Union[None, DXO]:
        """Called by runners to perform weight conversion.

        Args:
            dxo (DXO): dxo to be processed.
            shareable: the shareable that the dxo belongs to
            fl_ctx (FLContext): this context must include TASK_DATA, which is another shareable containing base weights.
              If not, the input shareable will be returned.

        Returns: filtered result
        """
        base_weights = self._get_base_weights(fl_ctx)
        if not base_weights:
            return None

        processed_algo = dxo.get_meta_prop(MetaKey.PROCESSED_ALGORITHM, None)
        if processed_algo:
            self.log_info(fl_ctx, f"cannot process task result since its processed by {processed_algo}")
            return None

        if self.direction == self.WEIGHTS_TO_DIFF:
            if dxo.data_kind != DataKind.WEIGHTS:
                self.log_warning(fl_ctx, f"cannot process task result: expect WEIGHTS but got {dxo.data_kind}")
                return None

            new_weights = dxo.data
            for k, _ in new_weights.items():
                if k in base_weights:
                    new_weights[k] -= base_weights[k]
            dxo.data_kind = DataKind.WEIGHT_DIFF
        else:
            # diff to weights
            if dxo.data_kind != DataKind.WEIGHT_DIFF:
                self.log_warning(fl_ctx, f"cannot process task result: expect WEIGHT_DIFF but got {dxo.data_kind}")
                return None

            new_weights = dxo.data
            for k, _ in new_weights.items():
                if k in base_weights:
                    new_weights[k] += base_weights[k]
            dxo.data_kind = DataKind.WEIGHTS

        return dxo
