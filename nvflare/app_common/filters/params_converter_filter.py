# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Client-edge params-conversion filter for the Client API execution modes.

Design: docs/design/client_api_execution_modes.md ("Configuration Surface" — "conversion
moves out of the executor into send/receive filters at the client edge"). This DXO filter wraps
a ParamsConverter (e.g. numpy<->torch) so the framework conversion the legacy executors did
inside the Client API happens as a client-side filter instead:

- as the LAST task-data filter, it converts the aggregation representation (numpy) the server
  sends into the framework-native representation (e.g. torch.Tensor) the training script expects,
  so flare.receive() still hands the script native tensors;
- as the FIRST task-result filter, it converts back to numpy so the server aggregates numpy and
  privacy/DP/HE filters (which run before it on results, after it on task data) still see numpy.

Because it is a client filter, it runs regardless of execution mode (in_process/external_process);
the trainer-side Client API stays pass-through.
"""

from typing import Union

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.params_converter import ParamsConverter


class ParamsConverterFilter(DXOFilter):
    def __init__(self, converter: ParamsConverter):
        """Applies a ParamsConverter to a params DXO at the client edge.

        Args:
            converter: the ParamsConverter to apply (e.g. NumpyToPTParamsConverter for the
                task-data direction, PTToNumpyParamsConverter for the task-result direction).
        """
        super().__init__(
            supported_data_kinds=[DataKind.WEIGHTS, DataKind.WEIGHT_DIFF],
            data_kinds_to_filter=[DataKind.WEIGHTS, DataKind.WEIGHT_DIFF],
        )
        self.converter = converter

    def process_dxo(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext) -> Union[None, DXO]:
        if dxo.data_kind not in (DataKind.WEIGHTS, DataKind.WEIGHT_DIFF):
            return None
        dxo.data = self.converter.convert(dxo.data, fl_ctx)
        return dxo
