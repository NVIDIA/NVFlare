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

from typing import Any, Dict, Union

import tenseal as ts

from nvflare.apis.dxo import DataKind, MetaKey, from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
from nvflare.app_opt.he import decomposers


class HEInTimeAccumulateWeightedAggregator(InTimeAccumulateWeightedAggregator):
    def __init__(
        self,
        exclude_vars: Union[str, Dict[str, str], None] = None,
        aggregation_weights: Union[Dict[str, Any], Dict[str, Dict[str, Any]], None] = None,
        expected_data_kind: Union[DataKind, Dict[str, DataKind]] = DataKind.WEIGHT_DIFF,
        weigh_by_local_iter=False,
    ):
        """In time aggregator for `Shareables` encrypted using homomorphic encryption (HE) with TenSEAL https://github.com/OpenMined/TenSEAL.
           Needed to register FOBS decomposer for HE (e.g. for CKKSVector).

        Args:
            exclude_vars ([list], optional): variable names that should be excluded from aggregation (use regular expression). Defaults to None.
            aggregation_weights ([dict], optional): dictionary of client aggregation. Defaults to None.
            weigh_by_local_iter (bool, optional): If true, multiply client weights on first in encryption space
                                 (default: `False` which is recommended for HE, first multiply happens in `HEModelEncryptor`)].
            expected_data_kind (str, optional): the data_kind this aggregator can process. Defaults to "WEIGHT_DIFF".
        """
        super().__init__(
            exclude_vars=exclude_vars,
            aggregation_weights=aggregation_weights,
            expected_data_kind=expected_data_kind,
            weigh_by_local_iter=weigh_by_local_iter,
        )

        decomposers.register()

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        shareable = super().aggregate(fl_ctx=fl_ctx)

        # get processed keys and add to dxo
        dxo = from_shareable(shareable)
        weights = dxo.data
        if not isinstance(weights, dict):
            raise ValueError(f"Expected weights to be of type dict but got type {type(weights)}")

        encrypted_layers = dict()
        for k, v in weights.items():
            if isinstance(v, ts.CKKSVector):
                encrypted_layers[k] = True
            else:
                encrypted_layers[k] = False
        dxo.set_meta_prop(MetaKey.PROCESSED_KEYS, encrypted_layers)

        return dxo.to_shareable()
