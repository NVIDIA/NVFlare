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

from typing import List

from nvflare.apis.filter import ContentBlockedException
from nvflare.apis.dxo_filter import DXOFilter, DXO
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable


class DXOBlocker(DXOFilter):
    def __init__(self,
                 data_kinds: List[str] = None):
        """Block certain kinds of DXO objects.

        Args:
            data_kinds: kinds of DXO object to block
        """
        super().__init__(
            supported_data_kinds=[],    # support all kinds
            data_kinds_to_filter=data_kinds
        )

    def process_dxo(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext) -> (bool, DXO):
        """
        Args:
            dxo (DXO): DXO to be filtered.
            shareable: that the dxo belongs to
            fl_ctx (FLContext): only used for logging.

        Returns: a tuple of:
            whether filtering is applied;
            DXO object with excluded weights
        """
        raise ContentBlockedException(f"DXO kind {dxo.data_kind} is blocked")
