# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.dxo_filter import DXO, DXOFilter
from nvflare.apis.filter import ContentBlockedException
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable


class DXOBlocker(DXOFilter):
    def __init__(self, data_kinds: List[str], allow_data_kinds: bool = False):
        """Block certain kinds of DXO objects.

        Args:
            allow_data_kinds: allow or block configured data kinds. If True, block everything not in
            the list; If False, block everything in the configured list.
            data_kinds: kinds of DXO object to block
        """
        super().__init__(supported_data_kinds=[], data_kinds_to_filter=[])  # support all kinds
        if not data_kinds:
            raise ValueError("data_kinds must be non-empty")
        if not isinstance(data_kinds, list):
            raise ValueError(f"data_kinds must be a list but got {type(data_kinds)}")
        if not all(isinstance(e, str) for e in data_kinds):
            raise ValueError("data_kinds must be a list of str but contains invalid element")
        self.configured_data_kinds = data_kinds
        self.allow_data_kinds = allow_data_kinds

    def process_dxo(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext) -> Union[None, DXO]:
        """
        Args:
            dxo (DXO): DXO to be filtered.
            shareable: that the dxo belongs to
            fl_ctx (FLContext): only used for logging.

        Returns: filtered dxo
        """
        if self.allow_data_kinds:
            if dxo.data_kind in self.configured_data_kinds:
                return None
            else:
                raise ContentBlockedException(f"DXO kind {dxo.data_kind} is blocked")
        else:
            if dxo.data_kind not in self.configured_data_kinds:
                return None
            else:
                raise ContentBlockedException(f"DXO kind {dxo.data_kind} is blocked")
