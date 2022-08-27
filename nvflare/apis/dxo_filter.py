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

from abc import ABC, abstractmethod
from typing import Union, List, Dict

from .filter import Filter
from .shareable import Shareable
from .fl_context import FLContext
from .fl_constant import ReturnCode
from .dxo import DXO, DataKind, from_shareable


class DXOFilter(Filter, ABC):
    """
    This is the base class for DXO-based filters
    """

    def __init__(self,
                 supported_data_kinds: Union[None, List[str]],
                 data_kinds_to_filter: Union[None, List[str]]):
        """

        Args:
            supported_data_kinds: kinds of DXO this filter supports. Empty means all kinds.
            data_kinds_to_filter: kinds of DXO data to filter. Empty means all kinds.
        """
        Filter.__init__(self)

        if supported_data_kinds and not isinstance(supported_data_kinds, list):
            raise ValueError(f"supported_data_kinds must be a list of str but got {type(supported_data_kinds)}")

        if data_kinds_to_filter and not isinstance(data_kinds_to_filter, list):
            raise ValueError(f"data_kinds_to_filter must be a list of str but got {type(data_kinds_to_filter)}")

        if supported_data_kinds and data_kinds_to_filter:
            if not all(dk in supported_data_kinds for dk in data_kinds_to_filter):
                raise ValueError(f"invalid data kinds: {data_kinds_to_filter}. Only support {data_kinds_to_filter}")

        if not data_kinds_to_filter:
            data_kinds_to_filter = supported_data_kinds

        self.data_kinds = data_kinds_to_filter

    def process(self, shareable: Shareable, fl_ctx: FLContext):
        rc = shareable.get_return_code()
        if rc != ReturnCode.OK:
            # don't process if RC not OK
            return shareable

        try:
            dxo = from_shareable(shareable)
        except:
            # not a DXO based shareable - pass
            return shareable

        if dxo.data is None:
            self.log_debug(fl_ctx, "DXO has no data to filter")
            return shareable

        start = [dxo]
        self._filter_dxos(start, shareable, fl_ctx)
        result_dxo = start[0]
        return result_dxo.update_shareable(shareable)

    @abstractmethod
    def process_dxo(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext) -> (bool, DXO):
        """Subclass must implement this method to filter the provided DXO

        Args:
            dxo: the DXO to be filtered
            shareable: the shareable that the dxo belongs to
            fl_ctx: the FL context

        Returns: a tuple of:
            whether the DXO is filtered;
            a DXO object that is the result of the filtering. It can be either the input DXO
        object or a new DXO object.

        """
        pass

    def _apply_filter(self, dxo: DXO, shareable, fl_ctx) -> DXO:
        if not dxo.data:
            self.log_debug(fl_ctx, "DXO has no data to filter")
            return dxo

        filtered, result = self.process_dxo(dxo, shareable, fl_ctx)
        if not filtered and not result:
            # in case the programmer forgot to return a result
            result = dxo

        if not isinstance(result, DXO):
            raise RuntimeError(
                f"Result from {self.__class__.__name__} is {type(result)} - should be DXO")
        if filtered:
            result.add_filter_history(self.__class__.__name__)
        return result

    def _filter_dxos(self, dxo_collection: Union[List[DXO], Dict[str, DXO]], shareable, fl_ctx):
        if isinstance(dxo_collection, list):
            for i in range(len(dxo_collection)):
                v = dxo_collection[i]
                if not isinstance(v, DXO):
                    continue
                if v.data_kind == DataKind.COLLECTION:
                    self._filter_dxos(v.data, shareable, fl_ctx)
                elif not self.data_kinds or v.data_kind in self.data_kinds:
                    dxo_collection[i] = self._apply_filter(v, shareable, fl_ctx)

        elif isinstance(dxo_collection, dict):
            for k, v in dxo_collection.items():
                assert isinstance(v, DXO)
                if v.data_kind == DataKind.COLLECTION:
                    self._filter_dxos(v.data, shareable, fl_ctx)
                elif not self.data_kinds or v.data_kind in self.data_kinds:
                    dxo_collection[k] = self._apply_filter(v, shareable, fl_ctx)
        else:
            raise ValueError(f"DXO COLLECTION must be a dict or list but got {type(dxo_collection)}")
