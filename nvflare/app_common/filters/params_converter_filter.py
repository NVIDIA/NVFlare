# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.params_converter import ParamsConverter


def _get_params_converter(params_converter_ids: List[str], fl_ctx: FLContext) -> List[ParamsConverter]:
    filters = []
    for component_id in params_converter_ids:
        c = fl_ctx.get_engine().get_component(component_id)
        # disabled component return None
        if c:
            if not isinstance(c, ParamsConverter):
                msg = f"component identified by {component_id} is type {type(c)} not type of ParamsConverter"
                raise ValueError(msg)
            filters.append(c)
    return filters


class ParamsConverterFilter(DXOFilter):
    def __init__(self, params_converter_ids: List[str]):
        """Call ParamsConverter.

        Args:
            params_converter_ids (List[str]): A list of params converter ids.

        """
        # TODO: any data kinds or supported types?
        super().__init__(supported_data_kinds=None, data_kinds_to_filter=None)
        self.params_converter_ids = params_converter_ids

    def process_dxo(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext) -> Union[None, DXO]:
        """Convert the dxo.data using the specified converter in the order that is specified.

        Args:
            dxo (DXO): DXO to be filtered.
            shareable: that the dxo belongs to
            fl_ctx (FLContext): only used for logging.

        Returns:
            Converted dxo.
        """
        self.log_info(fl_ctx, f"processing with {dxo.data_kind}")
        self.log_info(fl_ctx, "Starts ParamsConverterFilter")
        params_converters: List[ParamsConverter] = _get_params_converter(self.params_converter_ids, fl_ctx)

        identity_name = fl_ctx.get_identity_name()
        self.log_info(fl_ctx, f"apply ParamsConverterFilter for {identity_name}")

        for param_converter in params_converters:
            dxo.data = param_converter.convert(dxo.data, fl_ctx)

        self.log_info(fl_ctx, "end ParamsConverterFilter")
        return dxo
