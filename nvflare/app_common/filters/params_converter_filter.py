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

from typing import Union

from nvflare.apis.dxo_filter import DXO, DXOFilter
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.params_converter import ParamsConverter


def _get_params_converter(params_converter_id: str, fl_ctx: FLContext) -> ParamsConverter:
    c = fl_ctx.get_engine().get_component(params_converter_id)

    if c:
        if not isinstance(c, ParamsConverter):
            msg = f"component identified by {params_converter_id} is type {type(c)} not type of ParamsConverter"
            raise ValueError(msg)
    return c


class ParamsConverterFilter(DXOFilter):
    def __init__(self, params_converter_id: str):
        """Call ParamsConverter.

        Args:
            params_converter_id (str): ID to a ParamsConverter.

        """
        # TODO: any data kinds or supported types?
        super().__init__(supported_data_kinds=None, data_kinds_to_filter=None)
        self._params_converter_id = params_converter_id
        self._params_converter = None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self._params_converter: ParamsConverter = _get_params_converter(self._params_converter_id, fl_ctx)
        super().handle_event(event_type, fl_ctx)

    def process_dxo(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext) -> Union[None, DXO]:
        """Convert the dxo.data using the specified converter in the order that is specified.

        Args:
            dxo (DXO): DXO to be filtered.
            shareable: that the dxo belongs to
            fl_ctx (FLContext): only used for logging.

        Returns:
            Converted dxo.
        """
        dxo.data = self._params_converter.convert(dxo.data, fl_ctx)
        return dxo
