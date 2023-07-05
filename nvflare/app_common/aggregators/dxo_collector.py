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


from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator


class DXOCollector(Aggregator):
    def __init__(self):
        Aggregator.__init__(self)
        self.dxos = {}

    def reset(self, fl_ctx: FLContext):
        self.dxos = {}

    def accept(self, shareable: Shareable, fl_ctx: FLContext):
        try:
            dxo = from_shareable(shareable)
        except Exception:
            self.log_exception(fl_ctx, "shareable data is not a valid DXO")
            return False
        peer_ctx = fl_ctx.get_peer_context()
        client_name = peer_ctx.get_identity_name()
        if not client_name:
            self.log_error(fl_ctx, "no identity info in peer context!")
            return False
        self.dxos[client_name] = dxo
        return True

    def aggregate(self, fl_ctx: FLContext):
        collection_dxo = DXO(data_kind=DataKind.COLLECTION, data=self.dxos)
        return collection_dxo.to_shareable()
