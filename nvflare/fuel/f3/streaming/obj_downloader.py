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
from nvflare.fox.api.ctx import Context
from nvflare.fox.sys.backend import SysBackend
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.streaming.download_service import Downloadable, DownloadService


class ObjectDownloader:

    def __init__(
        self,
        cell: Cell,
        timeout: float,
        num_receivers: int,
        tx_id=None,
        transaction_done_cb=None,
        **cb_kwargs,
    ):
        self.cell = cell
        self.tx_id = DownloadService.new_transaction(
            cell=self.cell,
            timeout=timeout,
            num_receivers=num_receivers,
            tx_id=tx_id,
            transaction_done_cb=transaction_done_cb,
            **cb_kwargs,
        )

    def add_object(self, obj: Downloadable, ref_id=None) -> str:
        rid = DownloadService.add_object(
            transaction_id=self.tx_id,
            obj=obj,
            ref_id=ref_id,
        )
        return rid

    def delete_transaction(self):
        DownloadService.delete_transaction(self.tx_id)
