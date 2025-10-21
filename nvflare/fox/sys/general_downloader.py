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
from nvflare.fuel.f3.streaming.obj_downloader import Downloadable, ObjDownloader

from .constants import DownloaderKey


class GeneralDownloader:

    def __init__(
        self,
        num_receivers: int,
        timeout: float,
        ctx: Context,
    ):
        backend = ctx.backend
        if not isinstance(backend, SysBackend):
            raise ValueError(f"backend must be SysBackend but got {type(backend)}")

        self.cell = backend.cell
        self.tx_id = ObjDownloader.new_transaction(
            cell=self.cell,
            timeout=timeout,
            num_receivers=num_receivers,
        )

    def add_object(self, obj: Downloadable):
        rid = ObjDownloader.add_object(
            transaction_id=self.tx_id,
            obj=obj,
        )
        return {DownloaderKey.SOURCE: self.cell.get_fqcn(), DownloaderKey.REF_ID: rid}
