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
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.streaming.download_service import Downloadable, DownloadService


class ObjectDownloader:
    """Defines a universal object downloader that can be used to download any Downloadable objects."""

    def __init__(
        self,
        cell: Cell,
        timeout: float,
        num_receivers: int,
        tx_id=None,
        transaction_done_cb=None,
        **cb_kwargs,
    ):
        """Constructor of ObjectDownloader.

        Args:
            cell: the communication cell.
            timeout: timeout of the transaction
            num_receivers: number of sites to download the objects to. 0 means unknown.
            tx_id: if specified, this will be used as the ID of the new transaction. If not specified, dynamically
                generates the transaction id.
            transaction_done_cb: the callback to be called when the transaction is done.
            **cb_kwargs: kwargs to be passed to transaction_done_cb.

        Notes: the CB signature is:

            transaction_done_cb(tx_id, status, objects: list, **cb_kwargs)

        where tx_id is the ID of the transaction; status is a value as defined in TransactionDoneStatus;
        objects is a list of base objects to be downloaded. Note that the base object is not the Downloadable object!
        For example, in case of file downloading, the Downloadable object is FileDownloadable, whereas the base object
        is the name of the file.

        Downloadable object is only needed to work with the Downloader, and not useful for apps.
        """
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
        """Add a Downloadable object to the downloader.

        Args:
            obj: the Downloadable object to be added.
            ref_id: if specified, use it as the generated ref. If not specified, dynamically generates a ref ID.

        Returns: the ref ID for the object.

        """
        rid = DownloadService.add_object(
            transaction_id=self.tx_id,
            obj=obj,
            ref_id=ref_id,
        )
        return rid

    def delete_transaction(self):
        """Delete the download transaction forcefully.
        You call this method only if you want to stop the downloading process prematurely.

        Returns: None.

        """
        DownloadService.delete_transaction(self.tx_id)
