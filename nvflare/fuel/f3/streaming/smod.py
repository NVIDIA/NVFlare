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
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.streaming.obj_downloader import ObjDownloader, Producer
from nvflare.fuel.utils.log_utils import get_obj_logger


class SelfManagedObject(ABC):

    def set_transaction(self, tx_id, ref_id):
        pass

    @abstractmethod
    def produce(self, state: dict, requester: str) -> Tuple[str, Any, dict]:
        """Produce a small object to be sent (on object sender side).

        Args:
            state: current state of downloading, received from the downloading site
            requester: the FQCN of the site that is downloading

        Returns: a tuple of (return code, a small object to be sent, new state to be sent).

        """
        pass

    def downloaded_to_one(self, to_site: str, status: str):
        """Called when an object is downloaded to a site."""
        pass

    def downloaded_to_all(self):
        """Called when the object is fully downloaded to all sites."""
        pass

    def transaction_done(self, transaction_id: str, status: str):
        """Called when the transaction is finished."""
        pass


class _SMODProducer(Producer):

    def __init__(self):
        super().__init__()

    def produce(self, ref_id: str, obj: SelfManagedObject, state: dict, requester: str) -> Tuple[str, Any, dict]:
        return obj.produce(state, requester)

    def object_downloaded(self, ref_id: str, obj: SelfManagedObject, to_site: str, status: str):
        obj.downloaded_to_one(to_site, status)

    def object_done(self, ref_id: str, obj: SelfManagedObject):
        """Called when the object is fully downloaded to all sites."""
        obj.downloaded_to_all()

    def transaction_done(self, transaction_id: str, objs: List[SelfManagedObject], status: str):
        for obj in objs:
            obj.transaction_done(transaction_id, status)


class SelfManagedObjectDownloader:

    @classmethod
    def new_transaction(
        cls, cell: Cell, timeout: float, num_receivers: int = 0, tx_id=None, timeout_cb=None, **cb_kwargs
    ):
        return ObjDownloader.new_transaction(
            cell=cell,
            producer=_SMODProducer(),
            timeout=timeout,
            num_receivers=num_receivers,
            tx_id=tx_id,
            timeout_cb=timeout_cb,
            **cb_kwargs,
        )

    @classmethod
    def add_download_object(
        cls,
        transaction_id: str,
        obj: SelfManagedObject,
        ref_id=None,
    ) -> str:
        if not issubclass(type(obj), SelfManagedObject):
            raise TypeError(f"obj must be an instance of SelfManagedObject but got {type(obj)}")

        rid = ObjDownloader.add_download_object(
            transaction_id=transaction_id,
            obj=obj,
            ref_id=ref_id,
        )
        obj.set_transaction(transaction_id, rid)
        return rid
