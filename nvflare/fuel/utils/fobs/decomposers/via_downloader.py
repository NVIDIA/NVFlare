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
import json
import threading
import uuid
from abc import ABC, abstractmethod
from typing import Any, Tuple

import nvflare.fuel.utils.app_config_utils as acu
from nvflare.apis.fl_constant import ConfigVarName
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.streaming.download_service import Downloadable
from nvflare.fuel.f3.streaming.file_downloader import ObjectDownloader
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.datum import Datum, DatumManager, DatumType
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.fuel.utils.msg_root_utils import subscribe_to_msg_root

_MIN_DOWNLOAD_TIMEOUT = 60  # allow at least 1 minute gap between download activities


class EncKey:
    TYPE = "type"
    DATA = "data"


class EncType:
    NATIVE = "native"
    REF = "ref"


class _RefKey:
    REF_ID = "ref_id"
    FQCN = "fqcn"


class _CtxKey:
    MSG_ROOT_ID = "msg_root_id"
    MSG_ROOT_TTL = "msg_root_ttl"
    OBJECTS = "objects"  # objects to be downloaded
    FINAL_CB_REGISTERED = "final_cb_registered"


class _DecomposeCtx:

    def __init__(self):
        self.target_to_item = {}  # target_id => item_id
        self.target_items = {}  # item_id => item value
        self.last_item_id = 0

    def add_item(self, item: Any):
        target_id = id(item)
        item_id = self.target_to_item.get(target_id)
        if not item_id:
            item_id = f"T{self.last_item_id}"
            self.last_item_id += 1
            self.target_items[item_id] = item
            self.target_to_item[target_id] = item_id
        return item_id, target_id

    def get_item_count(self):
        return len(self.target_items)


class ViaDownloaderDecomposer(fobs.Decomposer, ABC):

    def __init__(self, max_chunk_size: int, config_var_prefix):
        self.logger = get_obj_logger(self)
        self.prefix = self.__class__.__name__
        self.decompose_ctx_key = f"{self.prefix}_dc"  # kept in fobs_ctx: each target type has its own DecomposeCtx
        self.items_key = f"{self.prefix}_items"  # in fobs_ctx: each target type has its own set of items
        self.config_var_prefix = config_var_prefix
        self.max_chunk_size = max_chunk_size

    @abstractmethod
    def to_downloadable(self, items: dict, max_chunk_size: int, fobs_ctx: dict) -> Downloadable:
        """Convert the items Downloadable object.

        Args:
            items: a dict of items of target object type to be converted
            max_chunk_size: max size of one chunk.
            fobs_ctx: FOBS Context

        Returns: a Downloadable object

        The "items" is a dict of target objects. The dict contains all objects of the target type in one payload.

        """
        pass

    @abstractmethod
    def download(
        self,
        from_fqcn: str,
        ref_id: str,
        per_request_timeout: float,
        cell: Cell,
        secure=False,
        optional=False,
        abort_signal=None,
    ) -> Tuple[str, dict]:
        pass

    def supported_dots(self):
        return [self.get_download_dot()]

    @abstractmethod
    def get_download_dot(self) -> int:
        """Get the Datum Object Type to be used for download ref datum

        Returns: the DOT for download ref datum

        """
        pass

    @abstractmethod
    def native_decompose(self, target: Any, manager: DatumManager = None) -> bytes:
        pass

    @abstractmethod
    def native_recompose(self, data: bytes, manager: DatumManager = None) -> Any:
        pass

    def _create_ref(self, target: Any, manager: DatumManager, fobs_ctx: dict):
        # create a reference item for the target object. The ref item represents the target object in
        # the serialized payload.
        dc = fobs_ctx.get(self.decompose_ctx_key)
        item_id, target_id = dc.add_item(target)
        if dc.get_item_count() == 1:
            # register the post_process callback to further process these items.
            # only register cb once!
            manager.register_post_cb(self._process_items_to_datum)
        return item_id, target_id

    def _create_downloadable(self, fobs_ctx: dict) -> Downloadable:
        dc = fobs_ctx.get(self.decompose_ctx_key)
        assert isinstance(dc, _DecomposeCtx)
        items = dc.target_items
        max_chunk_size = acu.get_int_var(
            self._config_var_name(ConfigVarName.DOWNLOAD_CHUNK_SIZE),
            self.max_chunk_size,
        )
        try:
            return self.to_downloadable(items, max_chunk_size, fobs_ctx)
        except Exception as e:
            self.logger.error(f"Error converting {len(items)} items to Downloadable: {e}")
            raise e

    @staticmethod
    def _determine_msg_root(fobs_ctx: dict):
        msg_root_id = fobs_ctx.get(_CtxKey.MSG_ROOT_ID)
        msg_root_ttl = fobs_ctx.get(_CtxKey.MSG_ROOT_TTL)

        if not msg_root_id:
            # try to get from msg
            msg = fobs_ctx.get(fobs.FOBSContextKey.MESSAGE)
            if msg:
                msg_root_id = msg.get_header(MessageHeaderKey.MSG_ROOT_ID)
                msg_root_ttl = msg.get_header(MessageHeaderKey.MSG_ROOT_TTL)
        return msg_root_id, msg_root_ttl

    def decompose(self, target: Any, manager: DatumManager = None) -> Any:
        if not manager:
            # this should never happen
            raise RuntimeError("FOBS System Error: missing DatumManager")

        max_chunk_size = acu.get_int_var(
            self._config_var_name(ConfigVarName.DOWNLOAD_CHUNK_SIZE),
            self.max_chunk_size,
        )
        fobs_ctx = manager.fobs_ctx
        cell = fobs_ctx.get(fobs.FOBSContextKey.CELL)
        if not cell:
            # If no cell, only support native decomposers
            fobs_ctx["native"] = True

        use_native = fobs_ctx.get("native", False)
        if max_chunk_size <= 0 or use_native:
            # use native decompose
            self.logger.debug("using native_decompose")
            data = self.native_decompose(target, manager)
            return {EncKey.TYPE: EncType.NATIVE, EncKey.DATA: data}
        else:
            self.logger.debug(f"using download decompose: {max_chunk_size=}")

        # Create a DecomposeCtx for this target type.
        # Note: there could be multiple target types - each target type has its own DecomposeCtx!
        dc = fobs_ctx.get(self.decompose_ctx_key)
        if not dc:
            dc = _DecomposeCtx()
            fobs_ctx[self.decompose_ctx_key] = dc

        item_id, target_id = self._create_ref(target, manager, fobs_ctx)
        self.logger.debug(f"ViaDownloader: created ref for target {target_id}: {item_id}")
        return {EncKey.TYPE: EncType.REF, EncKey.DATA: item_id}

    def _create_downloader(self, fobs_ctx: dict):
        msg_root_id, msg_root_ttl = self._determine_msg_root(fobs_ctx)

        if msg_root_ttl:
            timeout = msg_root_ttl
        else:
            timeout = _MIN_DOWNLOAD_TIMEOUT

        if timeout < _MIN_DOWNLOAD_TIMEOUT:
            timeout = _MIN_DOWNLOAD_TIMEOUT

        self.logger.debug(f"ViaDownloader: {msg_root_id=} {timeout=}")

        downloader = None
        cell = fobs_ctx.get(fobs.FOBSContextKey.CELL)
        if cell:
            downloader = ObjectDownloader(
                num_receivers=1,
                cell=cell,
                timeout=timeout,
            )

        if msg_root_id:
            subscribe_to_msg_root(
                msg_root_id=msg_root_id,
                cb=self._delete_download_tx_on_msg_root,
                downloader=downloader,
            )

        return downloader

    def _process_items_to_datum(self, mgr: DatumManager):
        """This method is called during serialization after all target items are serialized.
        For primary msg, we turn the collected items into a file, and add file info as a Datum to the datum manager.

        Args:
            mgr:

        Returns:

        """
        fobs_ctx = mgr.fobs_ctx
        dc = fobs_ctx.get(self.decompose_ctx_key)
        assert isinstance(dc, _DecomposeCtx)

        # create datum for the collected target items
        # This is called once for each target object type!

        # register the final CB to be called after the post_process.
        # Note that the post_process (this CB) only generates files but does not create download transaction.
        # For large object, file generation could take long time. If we create the download transaction, it may
        # become expired even before file generation is done!
        # This is why we do the file generation in this CB, and then create the transaction in the final_cb!
        final_cb_registered = fobs_ctx.get(_CtxKey.FINAL_CB_REGISTERED)
        if not final_cb_registered:
            # register final_cb
            mgr.register_post_cb(self._finalize_download_tx)
            fobs_ctx[_CtxKey.FINAL_CB_REGISTERED] = True

        try:
            if not mgr.get_error():
                datum = self._create_datum(fobs_ctx)
                mgr.add_datum(datum)
        except Exception as ex:
            self.logger.error(f"exception creating datum: {ex}")
            mgr.set_error(f"exception creating datum in {type(self)}")

    def _config_var_name(self, base_name: str):
        return f"{self.config_var_prefix}{base_name}"

    def _create_datum(self, fobs_ctx: dict):
        downloadable = self._create_downloadable(fobs_ctx)
        cell = fobs_ctx.get(fobs.FOBSContextKey.CELL)

        # use download DOT
        # keep files in fobs_ctx
        downloadable_objs = fobs_ctx.get(_CtxKey.OBJECTS)
        if not downloadable_objs:
            downloadable_objs = []
            fobs_ctx[_CtxKey.OBJECTS] = downloadable_objs

        # create a new ref id
        ref_id = str(uuid.uuid4())
        downloadable_objs.append((ref_id, downloadable))

        ref = {
            _RefKey.FQCN: cell.get_fqcn(),
            _RefKey.REF_ID: ref_id,
        }
        self.logger.debug(f"ViaDownloader: created download ref for target type {self.__class__.__name__}: {ref=}")
        datum = Datum(datum_type=DatumType.TEXT, value=json.dumps(ref), dot=self.get_download_dot())
        return datum

    def _finalize_download_tx(self, mgr: DatumManager):
        self.logger.debug("ViaDownloader: finalizing download tx")
        fobs_ctx = mgr.fobs_ctx
        downloadable_objs = fobs_ctx.get(_CtxKey.OBJECTS)

        if downloadable_objs:
            downloader = self._create_downloader(fobs_ctx)
            for ref_id, obj in downloadable_objs:
                self.logger.debug(f"ViaDownloader: adding object to downloader: {ref_id=}")
                downloader.add_object(obj, ref_id=ref_id)

    def _delete_download_tx_on_msg_root(self, msg_root_id: str, downloader: ObjectDownloader):
        # this CB is triggered when msg root is deleted.
        self.logger.debug(f"ViaDownloader: deleting download transaction associated with {msg_root_id=}")
        downloader.delete_transaction()

    def process_datum(self, datum: Datum, manager: DatumManager):
        """This is called by the manager to process a datum that has a DOT.
        This happens before the recompose processing.

        The datum contains information about where the data is:
        For bytes DOT, the data is included in the datum directly.
        For file DOT, the data is in a file, and the location of the file is further specified:
            - If the location is local, then the file is on local file system;
            - If the location is remote_cell, then the file is on a remote cell, and needs to be downloaded.

        Args:
            datum: datum to be processed.
            manager: the datum manager.

        Returns: None

        """
        self.logger.debug(f"ViaDownloader: pre-processing datum {datum.dot=} before recompose")
        fobs_ctx = manager.fobs_ctx

        # data is to be downloaded
        ref = json.loads(datum.value)
        items = self._download_from_remote_cell(manager.fobs_ctx, ref)
        fobs_ctx[self.items_key] = items

    def recompose(self, data: Any, manager: DatumManager = None) -> Any:
        if not manager:
            # should never happen!
            raise RuntimeError("missing DatumManager")

        if not isinstance(data, dict):
            self.logger.error(f"data to be recomposed should be dict but got {type(data)}")
            raise RuntimeError("FOBS protocol error")

        enc_type = data.get(EncKey.TYPE)
        data = data.get(EncKey.DATA)
        if not data:
            self.logger.error("missing 'data' property from the recompose data")
            raise RuntimeError("FOBS protocol error")

        if enc_type == EncType.NATIVE:
            self.logger.debug("using native_recompose")
            return self.native_recompose(data, manager)
        elif enc_type != EncType.REF:
            self.logger.error(f"invalid enc_type {enc_type} in recompose data")
            raise RuntimeError("FOBS protocol error")

        if not isinstance(data, str):
            self.logger.error(f"ref data must be str but got {type(data)}")
            raise RuntimeError("FOBS protocol error")

        # data is the item id
        tid = threading.get_ident()
        self.logger.debug(f"ViaDownloader: {tid=} recomposing data item {data}")
        item_id = data
        fobs_ctx = manager.fobs_ctx
        items = fobs_ctx.get(self.items_key)
        self.logger.debug(f"trying to get item for {item_id=} from {type(items)=}")
        item = items.get(item_id)
        self.logger.debug(f"{tid=} found item {item_id}: {type(item)}")
        if item is None:
            self.logger.error(f"cannot find item {item_id} from loaded data")
        return item

    def _download_from_remote_cell(self, fobs_ctx: dict, ref: dict):
        self.logger.debug(f"trying to download from remote cell for {ref=}")
        cell = fobs_ctx.get(fobs.FOBSContextKey.CELL)
        if not cell:
            self.logger.error("cannot download from remote cell since cell not available in fobs context")
            raise RuntimeError("FOBS Protocol Error")

        ref_id = ref.get(_RefKey.REF_ID)
        if not ref_id:
            self.logger.error(f"missing {_RefKey.REF_ID} from {ref}")
            raise RuntimeError("FOBS Protocol Error")

        fqcn = ref.get(_RefKey.FQCN)
        if not fqcn:
            self.logger.error(f"missing {_RefKey.FQCN} from {ref}")
            raise RuntimeError("FOBS Protocol Error")

        req_timeout = fobs_ctx.get(fobs.FOBSContextKey.DOWNLOAD_REQ_TIMEOUT, None)
        if not req_timeout:
            req_timeout = acu.get_positive_float_var(
                self._config_var_name(ConfigVarName.STREAMING_PER_REQUEST_TIMEOUT), 10.0
            )
        self.logger.debug(f"DOWNLOAD_REQ_TIMEOUT={req_timeout}")

        abort_signal = fobs_ctx.get(fobs.FOBSContextKey.ABORT_SIGNAL)

        self.logger.debug(f"trying to download: {ref_id=} {fqcn=}")
        err, items = self.download(
            from_fqcn=fqcn,
            ref_id=ref_id,
            per_request_timeout=req_timeout,
            cell=cell,
            abort_signal=abort_signal,
        )
        if err:
            self.logger.error(f"failed to download from {fqcn} for source {ref}: {err}")
            raise RuntimeError(f"failed to download from {fqcn}")
        else:
            self.logger.debug(f"downloaded {len(items)} items successfully")
        return items
