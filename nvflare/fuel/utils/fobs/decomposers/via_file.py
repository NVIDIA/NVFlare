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
import os
import tempfile
import threading
import uuid
from abc import ABC, abstractmethod
from typing import Any

from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.streaming.file_downloader import FileDownloader, download_file
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.cache import FobsCache
from nvflare.fuel.utils.fobs.datum import Datum, DatumManager, DatumType
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.fuel.utils.msg_root_utils import subscribe_to_msg_root

# if the file size for collected items is < _MIN_SIZE_FOR_FILE, they will be attached to the message.
_MIN_SIZE_FOR_FILE = 1024 * 1024 * 2

_MIN_MSG_ROOT_TTL = 60  # allow at least 1 minute gap between download activities


class _FileRefKey:
    LOCATION = "location"
    FILE_REF_ID = "file_ref_id"
    FQCN = "fqcn"


class _FileLocation:
    LOCAL = "local"
    REMOTE_CELL = "remote_cell"


class _CtxKey:
    MSG_ROOT_ID = "msg_root_id"
    MSG_ROOT_TTL = "msg_root_ttl"
    IS_PRIMARY = "is_primary"
    ROOT = "root"
    FILES = "files"  # files to be downloaded
    FINAL_CB_REGISTERED = "final_cb_registered"
    WAITER = "waiter"
    USE_LOCAL_FILE = "use_local_file"  # for test purpose


class _RootKey:
    WAITER = "waiter"
    FILES = "files"


class _DecomposeCtx:

    def __init__(self):
        self.target_to_item = {}  # target_id => item_id
        self.target_items = {}  # item_id => item value
        self.last_item_id = 0
        self.waiter = None

    def add_item(self, item: Any):
        target_id = id(item)
        item_id = self.target_to_item.get(target_id)
        if item_id:
            return item_id

        item_id = f"T{self.last_item_id}"
        self.last_item_id += 1
        self.target_items[item_id] = item
        self.target_to_item[target_id] = item_id
        return item_id

    def get_item_count(self):
        return len(self.target_items)


class ViaFileDecomposer(fobs.Decomposer, ABC):

    def __init__(self):
        self.logger = get_obj_logger(self)
        self.lock = threading.Lock()
        prefix = self.__class__.__name__
        self.decompose_ctx_key = f"{prefix}_dc"  # kept in fobs_ctx: each target type has its own DecomposeCtx
        self.items_key = f"{prefix}_items"  # in fobs_ctx: each target type has its own set of items
        self.datum_key = f"{prefix}_datum"  # in root: each target type has its own final datum

    @abstractmethod
    def dump_to_file(self, items: dict, path: str):
        pass

    @abstractmethod
    def load_from_file(self, path: str) -> dict:
        pass

    @abstractmethod
    def get_remote_dat(self) -> int:
        """Get the Datum App Type to be used for remote refs

        Returns: the dat for remote refs

        """
        pass

    @abstractmethod
    def get_local_dat(self) -> int:
        """Get the Datum App Type to be used for local refs

        Returns: the dat for local refs

        """
        pass

    @staticmethod
    def _get_temp_file_name():
        return os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))

    def _create_ref(self, target: Any, manager: DatumManager, fobs_ctx: dict):
        dc = fobs_ctx.get(self.decompose_ctx_key)
        item_id = dc.add_item(target)
        self.logger.info(f"ViaFile: created ref {item_id=}")
        if dc.get_item_count() == 1:
            # only register cb once!
            manager.register_post_cb(self._post_process)
        return item_id

    def _create_file(self, items: dict) -> (str, int):
        file_name = self._get_temp_file_name()
        self.logger.info(f"ViaFile: use file name {file_name}")
        try:
            self.logger.info(f"ViaFile: dumping {len(items)} items to file {file_name}")
            new_file_name = self.dump_to_file(items, file_name)
            if new_file_name:
                file_name = new_file_name
        except Exception as e:
            self.logger.error(f"Error dumping {len(items)} items to file {file_name}: {e}")
            raise e

        size = os.path.getsize(file_name)
        self.logger.info(f"ViaFile: created file {file_name=} {size=}")
        return file_name, size

    def decompose(self, target: Any, manager: DatumManager = None) -> Any:
        print(f"Decomposer ID {id(self)}")
        if not manager:
            # this should never happen
            raise RuntimeError("FOBS System Error: missing DatumManager")

        with self.lock:
            # make sure decomposition of this decomposer is done one at a time
            self._determine_primary_msg(manager)
        return self._do_decompose(target, manager)

    def _determine_primary_msg(self, manager: DatumManager):
        tid = threading.get_ident()
        fobs_ctx = manager.fobs_ctx

        # see if msg_root info is set in fobs_ctx already
        msg_root_id = fobs_ctx.get(_CtxKey.MSG_ROOT_ID)
        msg_root_ttl = fobs_ctx.get(_CtxKey.MSG_ROOT_TTL)
        if not msg_root_id:
            # try to get from msg
            msg = fobs_ctx.get(fobs.FOBSContextKey.MESSAGE)
            if msg:
                msg_root_id = msg.get_header(MessageHeaderKey.MSG_ROOT_ID)
                msg_root_ttl = msg.get_header(MessageHeaderKey.MSG_ROOT_TTL)

        if not msg_root_ttl or msg_root_ttl < _MIN_MSG_ROOT_TTL:
            msg_root_ttl = _MIN_MSG_ROOT_TTL

        fobs_ctx[_CtxKey.MSG_ROOT_ID] = msg_root_id
        fobs_ctx[_CtxKey.MSG_ROOT_TTL] = msg_root_ttl

        dc = fobs_ctx.get(self.decompose_ctx_key)
        if not dc:
            dc = _DecomposeCtx()
            fobs_ctx[self.decompose_ctx_key] = dc

        print(f"{tid=} got {msg_root_id=}")

        is_primary = fobs_ctx.get(_CtxKey.IS_PRIMARY)
        if is_primary is not None:
            # already determined
            pass
        elif msg_root_id:
            # when msg_root is available, it is shared among multiple messages.
            # we try to avoid creating the file for the same target when encoding these messages.
            # we create the file only once, and then save the generated ref in FobsCache under the msg_root_id.
            # note that we assume that the target object is not altered in the same msg_root!
            self.logger.info(f"ViaFile: got {msg_root_id=} {msg_root_ttl=}")
            root = FobsCache.get_item(msg_root_id)
            if root is None:
                # this is 1st item of the primary
                print("NO ROOT - create new")
                is_primary = True
                self.logger.info(f"ViaFile: created and cached root {root}")
                waiter = threading.Event()
                root = {_RootKey.WAITER: waiter}
                FobsCache.set_item(msg_root_id, root)
            else:
                is_primary = False
                waiter = root[_RootKey.WAITER]
            fobs_ctx[_CtxKey.WAITER] = waiter
            fobs_ctx[_CtxKey.ROOT] = root
        else:
            self.logger.info(f"ViaFile: no msg root")
            is_primary = True

        fobs_ctx[_CtxKey.IS_PRIMARY] = is_primary
        print(f"{tid=} determined {is_primary=}")

    def _do_decompose(self, target: Any, manager: DatumManager) -> Any:
        tid = threading.get_ident()
        fobs_ctx = manager.fobs_ctx
        target_id = id(target)
        dc = fobs_ctx.get(self.decompose_ctx_key)
        assert isinstance(dc, _DecomposeCtx)
        root = fobs_ctx.get(_CtxKey.ROOT)
        is_primary = fobs_ctx.get(_CtxKey.IS_PRIMARY)
        print(f"{tid=} got root {root=} {is_primary=}")

        if is_primary:
            # create and save ref in cache
            ref = self._create_ref(target, manager, fobs_ctx)
            print(f"{tid=} ViaFile: created ref for target {target_id}: {ref}")
            if root:
                root[target_id] = ref
                print(f"{tid=} ViaFile: cached ref for target {target_id}: {ref}: {root=}")
        else:
            # wait until the primary msg is done
            print(f"{tid=} waiting for primary msg done")
            waiter = fobs_ctx.get(_CtxKey.WAITER)
            waiter.wait()
            ref = root.get(target_id)
            if not ref:
                raise RuntimeError(f"{tid=}: cannot find ref from root {root=} for target {target_id}")
            else:
                self.logger.info(f"ViaFile: got ref from cached root {ref}")
                print(f"{tid=} ViaFile: got ref from cached root {ref}")
                dc.target_items[target_id] = ref
                if len(dc.target_items) == 1:
                    manager.register_post_cb(self._post_process)
        return ref

    def _create_download_tx(self, fobs_ctx: dict):
        msg_root_id = fobs_ctx[_CtxKey.MSG_ROOT_ID]
        msg_root_ttl = fobs_ctx[_CtxKey.MSG_ROOT_TTL]
        tx_id = None
        cell = fobs_ctx.get(fobs.FOBSContextKey.CELL)
        if cell:
            tx_id = FileDownloader.new_transaction(
                cell=cell,
                timeout=msg_root_ttl,
                timeout_cb=self._delete_download_tx,
                msg_root_id=msg_root_id,
            )

        return tx_id

    def _post_process(self, mgr: DatumManager):
        tid = threading.get_ident()
        self.logger.info("ViaFile: post processing ...")
        print(f"{tid}: post processing")
        fobs_ctx = mgr.fobs_ctx
        dc = fobs_ctx.get(self.decompose_ctx_key)
        assert isinstance(dc, _DecomposeCtx)
        waiter = fobs_ctx.get(_CtxKey.WAITER)
        root = fobs_ctx.get(_CtxKey.ROOT)

        is_primary = fobs_ctx.get(_CtxKey.IS_PRIMARY)
        print(f"{is_primary=}")

        if not is_primary:
            # this is the secondary msg
            self.logger.info(f"{tid=} ViaFile: secondary msg - wait for the Datum to be created")
            print(f"{tid=} ViaFile: secondary msg - wait for the Datum to be created")
            if not waiter:
                raise RuntimeError("no waiter for secondary msg")

            waiter.wait()
            datum = root.get(self.datum_key)
            if not isinstance(datum, Datum):
                raise RuntimeError(f"expect to get a Datum but got {type(datum)}")
            print(f"{tid=} got datum from cache")
            mgr.add_datum(datum)
            return

        # this is primary msg - create datum for the collected target items
        # must guarantee that the waiter is set finally; otherwise the secondary msgs will be stuck forever
        final_cb_registered = fobs_ctx.get(_CtxKey.FINAL_CB_REGISTERED)
        if not final_cb_registered:
            # register final_cb
            mgr.register_post_cb(self._finalize_download_tx)
            fobs_ctx[_CtxKey.FINAL_CB_REGISTERED] = True

        datum = None
        try:
            if not mgr.get_error():
                datum = self._create_datum(fobs_ctx)
                mgr.add_datum(datum)
                print(f"{tid=} datum created by primary msg")
        except Exception as ex:
            self.logger.error(f"exception creating datum: {ex}")
            mgr.set_error(f"exception creating datum in {type(self)}")
        finally:
            if root:
                root[self.datum_key] = datum

    def _create_datum(self, fobs_ctx: dict):
        dc = fobs_ctx.get(self.decompose_ctx_key)
        file_name, size = self._create_file(dc.target_items)
        cell = fobs_ctx.get(fobs.FOBSContextKey.CELL)
        use_local_file = fobs_ctx.get(_CtxKey.USE_LOCAL_FILE, False)

        if use_local_file:
            use_remote_dat = True
        else:
            use_remote_dat = cell and size > _MIN_SIZE_FOR_FILE

        if not use_remote_dat:
            # use local DAT
            with open(file_name, "rb") as f:
                data = f.read()
            os.remove(file_name)
            datum = Datum(datum_type=DatumType.BLOB, value=data, app_type=self.get_local_dat())
        else:
            # use remote DAT
            # keep files in fobs_ctx
            files = fobs_ctx.get(_CtxKey.FILES)
            if not files:
                files = []
                fobs_ctx[_CtxKey.FILES] = files
                root = fobs_ctx[_CtxKey.ROOT]
                root[_RootKey.FILES] = files

            # create a new ref id
            file_ref_id = str(uuid.uuid4())
            files.append((file_ref_id, file_name))

            if use_local_file:
                file_ref = {
                    _FileRefKey.LOCATION: _FileLocation.LOCAL,
                    _FileRefKey.FILE_REF_ID: file_name,
                }
            else:
                file_ref = {
                    _FileRefKey.LOCATION: _FileLocation.REMOTE_CELL,
                    _FileRefKey.FQCN: cell.get_fqcn(),
                    _FileRefKey.FILE_REF_ID: file_ref_id,
                }
            print(f"created final file ref: {file_ref=}")
            datum = Datum(datum_type=DatumType.TEXT, value=json.dumps(file_ref), app_type=self.get_remote_dat())
        return datum

    def _finalize_download_tx(self, mgr: DatumManager):
        # must be primary
        tid = threading.get_ident()
        print(f"{tid=} ViaFile: finalizing download tx ...")
        self.logger.info("ViaFile: finalizing download tx ...")

        fobs_ctx = mgr.fobs_ctx
        is_primary = fobs_ctx.get(_CtxKey.IS_PRIMARY)
        if not is_primary:
            raise RuntimeError("Program Error: finalize_download_tx called for non-primary!")

        download_tx_id = None

        use_local_file = fobs_ctx.get(_CtxKey.USE_LOCAL_FILE, False)
        if not use_local_file:
            files = fobs_ctx.get(_CtxKey.FILES)
            if files:
                download_tx_id = self._create_download_tx(fobs_ctx)
                for file_ref_id, file_name in files:
                    print(f"{tid=} ViaFile: adding file to downloader: {download_tx_id=} {file_name=}")
                    self.logger.info(f"ViaFile: adding file to downloader: {download_tx_id=} {file_name=}")
                    FileDownloader.add_file(
                        transaction_id=download_tx_id,
                        file_name=file_name,
                        ref_id=file_ref_id,
                    )

        msg_root_id = fobs_ctx.get(_CtxKey.MSG_ROOT_ID)
        if msg_root_id:
            subscribe_to_msg_root(
                msg_root_id=msg_root_id,
                cb=self._delete_msg_root,
                download_tx_id=download_tx_id,
                expected_msg_root=msg_root_id,
                fobs_ctx=fobs_ctx,
            )

        # Release waiters after the download tx is fully set.
        # This is to avoid the potential race condition that the msg receiver tries to download files even
        # before the download tx is ready. This is possible because a file could take long time to create.
        waiter = fobs_ctx.get(_CtxKey.WAITER)
        if waiter:
            print(f"{tid=} freed waiter")
            waiter.set()

    def _delete_msg_root(
        self, topic: str, msg_root_id: str, download_tx_id: str, expected_msg_root: str, fobs_ctx: dict
    ):
        self.logger.info(f"received DB event {topic}: deleting {download_tx_id=} for msg root {msg_root_id}")
        root = FobsCache.remove(msg_root_id)
        if root:
            use_local_file = fobs_ctx.get(_CtxKey.USE_LOCAL_FILE)
            if use_local_file:
                # clean up local files
                files = fobs_ctx.get(_CtxKey.FILES)
                if files:
                    for _, file_name in files:
                        os.remove(file_name)
                        print(f"removed local file {file_name}")

        if msg_root_id == expected_msg_root and download_tx_id:
            FileDownloader.delete_transaction(download_tx_id, call_cb=True)

    def _delete_download_tx(self, tx_id, file_names, msg_root_id):
        self.logger.info(f"ViaFile: deleting download tx: {tx_id}")
        if msg_root_id:
            self.logger.info(f"ViaFile: removed msg root {msg_root_id} from FobsCache")
            print(f"ViaFile: removed msg root {msg_root_id} from FobsCache")

        # delete all files
        for f in file_names:
            self.logger.info(f"ViaFile: deleting download file {f}")
            print(f"ViaFile: deleting download file {f}")
            try:
                os.remove(f)
            except FileNotFoundError:
                pass

    def process_datum(self, datum: Datum, manager: DatumManager):
        tid = threading.get_ident()
        print(f"{tid=} pre-processing datum {datum.app_type=} before recompose")
        fobs_ctx = manager.fobs_ctx
        if datum.app_type == self.get_local_dat():
            # data is in the value
            items = self._load_from_bytes(datum.value)
        else:
            file_ref = json.loads(datum.value)
            location = file_ref.get(_FileRefKey.LOCATION)
            if location == _FileLocation.LOCAL:
                file_path = file_ref.get(_FileRefKey.FILE_REF_ID)
                remove_after_loading = False
                print(f"{tid=} got local file path: {file_path}")
            else:
                file_path = self._download_from_remote_cell(manager.fobs_ctx, file_ref)
                remove_after_loading = True
                print(f"{tid=} got downloaded file path: {file_path}")

            items = self._load_items_from_file(file_path, remove_after_loading)
        fobs_ctx[self.items_key] = items

    def recompose(self, data: Any, manager: DatumManager = None) -> Any:
        if not manager:
            # should never happen!
            raise RuntimeError("missing DatumManager")

        if not isinstance(data, str):
            self.logger.error(f"data to be recomposed should be str but got {type(data)}")
            raise RuntimeError("FOBS protocol error")

        # data is the item id
        tid = threading.get_ident()
        print(f"{tid=} recomposing data item {data}")
        item_id = data
        fobs_ctx = manager.fobs_ctx
        items = fobs_ctx.get(self.items_key)
        self.logger.info(f"trying to get item for {item_id=} from {items=}")
        item = items.get(item_id)
        self.logger.info(f"found item {item_id}: {type(item)}")
        print(f"{tid=} found item {item_id}: {type(item)}")
        if item is None:
            self.logger.error(f"cannot find item {item_id} from loaded data")
        else:
            self.logger.info(f"found item {item_id}: {type(item)}")
        return item

    def _load_from_bytes(self, data: bytes):
        file_path = self._get_temp_file_name()
        with open(file_path, "wb") as f:
            f.write(data)
        self.logger.info(f"ViaFile recompose: created temp file {file_path}")
        return self._load_items_from_file(file_path, True)

    def _load_items_from_file(self, file_path: str, remove_after_loading: bool):
        items = self.load_from_file(file_path)
        self.logger.info(f"items loaded from file {file_path}: {type(items)}")
        if not isinstance(items, dict):
            self.logger.error(f"items loaded from file should be dict but got {type(items)}")
            items = {}
        else:
            self.logger.info(f"number of items loaded from file: {len(items)}")
        self.logger.info(f"loaded items: {items.keys()}")
        if remove_after_loading:
            os.remove(file_path)
        return items

    def _download_from_remote_cell(self, fobs_ctx: dict, file_ref: dict):
        self.logger.info(f"trying to download_from_remote_cell for {file_ref=}")
        cell = fobs_ctx.get(fobs.FOBSContextKey.CELL)
        if not cell:
            self.logger.error("cell not available in fobs context")
            raise RuntimeError("FOBS Protocol Error")

        file_ref_id = file_ref.get(_FileRefKey.FILE_REF_ID)
        if not file_ref_id:
            self.logger.error(f"missing {_FileRefKey.FILE_REF_ID} from {file_ref}")
            raise RuntimeError("FOBS Protocol Error")

        fqcn = file_ref.get(_FileRefKey.FQCN)
        if not fqcn:
            self.logger.error(f"missing {_FileRefKey.FQCN} from {file_ref}")
            raise RuntimeError("FOBS Protocol Error")

        req_timeout = fobs_ctx.get(fobs.FOBSContextKey.DOWNLOAD_REQ_TIMEOUT, 10.0)
        abort_signal = fobs_ctx.get(fobs.FOBSContextKey.ABORT_SIGNAL)

        self.logger.info(f"trying to download file: {file_ref_id=} {fqcn=}")
        err, file_path = download_file(
            from_fqcn=fqcn,
            ref_id=file_ref_id,
            per_request_timeout=req_timeout,
            cell=cell,
            abort_signal=abort_signal,
        )
        if err:
            self.logger.error(f"failed to download file from source {file_ref}: {err}")
            raise RuntimeError("System Error")
        else:
            self.logger.info(f"downloaded file to {file_path}")
        return file_path
