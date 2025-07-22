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
import threading
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Optional

from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.streaming.file_downloader import FileDownloader, download_file
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.cache import FobsCache
from nvflare.fuel.utils.fobs.datum import Datum, DatumManager, DatumType
from nvflare.fuel.utils.fobs.lobs import get_datum_dir
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.fuel.utils.msg_root_utils import subscribe_to_msg_root

# if the file size for collected items is < _MIN_SIZE_FOR_FILE, they will be attached to the message.
_MIN_SIZE_FOR_FILE = 1024 * 1024 * 2

_MIN_MSG_ROOT_TTL = 60  # allow at least 1 minute gap between download activities


class _FileRefKey:
    LOCATION = "location"
    FILE_REF_ID = "file_ref_id"
    FQCN = "fqcn"
    FILE_META = "file_meta"


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

        # some stats
        self.file_creation_time = None
        self.file_size = 0
        self.num_files = 0
        self.num_secondary_msgs = 0
        self.num_lookups = 0
        self.lock = threading.Lock()

    def set_file_creation_time(self, duration):
        self.file_creation_time = duration

    def set_file_size(self, size):
        self.file_size = size

    def inc_files(self):
        with self.lock:
            self.num_files += 1

    def inc_2nd_msgs(self):
        with self.lock:
            self.num_secondary_msgs += 1

    def inc_lookup(self):
        with self.lock:
            self.num_lookups += 1

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
        self.prefix = self.__class__.__name__
        self.decompose_ctx_key = f"{self.prefix}_dc"  # kept in fobs_ctx: each target type has its own DecomposeCtx
        self.items_key = f"{self.prefix}_items"  # in fobs_ctx: each target type has its own set of items
        self.datum_key = f"{self.prefix}_datum"  # in root: each target type has its own final datum

    @abstractmethod
    def dump_to_file(self, items: dict, path: str, fobs_ctx: dict) -> (Optional[str], Optional[dict]):
        """Dump the items to the file with the specified path

        Args:
            items: a dict of items of target object type to be dumped to file
            path: the path to the file.
            fobs_ctx: FOBS Context

        Returns: a tuple of (file name, meta info)

        The "path" is a temporary file name. You should create the file with the specified name.
        However, some frameworks (e.g. numpy) may add a special suffix to the name. In this case, you must return the
        modified name.

        The "items" is a dict of target objects. The dict contains all objects of the target type in one payload.
        The dict could be very big. You must create a file to contain all the objects.

        """
        pass

    @abstractmethod
    def load_from_file(self, path: str, fobs_ctx: dict, meta: dict = None) -> dict:
        """Load target object items from the specified file

        Args:
            path: the absolute path to the file to be loaded.
            fobs_ctx: FOBS Context.
            meta: meta info of the file.

        Returns: a dict of target objects.

        You must not delete the file after loading. Management of the file is done by the ViaFile class.

        """
        pass

    def supported_dots(self):
        return [self.get_bytes_dot(), self.get_file_dot()]

    @abstractmethod
    def get_file_dot(self) -> int:
        """Get the Datum Object Type to be used for file ref datum

        Returns: the DOT for file ref datum

        """
        pass

    @abstractmethod
    def get_bytes_dot(self) -> int:
        """Get the Datum Object Type to be used for bytes datum

        Returns: the DOT for bytes datum

        """
        pass

    def _get_temp_file_name(self):
        datum_dir = get_datum_dir()
        return os.path.join(datum_dir, f"{self.prefix}_{uuid.uuid4()}")

    def _create_ref(self, target: Any, manager: DatumManager, fobs_ctx: dict):
        # create a reference item for the target object. The ref item represents the target object in
        # the serialized payload.
        dc = fobs_ctx.get(self.decompose_ctx_key)
        item_id = dc.add_item(target)
        if dc.get_item_count() == 1:
            # register the post_process callback to further process these items.
            # only register cb once!
            manager.register_post_cb(self._post_process)
        return item_id

    def _create_file(self, items: dict, fobs_ctx: dict) -> (str, int, dict):
        dc = fobs_ctx.get(self.decompose_ctx_key)

        file_name = self._get_temp_file_name()
        try:
            self.logger.debug(f"ViaFile: dumping {len(items)} items to file {file_name}")
            start = time.time()
            new_file_name, meta = self.dump_to_file(items, file_name, fobs_ctx)
            end = time.time()
            dc.set_file_creation_time(end - start)
            dc.inc_files()
            if new_file_name:
                file_name = new_file_name
        except Exception as e:
            self.logger.error(f"Error dumping {len(items)} items to file {file_name}: {e}")
            raise e

        size = os.path.getsize(file_name)
        self.logger.debug(f"ViaFile: created file {file_name=} {size=}")
        dc.set_file_size(size)
        return file_name, size, meta

    def decompose(self, target: Any, manager: DatumManager = None) -> Any:
        if not manager:
            # this should never happen
            raise RuntimeError("FOBS System Error: missing DatumManager")

        with self.lock:
            # determination of primary msg must be done one by one sequentially!
            self._determine_primary_msg(manager)
        return self._do_decompose(target, manager)

    def _determine_primary_msg(self, manager: DatumManager):
        """Determine whether this msg is primary.
        Primary msg is responsible for generating files for all target types!
        All secondary msgs must wait until primary msg is done.

        Args:
            manager: the datum manager

        Returns: None

        """
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
            # make sure the TTL is long enough
            msg_root_ttl = _MIN_MSG_ROOT_TTL

        fobs_ctx[_CtxKey.MSG_ROOT_ID] = msg_root_id
        fobs_ctx[_CtxKey.MSG_ROOT_TTL] = msg_root_ttl

        # Create a DecomposeCtx for this target type.
        # Note: there could be multiple target types - each target type has its own DecomposeCtx!
        dc = fobs_ctx.get(self.decompose_ctx_key)
        if not dc:
            dc = _DecomposeCtx()
            fobs_ctx[self.decompose_ctx_key] = dc

        self.logger.debug(f"{tid=} got {msg_root_id=}")
        is_primary = fobs_ctx.get(_CtxKey.IS_PRIMARY)
        if is_primary is not None:
            # already determined
            pass
        elif msg_root_id:
            # when msg_root is available, it is shared among multiple messages.
            # we try to avoid creating the file for the same target when encoding these messages.
            # we create the file only once, and then save the generated ref in FobsCache under the msg_root_id.
            # note that we assume that the target object is not altered in the same msg_root!
            root = FobsCache.get_item(msg_root_id)
            if root is None:
                # this is 1st item of the primary
                is_primary = True
                self.logger.debug(f"ViaFile: created new msg root {root}")

                # create a waiter, which will be used to coordinate with secondary msgs
                waiter = threading.Event()
                root = {_RootKey.WAITER: waiter}
                FobsCache.set_item(msg_root_id, root)
            else:
                # root is already created. only the primary msg can create the root.
                is_primary = False
                waiter = root[_RootKey.WAITER]
                dc.inc_2nd_msgs()
                self.logger.debug(f"secondary: {dc.num_secondary_msgs=}")
                self.logger.debug(f"_determine_primary_msg: DCID: {id(dc)}")
            fobs_ctx[_CtxKey.WAITER] = waiter
            fobs_ctx[_CtxKey.ROOT] = root
        else:
            # if msg root is not present, then there are no secondary msgs!
            self.logger.debug("ViaFile: no msg root")
            is_primary = True

        fobs_ctx[_CtxKey.IS_PRIMARY] = is_primary
        self.logger.debug(f"{tid=} determined {is_primary=}")

    def _do_decompose(self, target: Any, manager: DatumManager) -> Any:
        tid = threading.get_ident()
        fobs_ctx = manager.fobs_ctx
        target_id = id(target)
        dc = fobs_ctx.get(self.decompose_ctx_key)
        assert isinstance(dc, _DecomposeCtx)
        root = fobs_ctx.get(_CtxKey.ROOT)
        is_primary = fobs_ctx.get(_CtxKey.IS_PRIMARY)
        self.logger.debug(f"{tid=} _do_decompose: got root {root=} {is_primary=}")

        if is_primary:
            # create and save ref in cache
            ref = self._create_ref(target, manager, fobs_ctx)
            self.logger.debug(f"{tid=} ViaFile: created ref for target {target_id}: {ref}")
            if root:
                root[target_id] = ref
                self.logger.debug(f"{tid=} ViaFile: cached ref for target {target_id}: {ref}: {root=}")
        else:
            # wait until the primary msg is done
            self.logger.debug(f"{tid=} waiting for primary msg done")
            waiter = fobs_ctx.get(_CtxKey.WAITER)
            waiter.wait()
            ref = root.get(target_id)
            if not ref:
                raise RuntimeError(f"{tid=}: cannot find ref from root {root=} for target {target_id}")
            else:
                self.logger.debug(f"{tid=} ViaFile: got ref from cached root {ref}")
                dc.inc_lookup()
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
        """This method is called during serialization after all target items are serialized.
        For primary msg, we turn the collected items into a file, and add file info as a Datum to the datum manager.

        Args:
            mgr:

        Returns:

        """
        tid = threading.get_ident()
        self.logger.debug(f"{tid}: post processing")
        fobs_ctx = mgr.fobs_ctx
        dc = fobs_ctx.get(self.decompose_ctx_key)
        assert isinstance(dc, _DecomposeCtx)
        waiter = fobs_ctx.get(_CtxKey.WAITER)
        root = fobs_ctx.get(_CtxKey.ROOT)
        msg_root_id = fobs_ctx.get(_CtxKey.MSG_ROOT_ID)

        is_primary = fobs_ctx.get(_CtxKey.IS_PRIMARY)
        self.logger.debug(f"{is_primary=}")

        if not is_primary:
            # this is the secondary msg
            self.logger.debug(f"{tid=} ViaFile: secondary msg - wait for the Datum to be created")
            if not waiter:
                raise RuntimeError("no waiter for secondary msg")

            waiter.wait()
            datum = root.get(self.datum_key)
            if not isinstance(datum, Datum):
                raise RuntimeError(f"expect to get a Datum but got {type(datum)}")
            self.logger.debug(f"{tid=} got datum from cache")
            mgr.add_datum(datum)

            if msg_root_id:
                subscribe_to_msg_root(
                    msg_root_id=msg_root_id,
                    cb=self._show_secondary_msg_stats,
                    dc=dc,
                )
            return

        # this is primary msg - create datum for the collected target items
        # This is called once for each target object type!

        # register the final CB to be called after the post_process.
        # Note that the post_process (this CB) only generates files but does not create download transaction.
        # For large object, file generation could take long time. If we create the download transaction, it may
        # become expired even before file generation is done!
        # This is why we do the file generation in this CB, and then create the transaction in the final_cb!
        if msg_root_id:
            subscribe_to_msg_root(
                msg_root_id=msg_root_id,
                cb=self._show_primary_msg_stats,
                dc=dc,
            )

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
                self.logger.debug(f"{tid=} datum created by primary msg")
        except Exception as ex:
            self.logger.error(f"exception creating datum: {ex}")
            mgr.set_error(f"exception creating datum in {type(self)}")
        finally:
            if root:
                root[self.datum_key] = datum

    def _create_datum(self, fobs_ctx: dict):
        dc = fobs_ctx.get(self.decompose_ctx_key)
        file_name, size, meta = self._create_file(dc.target_items, fobs_ctx)
        cell = fobs_ctx.get(fobs.FOBSContextKey.CELL)
        use_local_file = fobs_ctx.get(_CtxKey.USE_LOCAL_FILE, False)

        if use_local_file:
            use_file_dot = True
        elif meta:
            use_file_dot = True
        else:
            use_file_dot = cell and size > _MIN_SIZE_FOR_FILE

        if not use_file_dot:
            # use bytes DOT
            with open(file_name, "rb") as f:
                data = f.read()
            os.remove(file_name)
            datum = Datum(datum_type=DatumType.BLOB, value=data, dot=self.get_bytes_dot())
        else:
            # use file DOT
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
                    _FileRefKey.FILE_META: meta,
                }
            else:
                file_ref = {
                    _FileRefKey.LOCATION: _FileLocation.REMOTE_CELL,
                    _FileRefKey.FQCN: cell.get_fqcn(),
                    _FileRefKey.FILE_REF_ID: file_ref_id,
                    _FileRefKey.FILE_META: meta,
                }
            self.logger.debug(f"created file ref for target type {self.__class__.__name__}: {file_ref=}")
            datum = Datum(datum_type=DatumType.TEXT, value=json.dumps(file_ref), dot=self.get_file_dot())
        return datum

    def _finalize_download_tx(self, mgr: DatumManager):
        # must be primary
        fobs_ctx = mgr.fobs_ctx
        dc = fobs_ctx.get(self.decompose_ctx_key)
        use_local_file = fobs_ctx.get(_CtxKey.USE_LOCAL_FILE, False)

        tid = threading.get_ident()
        self.logger.debug(f"{tid=} ViaFile: finalizing download tx: {use_local_file=}")

        is_primary = fobs_ctx.get(_CtxKey.IS_PRIMARY)
        if not is_primary:
            raise RuntimeError("Program Error: finalize_download_tx called for non-primary!")

        download_tx_id = None

        files = fobs_ctx.get(_CtxKey.FILES)
        if not use_local_file:
            if files:
                download_tx_id = self._create_download_tx(fobs_ctx)
                for file_ref_id, file_name in files:
                    self.logger.debug(f"{tid=} ViaFile: adding file to downloader: {download_tx_id=} {file_name=}")
                    FileDownloader.add_file(
                        transaction_id=download_tx_id,
                        file_name=file_name,
                        ref_id=file_ref_id,
                    )

        msg_root_id = fobs_ctx.get(_CtxKey.MSG_ROOT_ID)
        files_to_delete = files if use_local_file else None
        self.logger.debug(f"{tid=} ViaFile: files to deleted with msg root: {files_to_delete}")
        if msg_root_id:
            subscribe_to_msg_root(
                msg_root_id=msg_root_id,
                cb=self._delete_msg_root,
                download_tx_id=download_tx_id,
                files_to_delete=files_to_delete,
            )

        # Release waiters after the download tx is fully set.
        # This is to avoid the potential race condition that the msg receiver tries to download files even
        # before the download tx is ready. This is possible because a file could take long time to create.
        waiter = fobs_ctx.get(_CtxKey.WAITER)
        if waiter:
            self.logger.debug(f"{tid=} freed waiter")
            waiter.set()

    def _show_secondary_msg_stats(self, msg_root_id: str, dc: _DecomposeCtx):
        # print stats
        stats = {
            "msg_root_id": msg_root_id,
            "num_lookups": dc.num_lookups,
        }
        self.logger.info(f"Secondary Message Info: {stats}")

    def _show_primary_msg_stats(self, msg_root_id: str, dc: _DecomposeCtx):
        # print stats
        stats = {
            "msg_root_id": msg_root_id,
            "file_creation_time": dc.file_creation_time,
            "file_size": dc.file_size,
            "num_files": dc.num_files,
            "num_items": len(dc.target_items),
        }
        self.logger.info(f"Primary Message Info: {stats}")

    def _delete_msg_root(self, msg_root_id: str, download_tx_id: str, files_to_delete):
        self.logger.debug(f"deleting msg root {msg_root_id}: {files_to_delete=}")

        FobsCache.remove_item(msg_root_id)
        if files_to_delete:
            # clean up local files
            for _, file_name in files_to_delete:
                os.remove(file_name)
                self.logger.debug(f"removed local file {file_name}")

        if download_tx_id:
            FileDownloader.delete_transaction(download_tx_id, call_cb=True)

    def _delete_download_tx(self, tx_id, file_names, msg_root_id):
        self.logger.debug(f"ViaFile: deleting download tx: {tx_id}")
        if msg_root_id:
            FobsCache.remove_item(msg_root_id)
            self.logger.debug(f"ViaFile: removed msg root {msg_root_id} from FobsCache")

        # delete all files in the tx
        for f in file_names:
            self.logger.debug(f"ViaFile: deleting download file {f}")
            try:
                os.remove(f)
            except FileNotFoundError:
                pass

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
        tid = threading.get_ident()
        self.logger.debug(f"{tid=} pre-processing datum {datum.dot=} before recompose")
        fobs_ctx = manager.fobs_ctx
        if datum.dot == self.get_bytes_dot():
            # data is in the value
            items = self._load_from_bytes(datum.value, fobs_ctx)
        else:
            # data is in a file
            file_ref = json.loads(datum.value)
            location = file_ref.get(_FileRefKey.LOCATION)
            if location == _FileLocation.LOCAL:
                # file is on local file system
                file_path = file_ref.get(_FileRefKey.FILE_REF_ID)
                remove_after_loading = False
                self.logger.debug(f"{tid=} got local file path: {file_path}")
            elif location == _FileLocation.REMOTE_CELL:
                # file is on remote cell - need to download it
                file_path = self._download_from_remote_cell(manager.fobs_ctx, file_ref)
                remove_after_loading = True
                self.logger.debug(f"{tid=} got downloaded file path: {file_path}")
            else:
                raise RuntimeError(f"unsupported file location {location}")

            file_meta = file_ref.get(_FileRefKey.FILE_META, None)
            items = self._load_items_from_file(file_path, remove_after_loading, fobs_ctx, file_meta)
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
        self.logger.debug(f"{tid=} recomposing data item {data}")
        item_id = data
        fobs_ctx = manager.fobs_ctx
        items = fobs_ctx.get(self.items_key)
        self.logger.debug(f"trying to get item for {item_id=} from {items=}")
        item = items.get(item_id)
        self.logger.debug(f"{tid=} found item {item_id}: {type(item)}")
        if item is None:
            self.logger.error(f"cannot find item {item_id} from loaded data")
        return item

    def _load_from_bytes(self, data: bytes, fobs_ctx: dict):
        file_path = self._get_temp_file_name()
        with open(file_path, "wb") as f:
            f.write(data)
        self.logger.debug(f"ViaFile recompose: created temp file {file_path}")
        return self._load_items_from_file(file_path, True, fobs_ctx, None)

    def _load_items_from_file(self, file_path: str, remove_after_loading: bool, fobs_ctx: dict, file_meta):
        items = self.load_from_file(file_path, fobs_ctx, file_meta)
        self.logger.debug(f"items loaded from file {file_path}: {type(items)}")
        if not isinstance(items, dict):
            self.logger.error(f"items loaded from file should be dict but got {type(items)}")
            items = {}
        else:
            self.logger.debug(f"number of items loaded from file: {len(items)}")
        self.logger.debug(f"loaded items: {items.keys()}")
        if remove_after_loading:
            os.remove(file_path)
        return items

    def _download_from_remote_cell(self, fobs_ctx: dict, file_ref: dict):
        self.logger.debug(f"trying to download_from_remote_cell for {file_ref=}")
        cell = fobs_ctx.get(fobs.FOBSContextKey.CELL)
        if not cell:
            self.logger.error("cannot download from remote cell since cell not available in fobs context")
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

        self.logger.debug(f"trying to download file: {file_ref_id=} {fqcn=}")
        err, file_path = download_file(
            from_fqcn=fqcn,
            ref_id=file_ref_id,
            location=get_datum_dir(),
            per_request_timeout=req_timeout,
            cell=cell,
            abort_signal=abort_signal,
        )
        if err:
            self.logger.error(f"failed to download file from {fqcn} for source {file_ref}: {err}")
            raise RuntimeError("System Error")
        else:
            self.logger.debug(f"downloaded file to {file_path}")
        return file_path
