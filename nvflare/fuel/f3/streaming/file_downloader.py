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
import os.path
import tempfile
import uuid
from typing import Any, List, Optional

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.streaming.obj_downloader import Consumer, ObjDownloader, Producer, ProduceRC, download_object
from nvflare.fuel.utils.validation_utils import check_positive_int

DEFAULT_CHUNK_SIZE = 5 * 1024 * 1024

"""
This package implements file downloading capability based on the ObjDownloader framework.
It provides implementation of the Producer and Consumer objects, required by ObjDownloader.
"""


class _StateKey:
    RECEIVED_BYTES = "received_bytes"


class _File:

    def __init__(self, file_name):
        """This is the "object" to be downloaded.

        Args:
            file_name: name of the file.
        """
        self.name = file_name
        self.size = os.path.getsize(file_name)


class _ChunkProducer(Producer):

    def __init__(self, chunk_size=None):
        Producer.__init__(self)
        if not chunk_size:
            chunk_size = DEFAULT_CHUNK_SIZE

        check_positive_int("chunk_size", chunk_size)
        self.chunk_size = chunk_size

    def produce(self, ref_id: str, obj, state: dict, requester: str) -> (str, Any, dict):
        assert isinstance(obj, _File)
        received_bytes = 0
        if state:
            received_bytes = state.get(_StateKey.RECEIVED_BYTES, 0)

        if not isinstance(received_bytes, int) or received_bytes < 0:
            self.logger.error(f"bad {_StateKey.RECEIVED_BYTES} {received_bytes} from {requester}")
            return ProduceRC.ERROR, None, None

        if received_bytes >= obj.size:
            # already done
            return ProduceRC.EOF, None, None

        num_bytes_to_send = min(self.chunk_size, obj.size - received_bytes)
        with open(obj.name, "rb") as f:
            f.seek(received_bytes)
            chunk = f.read(num_bytes_to_send)

        self.logger.debug(f"{received_bytes=}; sending {len(chunk)} bytes")
        return ProduceRC.OK, chunk, {_StateKey.RECEIVED_BYTES: received_bytes + len(chunk)}


class FileDownloader(ObjDownloader):

    @classmethod
    def new_transaction(
        cls,
        cell: Cell,
        timeout: float,
        timeout_cb,
        **cb_kwargs,
    ):
        """Create a new file download transaction.

        Args:
            cell: the cell for communication with recipients
            timeout: timeout for the transaction
            timeout_cb: CB to be called when the transaction is timed out
            **cb_kwargs: args to be passed to the CB

        Returns: transaction id

        The timeout_cb must follow this signature:

            cb(tx_id, file_names: List[str], **cb_args)

        """
        return ObjDownloader.new_transaction(
            cell=cell,
            producer=_ChunkProducer(),
            timeout=timeout,
            timeout_cb=cls._tx_timeout,
            app_timeout_cb=timeout_cb,
            **cb_kwargs,
        )

    @classmethod
    def _tx_timeout(cls, tx_id: str, objs: List[Any], app_timeout_cb, **cb_kwargs):
        if app_timeout_cb:
            file_names = [obj.name for obj in objs]
            app_timeout_cb(tx_id, file_names, **cb_kwargs)

    @classmethod
    def add_file(
        cls,
        transaction_id: str,
        file_name: str,
        ref_id=None,
        file_downloaded_cb=None,
        **cb_kwargs,
    ) -> str:
        """Add a file to be downloaded to the specified transaction.

        Args:
            transaction_id: ID of the transaction
            file_name: name of the file to be downloaded
            ref_id: ref id to be used, if provided
            file_downloaded_cb: CB to be called when the file is done downloading
            **cb_kwargs: args to be passed to the CB

        Returns: reference id for the file.

        The file_downloaded_cb must follow this signature:

            cb(ref_id: str, to_site: str, status: str, file_name: str, **cb_kwargs)

        """
        obj = _File(file_name)
        return ObjDownloader.add_download_object(
            transaction_id=transaction_id,
            obj=obj,
            ref_id=ref_id,
            obj_downloaded_cb=cls._file_downloaded,
            app_downloaded_cb=file_downloaded_cb,
            **cb_kwargs,
        )

    @classmethod
    def _file_downloaded(cls, ref_id: str, to_site: str, status: str, obj: _File, app_downloaded_cb, **cb_kwargs):
        if app_downloaded_cb:
            app_downloaded_cb(ref_id, to_site, status, obj.name, **cb_kwargs)

    @classmethod
    def download_file(
        cls,
        from_fqcn: str,
        ref_id: str,
        per_request_timeout: float,
        cell: Cell,
        location: str = None,
        secure=False,
        optional=False,
        abort_signal=None,
    ) -> (str, Optional[str]):
        """Download the referenced file from the file owner.

        Args:
            from_fqcn: FQCN of the file owner.
            ref_id: reference ID of the file to be downloaded.
            per_request_timeout: timeout for requests sent to the file owner.
            cell: cell to be used for communicating to the file owner.
            location: dir for keeping the received file. If not specified, will use temp dir.
            secure: P2P private mode for communication
            optional: supress log messages of communication
            abort_signal: signal for aborting download.

        Returns: tuple of (error message if any, full path of the downloaded file).

        """
        if location is not None:
            if not os.path.exists(location):
                raise ValueError(f"location '{location}' does not exist")

            if not os.path.isdir(location):
                raise ValueError(f"location '{location}' is not a valid dir")
        else:
            location = tempfile.gettempdir()

        consumer = _ChunkConsumer(location)
        download_object(
            from_fqcn=from_fqcn,
            ref_id=ref_id,
            consumer=consumer,
            per_request_timeout=per_request_timeout,
            cell=cell,
            secure=secure,
            optional=optional,
            abort_signal=abort_signal,
        )

        return consumer.error, consumer.file_path


class _ChunkConsumer(Consumer):

    def __init__(self, location: str):
        Consumer.__init__(self)
        self.location = location
        self.file_path = os.path.join(location, str(uuid.uuid4()))
        self.file = open(self.file_path, "wb")
        self.logger.debug(f"created file {self.file_path}")
        self.total_bytes = 0
        self.error = None

    def consume(self, ref_id, state: dict, data: Any) -> dict:
        assert isinstance(data, bytes)
        self.file.write(data)
        self.total_bytes += len(data)
        self.logger.debug(f"received {self.total_bytes} bytes for file {self.file_path}")
        return {_StateKey.RECEIVED_BYTES: self.total_bytes}

    def download_failed(self, ref_id, reason: str):
        self.logger.error(f"failed to download file with ref {ref_id}: {reason}")
        self.error = reason
        self.file.close()

    def download_completed(self, ref_id: str):
        self.file.close()
        self.logger.debug(f"closed file {self.file_path}")


def download_file(
    from_fqcn: str,
    ref_id: str,
    per_request_timeout: float,
    cell: Cell,
    location: str = None,
    secure=False,
    optional=False,
    abort_signal=None,
) -> (str, Optional[str]):
    return FileDownloader.download_file(
        from_fqcn, ref_id, per_request_timeout, cell, location, secure, optional, abort_signal
    )
