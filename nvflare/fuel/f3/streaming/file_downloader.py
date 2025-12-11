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
from typing import Any, Optional, Tuple

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.streaming.download_service import Consumer, Downloadable, ProduceRC, download_object
from nvflare.fuel.f3.streaming.obj_downloader import ObjectDownloader
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.fuel.utils.validation_utils import check_callable, check_positive_int

DEFAULT_CHUNK_SIZE = 5 * 1024 * 1024

"""
This package implements file downloading capability based on the ObjectDownloader framework.
It provides implementation of the Downloadable and Consumer objects, required by ObjDownloader.
"""


class _StateKey:
    RECEIVED_BYTES = "received_bytes"


class FileDownloadable(Downloadable):

    def __init__(
        self,
        file_name: str,
        chunk_size=None,
        file_downloaded_cb=None,
        **cb_kwargs,
    ):
        """Constructor of FileDownloadable.

        Args:
            file_name: name of the file to be downloaded.
            chunk_size: size of each chunk
            file_downloaded_cb: if specified, the callback to be called when the file is downloaded to a site.
            cb_kwargs: kwargs passed to the CB.

        Notes: The file_downloaded_cb will be called as follows:

            file_downloaded_cb(to_site, status, file_name, **cb_kwargs)

        where: to_site is the name of the site that the file is just downloaded to;
        status is a value of DownloadStatus as defined in nvflare.fuel.f3.streaming.download_service;
        file_name is the name of the file downloaded.

        The file_downloaded_cb is also called after it's downloaded to all sites. In this case, the value of
        "to_site" is empty, and the value of "status" is also empty.

        """
        super().__init__(file_name)
        self.name = file_name

        if not (os.path.isfile(file_name) and os.path.exists(file_name)):
            raise ValueError(f"file {file_name} does not exist or is not a valid file")

        self.size = os.path.getsize(file_name)

        if not chunk_size:
            chunk_size = DEFAULT_CHUNK_SIZE

        check_positive_int("chunk_size", chunk_size)
        if file_downloaded_cb:
            check_callable("file_downloaded_cb", file_downloaded_cb)
        self.chunk_size = chunk_size
        self.file_downloaded_cb = file_downloaded_cb
        self.cb_kwargs = cb_kwargs
        self.logger = get_obj_logger(self)

    def produce(self, state: dict, requester: str) -> Tuple[str, Any, dict]:
        received_bytes = 0
        if state:
            received_bytes = state.get(_StateKey.RECEIVED_BYTES, 0)

        if not isinstance(received_bytes, int) or received_bytes < 0:
            self.logger.error(f"bad {_StateKey.RECEIVED_BYTES} {received_bytes} from {requester}")
            return ProduceRC.ERROR, None, {}

        if received_bytes >= self.size:
            # already done
            return ProduceRC.EOF, None, {}

        num_bytes_to_send = min(self.chunk_size, self.size - received_bytes)
        with open(self.name, "rb") as f:
            f.seek(received_bytes)
            chunk = f.read(num_bytes_to_send)

        self.logger.debug(f"{received_bytes=}; sending {len(chunk)} bytes")
        return ProduceRC.OK, chunk, {_StateKey.RECEIVED_BYTES: received_bytes + len(chunk)}

    def downloaded_to_one(self, to_site: str, status: str):
        if self.file_downloaded_cb:
            self.file_downloaded_cb(to_site, status, self.name, **self.cb_kwargs)

    def downloaded_to_all(self):
        if self.file_downloaded_cb:
            self.file_downloaded_cb("", "", self.name, **self.cb_kwargs)


def add_file(
    downloader: ObjectDownloader,
    file_name: str,
    chunk_size=None,
    ref_id=None,
    file_downloaded_cb=None,
    **cb_kwargs,
) -> str:
    """Add a file to be downloaded to the specified downloader.

    Args:
        downloader: the downloader to add to.
        file_name: name of the file to be downloaded
        chunk_size: chunk size in bytes
        ref_id: ref id to be used, if provided
        file_downloaded_cb: CB to be called when the file is done downloading
        **cb_kwargs: args to be passed to the CB

    Returns: reference id for the file.

    The file_downloaded_cb must follow this signature:

        cb(to_site: str, status: str, file_name: str, **cb_kwargs)

    """
    obj = FileDownloadable(file_name, chunk_size=chunk_size, file_downloaded_cb=file_downloaded_cb, **cb_kwargs)
    return downloader.add_object(
        obj=obj,
        ref_id=ref_id,
    )


def download_file(
    from_fqcn: str,
    ref_id: str,
    per_request_timeout: float,
    cell: Cell,
    location: str = None,
    secure=False,
    optional=False,
    abort_signal=None,
) -> Tuple[str, Optional[str]]:
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
