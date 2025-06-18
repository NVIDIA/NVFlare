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
from typing import Any, Optional

from nvflare.fuel.f3.cellnet.cell import Cell

from .obj_downloader import Consumer, ObjDownloader, Producer, ProduceRC, download_object

CHUNK_SIZE = 5 * 1024 * 1024


class _StateKey:
    RECEIVED_BYTES = "received_bytes"


class _File:

    def __init__(self, file_name):
        self.name = file_name
        self.size = os.path.getsize(file_name)


class _ChunkProducer(Producer):

    def __init__(self):
        Producer.__init__(self)

    def produce(self, obj, state: dict, requester, logger) -> (str, Any, dict):
        assert isinstance(obj, _File)
        received_bytes = 0
        if state:
            received_bytes = state.get(_StateKey.RECEIVED_BYTES, 0)

        if not isinstance(received_bytes, int) or received_bytes < 0:
            logger.error(f"bad {_StateKey.RECEIVED_BYTES} {received_bytes} from {requester}")
            return ProduceRC.ERROR, None, None

        if received_bytes >= obj.size:
            # already done
            return ProduceRC.EOF, None, None

        num_bytes_to_send = min(CHUNK_SIZE, obj.size - received_bytes)
        with open(obj.name, "rb") as f:
            f.seek(received_bytes)
            chunk = f.read(num_bytes_to_send)

        logger.debug(f"{received_bytes=}; sending {len(chunk)} bytes")
        return ProduceRC.OK, chunk, {_StateKey.RECEIVED_BYTES: received_bytes + len(chunk)}


class FileDownloader:

    @classmethod
    def new_transaction(cls, cell: Cell, timeout: float, timeout_cb, **cb_kwargs):
        return ObjDownloader.new_transaction(
            cell=cell,
            producer=_ChunkProducer(),
            timeout=timeout,
            timeout_cb=timeout_cb,
            **cb_kwargs,
        )

    @classmethod
    def add_file(
        cls,
        transaction_id: str,
        file_name: str,
        file_downloaded_cb=None,
        **cb_kwargs,
    ) -> str:
        obj = _File(file_name)
        return ObjDownloader.add_download_object(
            transaction_id=transaction_id,
            obj=obj,
            obj_downloaded_cb=file_downloaded_cb,
            **cb_kwargs,
        )


class _ChunkConsumer(Consumer):

    def __init__(self, location: str):
        Consumer.__init__(self)
        self.location = location
        self.file_path = os.path.join(location, str(uuid.uuid4()))
        self.file = open(self.file_path, "wb")
        self.total_bytes = 0
        self.error = None

    def consume(self, state, data) -> dict:
        self.file.write(data)
        self.total_bytes += len(data)
        return {_StateKey.RECEIVED_BYTES: self.total_bytes}

    def download_failed(self, reason: str):
        self.error = reason
        self.file.close()

    def download_completed(self):
        self.file.close()


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
