# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import os
import tempfile
import uuid
from typing import Any, Dict, Tuple

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.stream_shareable import (
    StreamShareableGenerator,
    StreamShareableProcessor,
    StreamShareableProcessorFactory,
)
from nvflare.fuel.utils.obj_utils import get_logger
from nvflare.fuel.utils.validation_utils import check_positive_int, check_positive_number

_PREFIX = "FileStreamer."
_KEY_FILE_NAME = _PREFIX + "file_name"
_KEY_FILE_LOCATION = _PREFIX + "file_location"
_KEY_DATA_SIZE = _PREFIX + "size"
_KEY_DATA = _PREFIX + "data"
_KEY_EOF = _PREFIX + "eof"


class _ChunkProcessor(StreamShareableProcessor):
    def __init__(self, stream_meta: dict, dest_dir):
        file_name = stream_meta.get(_KEY_FILE_NAME)
        self.logger = get_logger(self)
        self.file_name = file_name
        self.dest_dir = dest_dir
        file_path = os.path.join(dest_dir, str(uuid.uuid4()))
        self.file = open(file_path, "wb")
        stream_meta[_KEY_FILE_LOCATION] = file_path

    def process(
        self,
        shareable: Shareable,
        stream_meta: dict,
        fl_ctx: FLContext,
    ) -> Tuple[bool, Shareable]:
        data = shareable.get(_KEY_DATA)
        if data:
            self.file.write(data)

        eof = shareable.get(_KEY_EOF)
        if eof:
            # stop streaming
            return False, make_reply(ReturnCode.OK)
        else:
            # continue streaming
            return True, make_reply(ReturnCode.OK)

    def finalize(self, stream_meta: dict, fl_ctx: FLContext):
        if self.file:
            file_location = stream_meta.get(_KEY_FILE_LOCATION)
            self.file.close()
            self.logger.info(f"closed file {file_location}")


class _ChunkProcessorFactory(StreamShareableProcessorFactory):
    def __init__(self, dest_dir: str):
        self.dest_dir = dest_dir

    def get_processor(self, stream_meta: dict, fl_ctx: FLContext) -> StreamShareableProcessor:
        return _ChunkProcessor(stream_meta, self.dest_dir)


class _ChunkGenerator(StreamShareableGenerator):
    def __init__(self, file, chunk_size, timeout):
        self.file = file
        self.chunk_size = chunk_size
        self.timeout = timeout
        self.eof = False
        self.logger = get_logger(self)

    def get_next(
        self,
        stream_meta: dict,
        fl_ctx: FLContext,
    ) -> Tuple[Shareable, float]:
        chunk = self.file.read(self.chunk_size)
        size = 0
        if chunk:
            size = len(chunk)

        if not chunk or len(chunk) < self.chunk_size:
            self.eof = True

        self.logger.debug(f"sending chunk {size=}")
        result = Shareable()
        result[_KEY_DATA] = chunk
        result[_KEY_DATA_SIZE] = size
        result[_KEY_EOF] = self.eof
        return result, self.timeout

    def process_replies(
        self,
        replies: Dict[str, Shareable],
        stream_meta: dict,
        fl_ctx: FLContext,
    ) -> Any:
        has_error = False
        for target, reply in replies.items():
            rc = reply.get_return_code(ReturnCode.OK)
            if rc != ReturnCode.OK:
                self.logger.error(f"error from target {target}: {rc}")
                has_error = True

        if has_error:
            # done - failed
            return False
        elif self.eof:
            # done - succeeded
            return True
        else:
            # not done yet - continue streaming
            return None


class FileStreamer:
    @staticmethod
    def register_stream_processing(
        fl_ctx: FLContext,
        channel: str,
        topic: str,
        dest_dir: str = None,
        stream_done_cb=None,
        **cb_kwargs,
    ):
        """Register for stream processing on the receiving side.

        Args:
            fl_ctx: the FLContext object
            channel: the app channel
            topic: the app topic
            dest_dir: the destination dir for received file. If not specified, system temp dir is used
            stream_done_cb: if specified, the callback to be called when the file is completely received
            **cb_kwargs: the kwargs for the stream_done_cb

        Returns: None

        Notes: the stream_done_cb must follow stream_done_cb_signature as defined in apis.stream_shareable.py.

        """
        if not dest_dir:
            dest_dir = tempfile.gettempdir()

        if not os.path.isdir(dest_dir):
            raise ValueError(f"dest_dir '{dest_dir}' is not a valid dir")

        engine = fl_ctx.get_engine()
        engine.register_stream_processing(
            channel=channel,
            topic=topic,
            factory=_ChunkProcessorFactory(dest_dir),
            stream_done_cb=stream_done_cb,
            **cb_kwargs,
        )

    @staticmethod
    def stream_file(
        channel: str,
        topic: str,
        stream_meta: dict,
        targets,
        file_name: str,
        fl_ctx: FLContext,
        chunk_size=None,
        chunk_timeout=None,
    ) -> bool:
        """Stream a file to one or more targets.

        Args:
            channel: the app channel
            topic: the app topic
            stream_meta: metadata of the stream
            targets: targets that the file will be sent to
            file_name: full path to the file to be streamed
            fl_ctx: a FLContext object
            chunk_size: size of each chunk to be streamed. If not specified, default to 1M bytes.
            chunk_timeout: timeout for each chunk of data sent to targets.

        Returns: whether the streaming completed successfully

        Notes: this is a blocking call - only returns after the streaming is done.
        """
        if not os.path.isfile(file_name):
            raise ValueError(f"file {file_name} is not a valid file")

        if not chunk_size:
            chunk_size = 1024 * 1024
        check_positive_int("chunk_size", chunk_size)

        if not chunk_timeout:
            chunk_timeout = 5.0
        check_positive_number("chunk_timeout", chunk_timeout)

        with open(file_name, "rb") as file:
            generator = _ChunkGenerator(file, chunk_size, chunk_timeout)
            engine = fl_ctx.get_engine()
            stream_meta[_KEY_FILE_NAME] = os.path.basename(file_name)
            return engine.stream_shareables(
                channel=channel,
                topic=topic,
                stream_meta=stream_meta,
                targets=targets,
                generator=generator,
                fl_ctx=fl_ctx,
            )

    @staticmethod
    def get_file_name(stream_meta: dict):
        """Get the file base name property from stream metadata.
        This method is intended to be used by the stream_done_cb() function of the receiving side.

        Args:
            stream_meta: the stream metadata

        Returns: file base name

        """
        return stream_meta.get(_KEY_FILE_NAME)

    @staticmethod
    def get_file_location(stream_meta: dict):
        """Get the file location property from stream metadata.
        This method is intended to be used by the stream_done_cb() function of the receiving side.

        Args:
            stream_meta: the stream metadata

        Returns: location (full file path) of the received file

        """
        return stream_meta.get(_KEY_FILE_LOCATION)
