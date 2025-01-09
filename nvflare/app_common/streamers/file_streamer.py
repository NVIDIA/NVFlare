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
from typing import Any, Dict, List, Tuple

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.streaming import ConsumerFactory, ObjectConsumer, ObjectProducer, StreamableEngine, StreamContext
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.fuel.utils.validation_utils import check_positive_int, check_positive_number

from .streamer_base import StreamerBase

_PREFIX = "FileStreamer."
_KEY_FILE_NAME = _PREFIX + "file_name"
_KEY_FILE_LOCATION = _PREFIX + "file_location"
_KEY_FILE_SIZE = _PREFIX + "file_size"
_KEY_DATA_SIZE = _PREFIX + "size"
_KEY_DATA = _PREFIX + "data"
_KEY_EOF = _PREFIX + "eof"


class _ChunkConsumer(ObjectConsumer):
    def __init__(self, stream_ctx: StreamContext, dest_dir):
        file_name = stream_ctx.get(_KEY_FILE_NAME)
        self.logger = get_obj_logger(self)
        self.file_name = file_name
        self.dest_dir = dest_dir
        self.file_size = stream_ctx.get(_KEY_FILE_SIZE)
        self.received_size = 0
        file_path = os.path.join(dest_dir, str(uuid.uuid4()))
        self.file = open(file_path, "wb")
        stream_ctx[_KEY_FILE_LOCATION] = file_path

    def consume(
        self,
        shareable: Shareable,
        stream_ctx: StreamContext,
        fl_ctx: FLContext,
    ) -> Tuple[bool, Shareable]:
        data = shareable.get(_KEY_DATA)
        data_size = shareable.get(_KEY_DATA_SIZE)

        if data:
            if data_size != len(data):
                err = f"received data {len(data)} does not match expected size {data_size}"
                self.logger.error(err)
                raise ValueError(err)
            self.received_size += data_size
            self.file.write(data)
        else:
            if data_size != 0:
                err = f"no data received but expected size is {data_size}"
                self.logger.error(err)
                raise ValueError(err)

        eof = shareable.get(_KEY_EOF)
        if eof:
            # stop streaming
            if self.received_size != self.file_size:
                err = f"received size {self.received_size} does not match expected file size {self.file_size}"
                self.logger.error(err)
                raise ValueError(err)
            else:
                return False, make_reply(ReturnCode.OK)
        else:
            # continue streaming
            return True, make_reply(ReturnCode.OK)

    def finalize(self, stream_ctx: StreamContext, fl_ctx: FLContext):
        if self.file:
            file_location = stream_ctx.get(_KEY_FILE_LOCATION)
            self.file.close()
            self.logger.info(f"closed file {file_location}")


class _ChunkConsumerFactory(ConsumerFactory):
    def __init__(self, dest_dir: str):
        self.dest_dir = dest_dir

    def get_consumer(self, stream_ctx: StreamContext, fl_ctx: FLContext) -> ObjectConsumer:
        return _ChunkConsumer(stream_ctx, self.dest_dir)


class _ChunkProducer(ObjectProducer):
    def __init__(self, file, chunk_size, timeout):
        self.file = file
        self.chunk_size = chunk_size
        self.timeout = timeout
        self.eof = False
        self.logger = get_obj_logger(self)

    def produce(
        self,
        stream_ctx: StreamContext,
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
        stream_ctx: StreamContext,
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


class FileStreamer(StreamerBase):
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

        Notes: the stream_done_cb must follow stream_done_cb_signature as defined in apis.streaming.

        """
        if not dest_dir:
            dest_dir = tempfile.gettempdir()

        if not os.path.isdir(dest_dir):
            raise ValueError(f"dest_dir '{dest_dir}' is not a valid dir")

        engine = fl_ctx.get_engine()
        if not isinstance(engine, StreamableEngine):
            raise RuntimeError(f"engine must be StreamableEngine but got {type(engine)}")

        engine.register_stream_processing(
            channel=channel,
            topic=topic,
            factory=_ChunkConsumerFactory(dest_dir),
            stream_done_cb=stream_done_cb,
            **cb_kwargs,
        )

    @staticmethod
    def stream_file(
        channel: str,
        topic: str,
        stream_ctx: StreamContext,
        targets: List[str],
        file_name: str,
        fl_ctx: FLContext,
        chunk_size=None,
        chunk_timeout=None,
        optional=False,
        secure=False,
    ) -> bool:
        """Stream a file to one or more targets.

        Args:
            channel: the app channel
            topic: the app topic
            stream_ctx: context data of the stream
            targets: targets that the file will be sent to
            file_name: full path to the file to be streamed
            fl_ctx: a FLContext object
            chunk_size: size of each chunk to be streamed. If not specified, default to 1M bytes.
            chunk_timeout: timeout for each chunk of data sent to targets.
            optional: whether the file is optional
            secure: whether P2P security is required

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

        file_stats = os.stat(file_name)
        file_size = file_stats.st_size
        if not stream_ctx:
            stream_ctx = {}
        stream_ctx[_KEY_FILE_SIZE] = file_size

        with open(file_name, "rb") as file:
            producer = _ChunkProducer(file, chunk_size, chunk_timeout)
            engine = fl_ctx.get_engine()

            if not isinstance(engine, StreamableEngine):
                raise RuntimeError(f"engine must be StreamableEngine but got {type(engine)}")

            stream_ctx[_KEY_FILE_NAME] = os.path.basename(file_name)

            return engine.stream_objects(
                channel=channel,
                topic=topic,
                stream_ctx=stream_ctx,
                targets=targets,
                producer=producer,
                fl_ctx=fl_ctx,
                optional=optional,
                secure=secure,
            )

    @staticmethod
    def get_file_name(stream_ctx: StreamContext):
        """Get the file base name property from stream context.
        This method is intended to be used by the stream_done_cb() function of the receiving side.

        Args:
            stream_ctx: the stream context

        Returns: file base name

        """
        return stream_ctx.get(_KEY_FILE_NAME)

    @staticmethod
    def get_file_location(stream_ctx: StreamContext):
        """Get the file location property from stream context.
        This method is intended to be used by the stream_done_cb() function of the receiving side.

        Args:
            stream_ctx: the stream context

        Returns: location (full file path) of the received file

        """
        return stream_ctx.get(_KEY_FILE_LOCATION)

    @staticmethod
    def get_file_size(stream_ctx: StreamContext):
        """Get the file size property from stream context.
        This method is intended to be used by the stream_done_cb() function of the receiving side.

        Args:
            stream_ctx: the stream context

        Returns: size (in bytes) of the received file

        """
        return stream_ctx.get(_KEY_FILE_SIZE)
