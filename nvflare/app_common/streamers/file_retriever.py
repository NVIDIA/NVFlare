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
from typing import Any

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable
from nvflare.apis.streaming import StreamContext

from .file_streamer import FileStreamer
from .object_retriever import ObjectRetriever


class FileRetriever(ObjectRetriever):
    def __init__(
        self,
        source_dir: str,
        topic: str = None,
        stream_msg_optional=False,
        stream_msg_secure=False,
        dest_dir=None,
        chunk_size=None,
        chunk_timeout=None,
    ):
        ObjectRetriever.__init__(self, topic)
        FLComponent.__init__(self)
        self.source_dir = source_dir
        self.dest_dir = dest_dir
        self.stream_msg_optional = stream_msg_optional
        self.stream_msg_secure = stream_msg_secure
        self.chunk_size = chunk_size
        self.chunk_timeout = chunk_timeout

    def register_stream_processing(
        self,
        channel: str,
        topic: str,
        fl_ctx: FLContext,
        stream_done_cb,
        **cb_kwargs,
    ):
        """Called on the stream sending side.

        Args:
            channel:
            topic:
            fl_ctx:
            stream_done_cb:
            **cb_kwargs:

        Returns:

        """
        FileStreamer.register_stream_processing(
            channel=channel,
            topic=topic,
            fl_ctx=fl_ctx,
            dest_dir=self.dest_dir,
            stream_done_cb=stream_done_cb,
            **cb_kwargs,
        )

    def validate_request(self, request: Shareable, fl_ctx: FLContext) -> (str, Any):
        # the property name "file_name" in the request must match "file_name" attribute of the retrieve_file method
        file_name = request.get("file_name")
        if not file_name:
            self.log_error(fl_ctx, "bad request: missing file_name")
            return ReturnCode.BAD_REQUEST_DATA, None

        file_path = os.path.join(self.source_dir, file_name)
        if not os.path.isfile(file_path):
            self.log_error(fl_ctx, f"bad request: requested file {file_path} is invalid")
            return ReturnCode.BAD_REQUEST_DATA, None

        return ReturnCode.OK, file_path

    def retrieve_file(self, from_site: str, fl_ctx: FLContext, timeout: float, file_name: str) -> (str, str):
        """Retrieve a file from the specified site.
        This method is to be called by the app.

        Args:
            from_site: the site that has the file to be retrieved
            fl_ctx: FLContext object
            timeout: how long to wait for the file
            file_name: name of the file to be retrieved

        Returns: a tuple of (ReturnCode, location of the file received)

        """
        return self.retrieve(from_site=from_site, fl_ctx=fl_ctx, timeout=timeout, file_name=file_name)

    def do_stream(
        self, target: str, request: Shareable, fl_ctx: FLContext, stream_ctx: StreamContext, validated_data: Any
    ):
        """Stream the file to the peer.
        Called on the stream sending side.

        Args:
            target: the receiving site
            request: data to be sent
            fl_ctx: FLContext object
            stream_ctx: the stream context
            validated_data: the file full path returned from the validate_request method

        Returns:

        """
        file_path = validated_data
        FileStreamer.stream_file(
            targets=[target],
            stream_ctx=stream_ctx,
            channel=self.stream_channel,
            topic=self.topic,
            file_name=file_path,
            fl_ctx=fl_ctx,
            optional=self.stream_msg_optional,
            secure=self.stream_msg_secure,
        )

    def get_result(self, stream_ctx: StreamContext) -> (str, Any):
        """Called on the stream receiving side.
        Get the final result of the streaming.
        The result is the location of the received file.

        Args:
            stream_ctx: the StreamContext

        Returns:

        """
        self.logger.info(f"getting result from stream ctx: {stream_ctx}")
        return FileStreamer.get_rc(stream_ctx), FileStreamer.get_file_location(stream_ctx)
