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
from nvflare.apis.shareable import ReturnCode
from nvflare.apis.streaming import ObjectConsumer, ObjectProducer, StreamContext, StreamContextKey
from nvflare.fuel.utils.log_utils import get_obj_logger

# ---------------------------------------------------------------------------
# Chunk-protocol wire keys
# ---------------------------------------------------------------------------
# Shared by FileStreamer and LogStreamer — same payload format, single namespace.
CHUNK_PREFIX = "Streamer."
KEY_FILE_NAME = CHUNK_PREFIX + "file_name"
KEY_DATA = CHUNK_PREFIX + "data"
KEY_DATA_SIZE = CHUNK_PREFIX + "size"
KEY_EOF = CHUNK_PREFIX + "eof"

# FileStreamer-specific keys
KEY_FILE_LOCATION = "FileStreamer.file_location"
KEY_FILE_SIZE = "FileStreamer.file_size"

# LogStreamer-specific keys
KEY_HEARTBEAT = "LogStreamer.heartbeat"
KEY_STREAM_DONE_CB = "LogStreamer.stream_done_cb"


class BaseChunkProducer(ObjectProducer):
    """Base for chunk-based producers.

    Provides the common ``process_replies`` implementation shared by
    FileStreamer and LogStreamer.  Subclasses must implement ``produce``.
    """

    def __init__(self):
        self.eof = False
        self.logger = get_obj_logger(self)

    def process_replies(self, replies, stream_ctx, fl_ctx):
        has_error = False
        final_result = {}
        for target, reply in replies.items():
            rc = reply.get_return_code(ReturnCode.OK)
            if rc == ReturnCode.OK:
                final_result[target] = reply
            else:
                self.logger.error(f"error from target {target}: {rc}")
                has_error = True

        if has_error:
            return False
        elif self.eof:
            return final_result
        else:
            return None


class BaseChunkConsumer(ObjectConsumer):
    """Base for chunk-based consumers.

    Provides shared data-chunk size validation.  Subclasses must implement
    ``consume``.
    """

    def __init__(self):
        self.logger = get_obj_logger(self)

    def _validate_chunk(self, data, data_size):
        """Validate that *data* and *data_size* are consistent.

        Raises ValueError if the sizes disagree.
        """
        if data is not None:
            if data_size != len(data):
                err = f"received data size {len(data)} does not match expected {data_size}"
                self.logger.error(err)
                raise ValueError(err)
        else:
            if data_size != 0:
                err = f"no data received but expected size is {data_size}"
                self.logger.error(err)
                raise ValueError(err)


class StreamerBase:
    """This is the base class for all future streamers.
    This base class provides methods for accessing common properties in the StreamContext.
    When a streamer class is defined as a subclass of this base, then all such StreamContext accessing methods
    will be inherited.
    """

    @staticmethod
    def get_channel(ctx: StreamContext):
        return ctx.get(StreamContextKey.CHANNEL)

    @staticmethod
    def get_topic(ctx: StreamContext):
        return ctx.get(StreamContextKey.TOPIC)

    @staticmethod
    def get_rc(ctx: StreamContext):
        return ctx.get(StreamContextKey.RC)
