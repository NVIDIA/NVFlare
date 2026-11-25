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

from collections import OrderedDict
from typing import Any, Dict, List, Tuple

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.streaming import ConsumerFactory, ObjectConsumer, ObjectProducer, StreamableEngine, StreamContext
from nvflare.app_common.streamers.streamer_base import StreamerBase
from nvflare.fuel.utils.class_loader import get_class_name
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.fuel.utils.validation_utils import check_positive_number

_PREFIX = "ContainerStreamer."

# Keys for StreamCtx
_CTX_TYPE = _PREFIX + "type"
_CTX_SIZE = _PREFIX + "size"
_CTX_RESULT = _PREFIX + "result"

# Keys for Shareable
_KEY_ENTRY = _PREFIX + "entry"
_KEY_LAST = _PREFIX + "last"

# Use the existing wire names to select only container constructors held locally.
_CONTAINER_TYPES = {get_class_name(cls): cls for cls in (dict, list, set, OrderedDict)}


def _get_container_type(container: Any) -> str:
    container_class = type(container)
    container_type = get_class_name(container_class)
    if _CONTAINER_TYPES.get(container_type) is not container_class:
        supported_types = ", ".join(_CONTAINER_TYPES)
        raise ValueError(
            f"Unsupported container type for streaming: {container_type}. Supported types: {supported_types}. "
            "To stream the entries of a dictionary subclass, pass dict(value) explicitly."
        )
    return container_type


class _EntryConsumer(ObjectConsumer):
    def __init__(self, stream_ctx: StreamContext, container_class):
        self.logger = get_obj_logger(self)
        self.container = container_class()
        self.size = stream_ctx.get(_CTX_SIZE)

    def consume(
        self,
        shareable: Shareable,
        stream_ctx: StreamContext,
        fl_ctx: FLContext,
    ) -> Tuple[bool, Shareable]:

        entry = shareable.get(_KEY_ENTRY)
        try:
            if isinstance(self.container, dict):
                key, value = entry
                self.container[key] = value
            elif isinstance(self.container, set):
                self.container.add(entry)
            else:
                self.container.append(entry)
        except Exception:
            error = f"Unable to add entry ({type(entry)} to container ({type(self.container)}"
            self.logger.error(error)
            raise ValueError(error)

        last = shareable.get(_KEY_LAST)
        if last:
            # Check if all entries are added
            if self.size != len(self.container):
                err = f"Container size {len(self.container)} does not match expected size {self.size}"
                self.logger.error(err)
                raise ValueError(err)
            else:
                stream_ctx[_CTX_RESULT] = self.container
                return False, make_reply(ReturnCode.OK)
        else:
            # continue streaming
            return True, make_reply(ReturnCode.OK)

    def finalize(self, stream_ctx: StreamContext, fl_ctx: FLContext):
        self.logger.debug(f"Container streaming is done for container type {type(self.container)}")


class _EntryConsumerFactory(ConsumerFactory):
    def __init__(self, on_unsupported_type=None):
        self.on_unsupported_type = on_unsupported_type

    def get_consumer(self, stream_ctx: StreamContext, fl_ctx: FLContext) -> ObjectConsumer:
        container_type = stream_ctx.get(_CTX_TYPE)
        if type(container_type) is not str or container_type not in _CONTAINER_TYPES:
            if self.on_unsupported_type:
                self.on_unsupported_type(stream_ctx, fl_ctx)
            raise ValueError("Unsupported container type for streaming")
        return _EntryConsumer(stream_ctx, _CONTAINER_TYPES[container_type])


class _EntryProducer(ObjectProducer):
    def __init__(self, container, entry_timeout):
        self.logger = get_obj_logger(self)
        if not container:
            error = "Can't stream empty container"
            self.logger.error(error)
            raise ValueError(error)

        self.container = container
        if isinstance(container, dict):
            self.iterator = iter(container.items())
        else:
            self.iterator = iter(container)
        self.size = len(container)
        self.count = 0
        self.last = False
        self.entry_timeout = entry_timeout

    def produce(
        self,
        stream_ctx: StreamContext,
        fl_ctx: FLContext,
    ) -> Tuple[Shareable, float]:

        try:
            entry = next(self.iterator)
            self.count += 1
            self.last = self.count >= self.size
        except StopIteration:
            self.logger.error(f"Producer called too many times {self.count}/{self.size}")
            self.last = True
            return None, 0.0

        result = Shareable()
        result[_KEY_ENTRY] = entry
        result[_KEY_LAST] = self.last
        return result, self.entry_timeout

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
        elif self.last:
            # done - succeeded
            return True
        else:
            # not done yet - continue streaming
            return None


class ContainerStreamer(StreamerBase):
    @staticmethod
    def register_stream_processing(
        fl_ctx: FLContext,
        channel: str,
        topic: str,
        stream_done_cb=None,
        *,
        on_unsupported_type=None,
        **cb_kwargs,
    ):
        """Register for stream processing on the receiving side.

        Args:
            fl_ctx: the FLContext object
            channel: the app channel
            topic: the app topic
            stream_done_cb: if specified, the callback to be called when the file is completely received
            on_unsupported_type: callback receiving the stream context and FLContext when type validation fails
            **cb_kwargs: the kwargs for the stream_done_cb

        Returns: None

        Notes: the stream_done_cb must follow stream_done_cb_signature as defined in apis.streaming.

        """

        engine = fl_ctx.get_engine()
        if not isinstance(engine, StreamableEngine):
            raise RuntimeError(f"engine must be StreamableEngine but got {type(engine)}")

        engine.register_stream_processing(
            channel=channel,
            topic=topic,
            factory=_EntryConsumerFactory(on_unsupported_type),
            stream_done_cb=stream_done_cb,
            **cb_kwargs,
        )

    @staticmethod
    def stream_container(
        channel: str,
        topic: str,
        stream_ctx: StreamContext,
        targets: List[str],
        container: Any,
        fl_ctx: FLContext,
        entry_timeout=None,
        optional=False,
        secure=False,
    ) -> bool:
        """Stream a dict, list, set, or collections.OrderedDict to one or more targets.

        Args:
            channel: the app channel
            topic: the app topic
            stream_ctx: context data of the stream
            targets: targets that the file will be sent to
            container: container to be streamed
            fl_ctx: a FLContext object
            entry_timeout: timeout for each entry sent to targets.
            optional: whether the file is optional
            secure: whether P2P security is required

        Returns: whether the streaming completed successfully

        Notes: this is a blocking call - only returns after the streaming is done.
        """
        if not entry_timeout:
            entry_timeout = 60.0
        check_positive_number("entry_timeout", entry_timeout)

        container_type = _get_container_type(container)
        producer = _EntryProducer(container, entry_timeout)
        engine = fl_ctx.get_engine()

        if not isinstance(engine, StreamableEngine):
            raise RuntimeError(f"engine must be StreamableEngine but got {type(engine)}")

        if not stream_ctx:
            stream_ctx = {}

        stream_ctx[_CTX_TYPE] = container_type
        stream_ctx[_CTX_SIZE] = len(container)

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
    def get_result(stream_ctx: StreamContext) -> Any:
        """Get the received container
        This method is intended to be used by the stream_done_cb() function of the receiving side.

        Args:
            stream_ctx: the stream context

        Returns: The received container

        """
        return stream_ctx.get(_CTX_RESULT)
