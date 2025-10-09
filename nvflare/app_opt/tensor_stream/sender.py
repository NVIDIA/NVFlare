# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.streaming import StreamableEngine, StreamContext
from nvflare.client.config import ExchangeFormat
from nvflare.fuel.utils.log_utils import get_obj_logger

from .producer import TensorProducer
from .store import TensorStore
from .types import TENSORS_CHANNEL
from .utils import get_topic_for_ctx_prop_key, validate_and_extract_tensors


class TensorSender:
    """Handles sending tensors between server and clients."""

    def __init__(
        self,
        engine: StreamableEngine,
        ctx_prop_key: FLContextKey,
        format: ExchangeFormat,
        channel: str = TENSORS_CHANNEL,
    ):
        """Initialize the TensorSender.

        Args:
            engine (StreamableEngine): The streamable engine to use for streaming.
            ctx_prop_key (FLContextKey): The context property key to send tensors for.
            channel (str): The channel to use for streaming. Default is TENSORS_CHANNEL.
        """
        self.engine = engine
        self.ctx_prop_key = ctx_prop_key
        self.format = format
        self.channel = channel
        self.logger = get_obj_logger(self)

    def send(
        self,
        fl_ctx: FLContext,
        store: TensorStore,
        targets: list[str],
        entry_timeout: float,
    ):
        """Send tensors to the peer.

        Args:
            fl_ctx (FLContext): The FLContext for the current operation.
        """
        tensors_map = store.get()
        peer_name = targets[0]
        for key, tensors in tensors_map.items():
            tensors = validate_and_extract_tensors(tensors, key, self.format)
            producer = TensorProducer(tensors, peer_name, entry_timeout, key)
            msg = f"Starting to send {len(tensors)} tensors to peer '{peer_name}' and task id '{store.task_id}'."
            if key:
                msg += f"With root key '{key}'"
            self.logger.info(msg)
            self._send_tensors(targets, producer, fl_ctx)
            self.logger.info(f"Finished sending tensors to peer '{peer_name}' and task id '{store.task_id}'.")

        store.clear()

    def _send_tensors(self, targets: list[str], producer: TensorProducer, fl_ctx: FLContext):
        """Send tensors to the peer using the StreamableEngine."""
        stream_ctx = StreamContext()
        self.engine.stream_objects(
            channel=self.channel,
            topic=get_topic_for_ctx_prop_key(self.ctx_prop_key),
            stream_ctx=stream_ctx,
            targets=targets,
            producer=producer,
            fl_ctx=fl_ctx,
            optional=False,
            secure=False,
        )
