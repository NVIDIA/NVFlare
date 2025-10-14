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
from .types import TENSORS_CHANNEL
from .utils import get_dxo_from_ctx, get_targets_for_ctx_and_prop_key, get_tensors_from_dxo, get_topic_for_ctx_prop_key


class TensorSender:
    """Handles sending tensors between server and clients."""

    def __init__(
        self,
        engine: StreamableEngine,
        ctx_prop_key: FLContextKey,
        format: ExchangeFormat,
        tasks: list[str],
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
        self.tasks = tasks
        self.channel = channel
        self.logger = get_obj_logger(self)

    def send(
        self,
        fl_ctx: FLContext,
        entry_timeout: float,
    ) -> bool:
        """Send tensors to the peer.

        Args:
            fl_ctx (FLContext): The FLContext for the current operation.
        """
        peer_name = fl_ctx.get_peer_context().get_identity_name()
        task_id = fl_ctx.get_prop(FLContextKey.TASK_ID, None)
        if not task_id:
            raise ValueError("No task_id found in FLContext.")
        targets = get_targets_for_ctx_and_prop_key(fl_ctx, self.ctx_prop_key)

        try:
            dxo = get_dxo_from_ctx(fl_ctx, self.ctx_prop_key, self.tasks)
        except ValueError as exc:
            self.logger.warning(f"{exc} Nothing to send.")
            return False

        root_keys = []
        for key, value in dxo.data.items():
            # auto-detect tensor stored on root keys
            if not isinstance(value, dict) and "" not in root_keys:
                root_keys.append("")
            elif isinstance(value, dict) and key not in root_keys:
                root_keys.append(key)

        for key in root_keys:
            tensors = get_tensors_from_dxo(dxo, key, self.format)
            producer = TensorProducer(tensors, task_id, entry_timeout, root_key=key)
            msg = f"Starting to send tensors to peer '{peer_name}'."
            if key:
                msg += f" Root Key: '{key}'."
            msg += f" Task ID: '{task_id}'."
            self.logger.info(msg)
            self._send_tensors(targets, producer, fl_ctx)
            # Explicitly delete tensors after streaming to free memory
            del tensors

        return True

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
