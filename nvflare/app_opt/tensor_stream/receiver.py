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

import time

from nvflare.apis.dxo import DataKind
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.streaming import StreamableEngine
from nvflare.client.config import ExchangeFormat
from nvflare.fuel.utils.log_utils import get_obj_logger

from .consumer import TensorConsumerFactory
from .types import TENSORS_CHANNEL, TensorCustomKeys, TensorsMap
from .utils import get_topic_for_ctx_prop_key, to_numpy_recursive


class TensorReceiver:
    """A component to receive tensors from clients using NVFlare's streaming capabilities."""

    def __init__(
        self,
        engine: StreamableEngine,
        ctx_prop_key: FLContextKey,
        format: ExchangeFormat = ExchangeFormat.PYTORCH,
        channel: str = TENSORS_CHANNEL,
    ):
        """Initialize the TensorReceiver.

        Args:
            engine (StreamableEngine): The streamable engine to use for streaming.
            ctx_prop_key (FLContextKey): The context property key to receive tensors for.
            format (ExchangeFormat): The format of the tensors to receive. Default is ExchangeFormat.PYTORCH.
            channel (str): The channel to use for streaming. Default is TENSORS_CHANNEL.
        """
        super().__init__()
        self.engine = engine
        self.ctx_prop_key = ctx_prop_key
        self.format = format
        self.channel = channel
        # key: peer_name, value: tensors received from the peer
        self.tensors: dict[str, TensorsMap] = {}
        self.logger = get_obj_logger(self)
        self._register()

    def _register(self):
        """Register the consumer factory with the engine."""
        topic = get_topic_for_ctx_prop_key(self.ctx_prop_key)
        self.engine.register_stream_processing(
            channel=self.channel,
            topic=topic,
            factory=TensorConsumerFactory(),
            stream_done_cb=self._save_tensors_cb,
        )
        self.logger.debug(
            f"Registered tensor receiver for context property '{self.ctx_prop_key}' "
            f"on '{self.channel}:{topic}' with format '{self.format}'.",
        )

    def _save_tensors_cb(self, success: bool, fl_ctx: FLContext):
        """Save tensors received from stream. Called when the stream is done.

        Args:
            success (bool): no error happen on the stream consumer
            fl_ctx (FLContext): the FLContext for the current operation
        """
        peer_name = fl_ctx.get_peer_context().get_identity_name()
        task_id = fl_ctx.get_custom_prop(TensorCustomKeys.TASK_ID)
        if not task_id:
            raise ValueError(f"No task_id found from peer {peer_name}.")

        if not success:
            raise ValueError(f"Failed to receive tensors from peer '{peer_name}' and task '{task_id}'.")

        tensors = fl_ctx.get_custom_prop(TensorCustomKeys.SAFE_TENSORS_PROP_KEY)
        if not tensors:
            raise ValueError(f"No tensors found from peer '{peer_name}' and task '{task_id}'.")

        # add or update (when multiple root keys are present) the tensors received from the same task_id
        if task_id not in self.tensors:
            self.tensors[task_id] = tensors
        else:
            self.tensors[task_id].update(tensors)

        # Clean up custom properties to reduce memory usage
        fl_ctx.set_custom_prop(TensorCustomKeys.SAFE_TENSORS_PROP_KEY, None)
        fl_ctx.set_custom_prop(TensorCustomKeys.TASK_ID, None)
        del tensors

    def set_ctx_with_tensors(self, fl_ctx: FLContext):
        """Update the context with the received tensors.

        Args:
            fl_ctx (FLContext): The FLContext for the current operation.
        """
        task_id = fl_ctx.get_prop(FLContextKey.TASK_ID, None)
        if not task_id:
            raise ValueError("No task_id found in FLContext.")

        # there is a race condition where the event is fired before the tensors are set in self.tensors
        # wait for a short period of time to allow the callback to set the tensors
        start_wait = time.time()
        max_wait = 5  # seconds
        while task_id not in self.tensors:
            self.logger.debug(f"Waiting for tensors for task_id '{task_id}'...")
            time.sleep(0.1)  # wait for tensors to be received
            if time.time() - start_wait > max_wait:
                self.logger.error(f"Timeout waiting for tensors for task_id '{task_id}'.")
                return  # exit without setting tensors

        peer_name = fl_ctx.get_peer_context().get_identity_name()
        tensors = self.tensors.pop(task_id, None)
        if not tensors:
            msg = f"No tensors found in FLContext for peer '{peer_name}' and task '{task_id}' to be set."
            self.logger.warning(msg)
            return

        s: Shareable = fl_ctx.get_prop(self.ctx_prop_key)
        if not s:
            msg = f"No shareable found in FLContext for key {self.ctx_prop_key}."
            self.logger.error(msg)
            raise RuntimeError(msg)

        dxo = s.get("DXO")
        if not dxo:
            msg = f"No DXO found in shareable for key {self.ctx_prop_key}."
            self.logger.error(msg)
            raise RuntimeError(msg)

        if dxo["kind"] not in (DataKind.WEIGHTS, DataKind.WEIGHT_DIFF):
            msg = f"Task data kind is not WEIGHTS or WEIGHT_DIFF: {dxo['kind']}"
            self.logger.error(msg)
            raise RuntimeError(msg)

        if len(dxo["data"]) == 0 and not tensors:
            self.logger.error(
                f"Peer '{fl_ctx.get_identity_name()}':received task with empty data, no tensors "
                f"are present for '{peer_name}'. Task ID: '{task_id}'.",
            )
            raise RuntimeError(msg)

        if self.format == ExchangeFormat.PYTORCH:
            dxo["data"] = tensors
        elif self.format == ExchangeFormat.NUMPY:
            dxo["data"] = to_numpy_recursive(tensors)
        else:
            msg = f"Unsupported tensor format: {self.format}"
            self.logger.error(msg)
            raise RuntimeError(msg)

        s["DXO"] = dxo
        fl_ctx.set_prop(self.ctx_prop_key, s, private=True, sticky=False)

        # Explicitly delete local reference to aid garbage collection
        del tensors

        self.logger.info(
            f"Peer '{fl_ctx.get_identity_name()}': updated task data with tensors received from peer "
            f"'{peer_name}'. Task ID: '{task_id}'."
        )
