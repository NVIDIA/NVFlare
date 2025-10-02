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

from nvflare.apis.dxo import DataKind
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.streaming import StreamableEngine
from nvflare.fuel.utils.log_utils import get_obj_logger

from .consumer import TorchTensorsConsumerFactory
from .types import SAFE_TENSORS_PROP_KEY, TENSORS_CHANNEL, TensorsMap
from .utils import get_topic_for_ctx_prop_key


class TensorReceiver:
    """A component to receive tensors from clients using NVFlare's streaming capabilities."""

    def __init__(
        self,
        engine: StreamableEngine,
        ctx_prop_key: FLContextKey,
        format: str,
        channel: str = TENSORS_CHANNEL,
    ):
        """Initialize the TensorReceiver.

        Args:
            engine (StreamableEngine): The streamable engine to use for streaming.
            ctx_prop_key (FLContextKey): The context property key to receive tensors for.
            format (str): The format of the tensors to receive. Currently only "torch" is
                supported.
            channel (str): The channel to use for streaming. Default is TENSORS_CHANNEL.
        """
        super().__init__()
        self.engine = engine
        self.ctx_prop_key = ctx_prop_key
        self.format = format
        self.channel = channel
        self.tensors: dict[str, TensorsMap] = {}
        self.logger = get_obj_logger(self)
        self._register()

    def _register(self):
        """Register the consumer factory with the engine."""
        topic = get_topic_for_ctx_prop_key(self.ctx_prop_key)
        self.engine.register_stream_processing(
            channel=self.channel,
            topic=topic,
            factory=TorchTensorsConsumerFactory(),
            stream_done_cb=self._save_tensors_cb,
        )
        self.logger.info(
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
        if not success:
            self.logger.error(f"Failed to receive tensors from peer {peer_name}.")
            return

        tensors = fl_ctx.get_custom_prop(SAFE_TENSORS_PROP_KEY)
        if not tensors:
            self.logger.error(f"No tensors found from peer {peer_name}.")
            return

        self.tensors[peer_name] = tensors
        self.logger.info(f"Storing tensors received from peer {peer_name}.")

    def set_ctx_with_tensors(self, fl_ctx: FLContext):
        """Update the context with the received tensors.

        Args:
            fl_ctx (FLContext): The FLContext for the current operation.
        """
        peer_name = fl_ctx.get_peer_context().get_identity_name()
        tensors = self.tensors.pop(peer_name, None)
        if not tensors:
            msg = f"No tensors found for peer {peer_name} to set in FLContext."
            self.logger.error(msg)
            raise RuntimeError(msg)

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

        total_values = len(dxo["data"])

        if total_values > 0:
            self.logger.error(
                f"Peer '{fl_ctx.get_identity_name()}': received DXO with task data from peer"
                f"'{peer_name}' and {total_values} tensors.",
            )

        if self.format == "torch":
            dxo["data"] = tensors
        else:
            dxo["data"] = {k: v.numpy() for k, v in tensors.items()}

        s["DXO"] = dxo
        fl_ctx.set_prop(self.ctx_prop_key, s, private=True, sticky=False)
        self.logger.info(
            f"Peer '{fl_ctx.get_identity_name()}': updated task data with "
            f"{len(tensors.keys())} tensors received from peer '{peer_name}'.",
        )
