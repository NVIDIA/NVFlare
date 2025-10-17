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

from collections import defaultdict

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.streaming import StreamableEngine, StreamContext
from nvflare.client.config import ExchangeFormat
from nvflare.fuel.utils.log_utils import get_obj_logger

from .producer import TensorProducer
from .types import TENSORS_CHANNEL, TensorsMap
from .utils import get_dxo_from_ctx, get_targets_for_ctx_and_prop_key, get_topic_for_ctx_prop_key


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
        # key: task_id, value: tensors to send to the peer
        self.task_params: dict[str, TensorsMap] = defaultdict(dict)
        self.logger = get_obj_logger(self)

    def store_tensors(self, fl_ctx: FLContext):
        """ "Parse tensors from the FLContext and store them for sending.

        Args:
            fl_ctx (FLContext): The FLContext for the current operation.
        """
        peer_name = fl_ctx.get_peer_context().get_identity_name()
        task_id = fl_ctx.get_prop(FLContextKey.TASK_ID, None)
        if not task_id:
            raise ValueError("No task_id found in FLContext.")

        try:
            dxo = get_dxo_from_ctx(fl_ctx, self.ctx_prop_key, self.tasks)
        except ValueError as exc:
            self.logger.warning(f"{exc} Nothing to send.")
            return False

        params = dxo.data
        self.task_params[task_id] = params
        self.logger.info(f"Stored reference to params to be sent to peer '{peer_name}'. Task ID: '{task_id}'.")
        del params

    def send(
        self,
        fl_ctx: FLContext,
        entry_timeout: float,
    ) -> None:
        """Send tensors to the peer.

        Args:
            fl_ctx (FLContext): The FLContext for the current operation.
            entry_timeout (float): Timeout for each tensor entry transfer.
        """
        peer_name = fl_ctx.get_peer_context().get_identity_name()
        task_id = fl_ctx.get_prop(FLContextKey.TASK_ID, None)
        if not task_id:
            raise ValueError("No task_id found in FLContext.")
        targets = get_targets_for_ctx_and_prop_key(fl_ctx, self.ctx_prop_key)

        # Important: pop the tensors to release memory after sending
        # Each task_id is unique per client, so we only send once per task_id
        params = self.task_params.pop(task_id, None)
        if not params:
            raise ValueError(f"No tensors stored for peer '{peer_name}'. Task ID: '{task_id}'.")

        producer = TensorProducer(params, task_id, entry_timeout)
        msg = f"Starting to send tensors to peer '{peer_name}'."
        msg += f" Task ID: '{task_id}'."
        self.logger.info(msg)
        self._send_tensors(targets, producer, fl_ctx)

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
