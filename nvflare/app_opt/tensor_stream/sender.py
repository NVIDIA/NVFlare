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

import torch

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.streaming import StreamableEngine, StreamContext
from nvflare.client.config import ExchangeFormat
from nvflare.fuel.utils.log_utils import get_obj_logger

from .producer import TorchTensorsProducer
from .types import TENSORS_CHANNEL
from .utils import (
    get_targets_for_ctx_and_prop_key,
    get_topic_for_ctx_prop_key,
    to_torch_recursive,
    validate_numpy_dict_params_recursive,
    validate_torch_dict_params_recursive,
)


class TensorSender:
    """Handles sending tensors between server and clients."""

    def __init__(
        self,
        engine: StreamableEngine,
        ctx_prop_key: FLContextKey,
        root_keys: list[str],
        format: ExchangeFormat = ExchangeFormat.PYTORCH,
        channel: str = TENSORS_CHANNEL,
    ):
        """Initialize the TensorSender.

        Args:
            engine (StreamableEngine): The streamable engine to use for streaming.
            ctx_prop_key (FLContextKey): The context property key to send tensors for.
            root_keys (list[str]): The root keys in the DXO data dict to send tensors for.
            channel (str): The channel to use for streaming. Default is TENSORS_CHANNEL.
        """
        self.engine = engine
        self.ctx_prop_key = ctx_prop_key
        self.root_keys = root_keys
        self.format = format
        self.channel = channel
        self.logger = get_obj_logger(self)

    def send(
        self,
        fl_ctx: FLContext,
        entry_timeout: float,
    ):
        """Send tensors to the peer.

        Args:
            fl_ctx (FLContext): The FLContext for the current operation.
        """
        peer_name = fl_ctx.get_peer_context().get_identity_name()
        targets = get_targets_for_ctx_and_prop_key(fl_ctx, self.ctx_prop_key)

        self.logger.debug(f"Sending tensors to peer: {peer_name}")

        data = self._get_dxo_from_ctx(fl_ctx)
        for key in self.root_keys:
            tensors = self._get_tensors_from_dxo(data, key=key)
            producer = TorchTensorsProducer(tensors, entry_timeout, root_key=key)
            self._send_tensors(targets, producer, fl_ctx)

    def _send_tensors(self, targets: list[str], producer: TorchTensorsProducer, fl_ctx: FLContext):
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

    def _get_dxo_from_ctx(self, fl_ctx: FLContext) -> DXO:
        """Extract model parameters from the FLContext based on the provided property key.

        Args:
            fl_ctx (FLContext): The FLContext containing the data.

        Returns:
            dict[str, torch.Tensor]: A dictionary of data extracted from the FLContext.
        """
        data: Shareable = fl_ctx.get_prop(self.ctx_prop_key)
        if data is None:
            self.logger.warning("No task data found in FLContext")
            return None

        dxo = from_shareable(data)
        if dxo.data_kind not in (DataKind.WEIGHTS, DataKind.WEIGHT_DIFF):
            return None

        return dxo

    def _get_tensors_from_dxo(self, dxo: DXO, key: str) -> dict[str, torch.Tensor]:
        """Extract tensors from the FLContext based on the provided property key.

        Args:
            dxo (DXO): The DXO containing the data.
            key (str): The key to extract tensors for.

        Returns:
            dict[str, torch.Tensor]: A dictionary of tensors extracted from the FLContext.
        Raises:
            TypeError: If the tensor data type is unsupported.
            ValueError: If no tensors are found in the context shareable.
        """
        if not key:
            data = dxo.data
        else:
            data = dxo.data.get(key)

        if not data:
            msg = "No tensor data found on the context shareable."
            if key:
                msg += f" Key='{key}'"
            self.logger.error(msg)
            raise ValueError(msg)

        if not isinstance(data, dict):
            self.logger.error(f"Expected tensor data to be a dict, but got {type(data)}")
            raise ValueError(f"Expected tensor data to be a dict, but got {type(data)}")

        if self.format == ExchangeFormat.PYTORCH:
            validate_torch_dict_params_recursive(data)
            tensors = data
        elif self.format == ExchangeFormat.NUMPY:
            validate_numpy_dict_params_recursive(data)
            tensors = to_torch_recursive(data)
        else:
            self.logger.error(f"Unsupported tensor data type: {self.format}")
            raise TypeError(f"Unsupported tensor data type: {self.format}")

        if not tensors:
            self.logger.error(f"No tensors found on context shareable with key {self.ctx_prop_key}")
            raise ValueError(f"No tensors found on context shareable with key {self.ctx_prop_key}")

        return tensors
