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

from typing import Any, Dict, Tuple

import torch
from safetensors.torch import save as save_tensors

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable
from nvflare.apis.streaming import ObjectProducer, StreamContext
from nvflare.fuel.utils.log_utils import get_obj_logger

from .types import TensorBlobKeys


class TorchTensorsProducer(ObjectProducer):
    """TorchTensorsProducer produces stream data bytes objects from a map of torch tensors.

    Attributes:
        logger: Logger for logging messages.
        entry_timeout: Timeout for each entry in the stream.
        last: Flag indicating if the last tensor has been sent.
        tensors: Dictionary of tensors to be sent.
        tensors_keys: List of keys for the tensors to be sent.
        start: Starting index for streaming.
        current: Current index in the streaming process.
        end: Ending index for streaming.
    Methods:
        produce(stream_ctx, fl_ctx): Produces the next chunk of tensors to be sent.
        process_replies(replies, stream_ctx, fl_ctx): Processes replies from peers after sending tensors.
    """

    def __init__(self, tensors: dict[str, torch.Tensor], entry_timeout: float, root_key: str = ""):
        self.logger = get_obj_logger(self)
        self.entry_timeout = entry_timeout
        self.root_key = root_key
        self.last = False
        self.tensors = tensors
        self.tensors_keys = list(tensors.keys()) if tensors else []
        self.start = 0
        self.current = self.start
        self.end = len(self.tensors_keys)
        self.total_bytes = 0

    def produce(
        self,
        stream_ctx: StreamContext,
        fl_ctx: FLContext,
    ) -> Tuple[Shareable, float]:
        """Produce the next chunk of tensors to be sent.

        It serializes and return the next tensor using safetensors and prepares them for sending.
        Args:
            stream_ctx (StreamContext): The stream context for the current operation.
            fl_ctx (FLContext): The FL context for the current operation.
        Returns:
            Tuple[Shareable, float]: A tuple containing the shareable object with the tensor data
            and the timeout for the entry.
        Raises:
            Warning: If no tensors are found in the FLContext.
        """
        tensors = self.tensors
        if tensors is None:
            self.logger.warning("No tensors found in FLContext")
            return None, self.entry_timeout

        data = Shareable()
        key = self.tensors_keys[self.current]
        tensor = {key: self.tensors[key]}

        data[TensorBlobKeys.SAFETENSORS_BLOB] = save_tensors(tensor)
        data[TensorBlobKeys.TENSOR_KEYS] = [key]
        data[TensorBlobKeys.ROOT_KEY] = self.root_key

        self.total_bytes += len(data[TensorBlobKeys.SAFETENSORS_BLOB])
        self.last = self.current >= self.end - 1
        self.current += 1
        if self.last:
            peer_name = fl_ctx.get_peer_context().get_identity_name()
            self.logger.info(
                f"Peer '{fl_ctx.get_identity_name()}': produced blobs for peer '{peer_name}' "
                f"with {len(self.tensors_keys)} tensors from root_key='{self.root_key}' and "
                f"{round(self.total_bytes / (1024 * 1024), 2)} Mbytes."
            )
        return data, self.entry_timeout

    def process_replies(
        self,
        replies: Dict[str, Shareable],
        stream_ctx: StreamContext,
        fl_ctx: FLContext,
    ) -> Any:
        """Process replies from peers after sending tensors.

        Args:
            replies (Dict[str, Shareable]): A dictionary of replies from peers.
            stream_ctx (StreamContext): The stream context for the current operation. (not used)
            fl_ctx (FLContext): The FL context for the current operation. (not used)

        Returns:
            Any: True if all replies are successful and the last tensor has been sent,
                 False if there was an error in any reply,
                 None if more tensors need to be sent.
        """
        has_error = False
        for target, reply in replies.items():
            rc = reply.get_return_code(ReturnCode.OK)
            if rc != ReturnCode.OK:
                self.logger.error(f"error from target {target}: {rc}")
                has_error = True

        if has_error:
            # done - failed
            self.tensors = {}
            return False
        elif self.last:
            # done - succeeded
            self.tensors = {}
            return True
        else:
            # not done yet - continue streaming
            return None
