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
from .utils import chunk_tensors_from_params


class TensorProducer(ObjectProducer):
    """TensorProducer produces stream data bytes objects from a map of torch tensors.

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

    def __init__(self, tensors: dict[str, torch.Tensor], task_id: str, entry_timeout: float):
        """Initialize the TensorProducer.

        Args:
            tensors (dict): A dictionary of tensors to be sent.
            task_id (str): The task ID associated with the tensors.
            entry_timeout (float): The timeout for each entry in the stream.
        Raises:
            ValueError: If no tensors are provided.
        """
        if tensors is None:
            raise ValueError("No tensors received for serialization.")
        self.task_id = task_id
        self.entry_timeout = entry_timeout
        self.last = False
        self.total_bytes = 0
        self.num_tensors = 0
        # Pass tensors to generator; they're not stored in this class to minimize memory usage.
        # The generator will handle serialization and the tensors can be garbage collected
        # after the generator completes.
        self.chunks_generator = chunk_tensors_from_params(tensors)
        self.tensors_keys = list(tensors.keys())
        self.logger = get_obj_logger(self)

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
        data = Shareable()
        try:
            parent_keys, tensors = next(self.chunks_generator, (None, None))
        except StopIteration:
            return None, self.entry_timeout
        else:
            if tensors is None:
                self.last = True
                self.log_completion(fl_ctx)
                return None, self.entry_timeout

            tensors_blob = save_tensors(tensors)
            data[TensorBlobKeys.SAFETENSORS_BLOB] = tensors_blob
            data[TensorBlobKeys.TENSOR_KEYS] = list(tensors.keys())
            data[TensorBlobKeys.PARENT_KEYS] = parent_keys
            data[TensorBlobKeys.TASK_ID] = self.task_id
            self.total_bytes += len(tensors_blob)
            self.num_tensors += len(tensors)

        return data, self.entry_timeout

    def log_completion(self, fl_ctx: FLContext):
        peer_name = fl_ctx.get_peer_context().get_identity_name()
        msg = (
            f"Peer '{fl_ctx.get_identity_name()}': produced blobs for peer '{peer_name}' "
            f"with {self.num_tensors} tensors, total size: "
            f"{round(self.total_bytes / (1024 * 1024), 2)} Mbytes ({self.total_bytes} bytes). "
            f"Task ID: {self.task_id}"
        )
        self.logger.info(msg)

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
            del self.chunks_generator  # free memory
            return False
        elif self.last:
            # done - succeeded
            del self.chunks_generator  # free memory
            return True
        else:
            # not done yet - continue streaming
            return None
