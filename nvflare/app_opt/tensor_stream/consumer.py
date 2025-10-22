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

from safetensors.torch import load as load_safetensors

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.streaming import ConsumerFactory, ObjectConsumer, StreamContext
from nvflare.fuel.utils.log_utils import get_obj_logger

from .types import TensorBlobKeys, TensorCustomKeys, TensorsMap
from .utils import update_params_with_tensors


class TensorConsumerFactory(ConsumerFactory):
    """Factory for creating TensorConsumer instances.

    Methods:
        get_consumer(stream_ctx, fl_ctx): Creates and returns a TensorConsumer instance.
    """

    def get_consumer(self, stream_ctx: StreamContext, fl_ctx: FLContext) -> ObjectConsumer:
        return TensorConsumer(stream_ctx, fl_ctx)


class TensorConsumer(ObjectConsumer):
    """TensorConsumer handles receiving and reconstructing torch tensors from a stream of byte objects.

    Attributes:
        logger: Logger for logging messages.
        tensors_map: Dictionary to store received tensors.
        total_bytes: Dictionary to track total bytes received per root key.
        num_tensors: Dictionary to track number of tensors received per root key.
        task_ids: Set to track unique task IDs received.
    """

    def __init__(self, stream_ctx: StreamContext, fl_ctx: FLContext):
        """Initialize the TensorConsumer.

        Args:
            stream_ctx (StreamContext): The stream context for the current operation. (not used)
            fl_ctx (FLContext): The FL context for the current operation. (not used)
        """
        self.logger = get_obj_logger(self)
        self.params: TensorsMap = {}
        self.total_bytes: int = 0
        self.num_tensors: int = 0
        self.task_ids: set[str] = set()

    def consume(
        self,
        shareable: Shareable,
        stream_ctx: StreamContext,
        fl_ctx: FLContext,
    ) -> tuple[bool, Shareable]:
        """Consume a shareable object and extract tensors.

        Args:
            shareable (Shareable): The shareable object containing tensor data.
            stream_ctx (StreamContext): The stream context for the current operation. (not used)
            fl_ctx (FLContext): The FL context for the current operation. (not used)

        Returns:
            tuple[bool, Shareable]: A tuple containing a success flag and a reply shareable.
        """
        try:
            self.process_shareable(shareable)
        except ValueError as ve:
            self.logger.error(f"Error deserializing tensors: {ve}")
            return False, make_reply(ReturnCode.ERROR, str(ve))
        except Exception as e:
            self.logger.error(f"Unexpected error deserializing tensors: {e}")
            return False, make_reply(ReturnCode.ERROR, str(e))

        return True, make_reply(ReturnCode.OK)

    def process_shareable(
        self,
        shareable: Shareable,
    ):
        """Process a received shareable object containing tensor data.

        Args:
            shareable (Shareable): The shareable object containing tensor data.

        Raises:
            ValueError: If the shareable object is invalid or contains errors.
        """
        tensors_blob = shareable.get(TensorBlobKeys.SAFETENSORS_BLOB, b"")
        if not tensors_blob:
            raise ValueError("Received empty tensor blob")
        tensor_keys = shareable.get(TensorBlobKeys.TENSOR_KEYS, [])
        if not tensor_keys:
            raise ValueError("Received empty tensor keys list")
        task_id = shareable.get(TensorBlobKeys.TASK_ID)
        if not task_id:
            raise ValueError("Received shareable without task_id")
        parent_keys = shareable.get(TensorBlobKeys.PARENT_KEYS, None)
        if parent_keys is None:
            raise ValueError("Received shareable without parent_keys")

        loaded_tensors = load_safetensors(tensors_blob)
        received_keys = list(loaded_tensors.keys())
        if set(tensor_keys) != set(loaded_tensors.keys()):
            raise ValueError(f"Mismatch in tensor keys. Expected: {tensor_keys}, Received: {received_keys}")

        self.task_ids.add(task_id)
        self.total_bytes += len(tensors_blob)
        self.num_tensors += len(tensor_keys)

        # at this point we don't care about the to_ndarray conversion
        update_params_with_tensors(self.params, parent_keys, loaded_tensors)

        # Clean up temporary references to free memory
        del tensors_blob
        del loaded_tensors

    def finalize(self, stream_ctx: StreamContext, fl_ctx: FLContext):
        """Finalize the consumer, ensuring all data is written and resources are released.

        It updates the FLContext with the received tensors.

        Args:
            stream_ctx (StreamContext): The stream context. (not used)
            fl_ctx (FLContext): The FL context. (not used)
        """
        identity = fl_ctx.get_identity_name()
        peer_name = fl_ctx.get_peer_context().get_identity_name()

        if len(self.task_ids) == 0:
            raise ValueError("No valid task_id found in received shareables")

        if len(self.task_ids) > 1:
            raise ValueError(f"Expected one task_id, but found multiple: {self.task_ids}")

        task_id = self.task_ids.pop()
        if not task_id:
            raise ValueError("Invalid task_id (empty or None) found in received shareables")

        fl_ctx.set_custom_prop(TensorCustomKeys.SAFE_TENSORS_PROP_KEY, self.params)
        fl_ctx.set_custom_prop(TensorCustomKeys.TASK_ID, task_id)

        # Clear temporary references to free memory
        self.params = {}
        self.log_received(task_id, identity, peer_name)

    def log_received(self, task_id: str, identity: str, peer_name: str):
        """Log the received tensors for debugging purposes.

        Args:
            identity (str): The identity of the peer.
            peer_name (str): The name of the peer.
            tensor_keys (list[str]): The keys of the received tensors.
        """
        total_bytes = self.total_bytes
        num_tensors = self.num_tensors
        msg = (
            f"Peer '{identity}': consumed blobs from peer '{peer_name}' "
            f"with {num_tensors} tensors, total size: "
            f"{round(total_bytes / (1024 * 1024), 2)} Mbytes ({total_bytes} bytes). "
            f"Task ID: {task_id}"
        )
        self.logger.info(msg)
