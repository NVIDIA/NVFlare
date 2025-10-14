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
        deserializer: Generator to handle deserialization of received tensor blobs.
    """

    def __init__(self, stream_ctx: StreamContext, fl_ctx: FLContext):
        """Initialize the TensorConsumer.

        Args:
            stream_ctx (StreamContext): The stream context for the current operation. (not used)
            fl_ctx (FLContext): The FL context for the current operation. (not used)
        """
        self.logger = get_obj_logger(self)
        self.tensors_map: TensorsMap = {}
        self.total_bytes: dict[str, int] = {}
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
        root_key = shareable.get(TensorBlobKeys.ROOT_KEY, "")
        task_id = shareable.get(TensorBlobKeys.TASK_ID)
        if not task_id:
            raise ValueError("Received shareable without task_id")

        self.task_ids.add(task_id)
        self.total_bytes[root_key] = self.total_bytes.get(root_key, 0) + len(tensors_blob)
        loaded_tensors = load_safetensors(tensors_blob)

        if set(tensor_keys) != set(loaded_tensors.keys()):
            raise ValueError(
                f"Mismatch in tensor keys. Expected: {tensor_keys}, Received: {list(loaded_tensors.keys())}"
            )

        if root_key not in self.tensors_map:
            self.tensors_map[root_key] = {}
        self.tensors_map[root_key].update(loaded_tensors)

    def finalize(self, stream_ctx: StreamContext, fl_ctx: FLContext):
        """Finalize the consumer, ensuring all data is written and resources are released.

        It updates the FLContext with the received tensors.

        Args:
            stream_ctx (StreamContext): The stream context. (not used)
            fl_ctx (FLContext): The FL context. (not used)
        """
        identity = fl_ctx.get_identity_name()
        peer_name = fl_ctx.get_peer_context().get_identity_name()
        root_keys = list(self.tensors_map.keys())
        if len(self.task_ids) > 1:
            raise ValueError(f"Expected one task_id, but found multiple: {self.task_ids}")

        task_id = self.task_ids.pop()
        if not task_id:
            raise ValueError("No valid task_id found in received shareables")

        for root_key in root_keys:
            tensor_keys = list(self.tensors_map[root_key].keys())
            self.log_received(task_id, identity, peer_name, root_key, tensor_keys)

        # If there's only one root key which is an empty string, it means all tensors are at the top level
        if root_keys == [""]:
            tensors = self.tensors_map[""]
        else:
            tensors = self.tensors_map

        fl_ctx.set_custom_prop(TensorCustomKeys.SAFE_TENSORS_PROP_KEY, tensors)
        fl_ctx.set_custom_prop(TensorCustomKeys.TASK_ID, task_id)

        # Clear temporary references to free memory
        self.tensors_map = {}
        self.total_bytes = {}
        del tensors

    def log_received(self, task_id: str, identity: str, peer_name: str, root_key: str, tensor_keys: list[str]):
        """Log the received tensors for debugging purposes.

        Args:
            identity (str): The identity of the peer.
            peer_name (str): The name of the peer.
            tensor_keys (list[str]): The keys of the received tensors.
        """
        total_bytes = self.total_bytes.get(root_key, 0)
        msg = (
            f"Peer '{identity}': consumed blobs from peer '{peer_name}' "
            f"with {len(tensor_keys)} tensors, total size: "
            f"{round(total_bytes / (1024 * 1024), 2)} Mbytes ({total_bytes} bytes). "
            f"Task ID: {task_id}"
        )
        if root_key:
            msg += f", root key: '{root_key}'"
        self.logger.info(msg)
