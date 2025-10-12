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
from typing import Generator

from safetensors.torch import load as load_safetensors

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.streaming import ConsumerFactory, ObjectConsumer, StreamContext
from nvflare.fuel.utils.log_utils import get_obj_logger

from .types import SAFE_TENSORS_PROP_KEY, TensorBlobKeys, TensorsMap


def tensor_deserializer_generator(
    tensors_map: TensorsMap, total_bytes: dict[str, int]
) -> Generator[None, Shareable, None]:
    """Generator that receives safetensors blobs and updates the tensors dictionary.

    Args:
        tensors_map: Empty dictionary to be populated with received tensors
        total_bytes: Dictionary to track total bytes received per root key

    Yields:
        None: Indicates ready to receive a Shareable object containing a safetensors blob

    Receives:
        shareable (Shareable): A Shareable object containing:
            - TensorBlobKeys.SAFETENSORS_BLOB: The safetensors blob as bytes
            - TensorBlobKeys.TENSOR_KEYS: List of tensor keys included in the blob
            - TensorBlobKeys.ROOT_KEY: The root key associated with the tensors
    Updates:
        tensors_map: Populated with tensors extracted from received blobs
        total_bytes: Updated with the total bytes received for each root key
    """
    while True:
        data = yield
        if data is None:
            break

        tensors_blob = data.get(TensorBlobKeys.SAFETENSORS_BLOB, b"")
        if not tensors_blob:
            raise ValueError("Received empty tensor blob")
        tensor_keys = data.get(TensorBlobKeys.TENSOR_KEYS, [])
        if not tensor_keys:
            raise ValueError("Received empty tensor keys list")
        root_key = data.get(TensorBlobKeys.ROOT_KEY, "")

        total_bytes[root_key] = total_bytes.get(root_key, 0) + len(tensors_blob)
        loaded_tensors = load_safetensors(tensors_blob)

        if set(tensor_keys) != set(loaded_tensors.keys()):
            raise ValueError(
                f"Mismatch in tensor keys. Expected: {tensor_keys}, Received: {list(loaded_tensors.keys())}"
            )

        if root_key not in tensors_map:
            tensors_map[root_key] = {}
        tensors_map[root_key].update(loaded_tensors)


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
        self.deserializer = tensor_deserializer_generator(self.tensors_map, self.total_bytes)
        next(self.deserializer)  # Prime the generator

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
            self.deserializer.send(shareable)
        except ValueError as ve:
            self.logger.error(f"Error deserializing tensors: {ve}")
            return False, make_reply(ReturnCode.ERROR, str(ve))
        except Exception as e:
            self.logger.error(f"Unexpected error deserializing tensors: {e}")
            return False, make_reply(ReturnCode.ERROR, str(e))

        return True, make_reply(ReturnCode.OK)

    def finalize(self, stream_ctx: StreamContext, fl_ctx: FLContext):
        """Finalize the consumer, ensuring all data is written and resources are released.

        It updates the FLContext with the received tensors.

        Args:
            stream_ctx (StreamContext): The stream context. (not used)
            fl_ctx (FLContext): The FL context. (not used)
        """
        # Close the generator
        try:
            self.deserializer.send(None)
        except StopIteration:
            pass  # Normal termination of the generator

        identity = fl_ctx.get_identity_name()
        peer_name = fl_ctx.get_peer_context().get_identity_name()
        root_keys = list(self.tensors_map.keys())

        for root_key in root_keys:
            tensor_keys = list(self.tensors_map[root_key].keys())
            self._log_received_tensors(identity, peer_name, root_key, tensor_keys)

        # If there's only one root key which is an empty string, it means all tensors are at the top level
        if root_keys == [""]:
            tensors = self.tensors_map[""]
        else:
            tensors = self.tensors_map

        fl_ctx.set_custom_prop(SAFE_TENSORS_PROP_KEY, tensors)

        # Clear tensors after setting them in the context
        self.tensors_map = {}

    def _log_received_tensors(self, identity: str, peer_name: str, root_key: str, tensor_keys: list[str]):
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
            f"{round(total_bytes / (1024 * 1024), 2)} Mbytes ({total_bytes} bytes)"
        )
        if root_key:
            msg += f", root key: '{root_key}'"
        self.logger.info(msg)
