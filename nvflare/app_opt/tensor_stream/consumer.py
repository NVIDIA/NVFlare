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

from .types import SAFE_TENSORS_PROP_KEY, TensorBlobKeys, TensorsMap


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
        tensors: Dictionary to store received tensors.
    """

    def __init__(self, stream_ctx: StreamContext, fl_ctx: FLContext):
        """Initialize the TensorConsumer.

        Args:
            stream_ctx (StreamContext): The stream context for the current operation. (not used)
            fl_ctx (FLContext): The FL context for the current operation. (not used)
        """
        self.logger = get_obj_logger(self)
        self.tensors: TensorsMap = {}
        self.total_bytes = {}
        self.root_keys = []

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
        tensor_blob = shareable.get(TensorBlobKeys.SAFETENSORS_BLOB, b"")
        if not tensor_blob:
            self.logger.error("No tensor blob found in the shareable.")
            return False, make_reply(ReturnCode.ERROR)

        tensor_keys = shareable.get(TensorBlobKeys.TENSOR_KEYS, [])
        if not tensor_keys:
            self.logger.error("No tensor keys found in the shareable.")
            return False, make_reply(ReturnCode.ERROR)

        root_key = shareable.get(TensorBlobKeys.ROOT_KEY, None)
        if root_key is None:
            self.logger.error("No root key found in the shareable.")
            return False, make_reply(ReturnCode.ERROR)

        # Update total bytes received for this root key
        self.total_bytes[root_key] = self.total_bytes.get(root_key, 0) + len(tensor_blob)
        tensor = load_safetensors(tensor_blob)
        if set(tensor_keys) != set(tensor.keys()):
            self.logger.error(f"Mismatch in tensor keys. Expected: {tensor_keys}, Received: {list(tensor.keys())}")
            return False, make_reply(ReturnCode.ERROR)

        # handle tensors stores in root key, i.e. inside "state_dict" or without root key which is ""
        if root_key not in self.tensors:
            self.tensors[root_key] = {}
        self.tensors[root_key].update(tensor)
        if root_key not in self.root_keys:
            self.root_keys.append(root_key)

        return True, make_reply(ReturnCode.OK)

    def finalize(self, stream_ctx: StreamContext, fl_ctx: FLContext):
        """Finalize the consumer, ensuring all data is written and resources are released.

        It updates the FLContext with the received tensors.

        Args:
            stream_ctx (StreamContext): The stream context. (not used)
            fl_ctx (FLContext): The FL context. (not used)
        """
        identity = fl_ctx.get_identity_name()
        peer_name = fl_ctx.get_peer_context().get_identity_name()

        for root_key in self.root_keys:
            tensor_keys = list(self.tensors[root_key].keys())
            self._log_received_tensors(identity, peer_name, root_key, tensor_keys)

        # If there's only one root key which is an empty string, it means all tensors are at the top level
        if self.root_keys == [""]:
            tensors = self.tensors[""]
        else:
            tensors = self.tensors

        fl_ctx.set_custom_prop(SAFE_TENSORS_PROP_KEY, tensors)

        # Clear tensors after setting them in the context
        self.tensors = {}

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
