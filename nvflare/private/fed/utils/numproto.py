# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

"""NumPy ndarray to protobuf serialization and deserialization."""

from io import BytesIO

import numpy as np

from nvflare.private.fed.protos.federated_pb2 import NDArray


def ndarray_to_proto(nda: np.ndarray) -> NDArray:
    """Serializes a numpy array into an NDArray protobuf message.

    Args:
        nda (np.ndarray): numpy array to serialize.

    Returns:
        Returns an NDArray protobuf message.
    """
    nda_bytes = BytesIO()
    np.save(nda_bytes, nda, allow_pickle=False)

    return NDArray(ndarray=nda_bytes.getvalue())


def proto_to_ndarray(nda_proto: NDArray) -> np.ndarray:
    """Deserializes an NDArray protobuf message into a numpy array.

    Args:
        nda_proto (NDArray): NDArray protobuf message to deserialize.

    Returns:
        Returns a numpy.ndarray.
    """
    nda_bytes = BytesIO(nda_proto.ndarray)

    return np.load(nda_bytes, allow_pickle=False)


def bytes_to_proto(data: bytes) -> NDArray:
    """Serializes a bytes into an NDArray protobuf message.

    Args:
        data : bytes data

    Returns:
        Returns an NDArray protobuf message.
    """
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes but got {}".format(type(data)))
    return NDArray(ndarray=data)


def proto_to_bytes(nda_proto: NDArray) -> bytes:
    """Deserializes an NDArray protobuf message into bytes.

    Args:
        nda_proto (NDArray): bytes.

    Returns:
        Returns bytes.
    """
    nda_bytes = BytesIO(nda_proto.ndarray)

    return nda_bytes.read()
